"""
S2 -- Validate the two candidates from the sweep:

A) z_drift_6h < ~0.55 -> block SOL 15m NO (catches all 4 streak trades).
   Checks: fine threshold curve (plateau vs spike), weekly $ + edge, era
   stability (pre/post 06-24 gate-set change), overlap vs the oversold
   cluster, rescue search WITHIN the bucket, complement health, YES-side
   mirror (is this drift-model bias?).

B) The existing sol_15m_no_stoch_oversold_gate's chg_5m<-0.20 rescue:
   post-06-24, every taken NO with stoch_k_15m<20 is by construction a
   rescued trade -- measure whether the rescue population actually wins
   (original claim WR=54.3%) or leaks losses.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1013)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

no = pd.read_csv(f"{OUT}/no_book_reconstructed.csv", low_memory=False)
no["logged_at_p"] = pd.to_datetime(no["logged_at_p"], utc=True)
no["week"] = no["logged_at_p"].dt.to_period("W-FRI").astype(str)
print(f"NO book: {len(no)}  baseline edge={no['tedge'].mean():+.4f}  $={no['would_pnl'].sum():+.2f}")


def ep_boot(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means >= 0).mean()


# ── A1: z_drift_6h fine threshold curve ──────────────────────────────────
print("\n=== A1: z_drift_6h threshold curve (block NO when z_drift_6h < thr) ===")
print(f"{'thr':>5} {'n':>5} {'eps':>5} {'edge':>8} {'ep_edge':>8} {'P_pos':>7} {'$':>10} {'streak':>7}")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    m = (no["z_drift_6h"] < thr).fillna(False)
    d = no[m]
    if len(d) < 30:
        continue
    ne, ee, pp = ep_boot(d)
    sh = int(d["contract_ticker"].isin(STREAK).sum())
    print(f"{thr:>5.2f} {len(d):>5} {ne:>5} {d['tedge'].mean():>+8.4f} {ee:>+8.4f} {pp:>7.4f} "
          f"{d['would_pnl'].sum():>+10.2f} {sh:>4}/4")

# ── A2: weekly + era stability at thr=0.55 ───────────────────────────────
THR = 0.55
m = (no["z_drift_6h"] < THR).fillna(False)
d = no[m]
print(f"\n=== A2: z_drift_6h<{THR}: weekly breakdown (n={len(d)}) ===")
wk = d.groupby("week").agg(n=("tedge", "size"), edge=("tedge", "mean"), pnl=("would_pnl", "sum"))
print(wk.round(4).to_string())
cov = no.groupby("week")["z_drift_6h"].apply(lambda s: s.notna().mean())
print("\nz_drift_6h coverage by week (fraction non-null):")
print(cov.round(3).to_string())

for lbl, mask_era in [("pre-06-24 era", no["logged_at_p"] < pd.Timestamp("2026-06-24", tz="UTC")),
                      ("post-06-24 era (current gate set)", no["logged_at_p"] >= pd.Timestamp("2026-06-24", tz="UTC"))]:
    de = no[m & mask_era]
    if len(de) < 20:
        print(f"{lbl}: n={len(de)} thin")
        continue
    ne, ee, pp = ep_boot(de)
    print(f"{lbl}: n={len(de)} eps={ne} edge={de['tedge'].mean():+.4f} ep_edge={ee:+.4f} "
          f"P_pos={pp:.4f} $={de['would_pnl'].sum():+.2f}")

# complement health
comp = no[~m & no["z_drift_6h"].notna()]
ne, ee, pp = ep_boot(comp)
print(f"\ncomplement (z_drift>= {THR}): n={len(comp)} edge={comp['tedge'].mean():+.4f} "
      f"ep_edge={ee:+.4f} $={comp['would_pnl'].sum():+.2f}")

# ── A3: overlap with the oversold cluster ────────────────────────────────
oversold = (no["stoch_k_15m"] < 21).fillna(False)
inter = (m & oversold).sum()
print(f"\n=== A3: overlap: z_drift<{THR} n={m.sum()}, stoch15m<21 n={oversold.sum()}, "
      f"intersection={inter} (jaccard={inter/max((m|oversold).sum(),1):.2f}) ===")

# ── A4: rescue search WITHIN z_drift bucket ──────────────────────────────
print(f"\n=== A4: rescue search within z_drift<{THR} bucket (positive-edge subsets) ===")
bucket = no[m].copy()
RESCUE_CANDS = [c for c in bucket.columns if (
    c.startswith("r_") or c in ("stoch_k_15m", "stoch_k_5m", "stoch_k_1h", "chg_5m", "chg_15m",
    "chg_1h", "vwap_dist", "bp_5m", "bp_15m", "bp_1h", "kalman_velocity", "kalman_residual",
    "hurst_exponent", "ou_theta", "cvd_4h", "cg_futures_delta_4h", "cg_futures_ratio_4h",
    "ls_long_pct", "liq_score", "fear_greed", "p_market", "offset_pct", "tau_minutes",
    "ema_bias", "ema_bias_1h", "consec_dir_1h", "mu6h", "mu12h", "mu24h", "regime_z"))]
resc_rows = []
for feat in RESCUE_CANDS:
    col = pd.to_numeric(bucket[feat], errors="coerce")
    if col.notna().sum() < 150 or col.dropna().nunique() < 5:
        continue
    for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        th = col.quantile(q)
        for lab, mk in [(f">= {th:.4g}", col >= th), (f"< {th:.4g}", col < th)]:
            s = bucket[mk.fillna(False)]
            if len(s) < 60 or len(bucket) - len(s) < 60:
                continue
            if s["tedge"].mean() < 0.01:
                continue
            ne, ee, pp = ep_boot(s)   # for a rescue we want P(edge<=0) small -> report 1-P_pos... careful
            eps = s.groupby("episode")["tedge"].mean().values
            means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(3000)])
            p_neg = (means <= 0).mean()
            sh = int(s["contract_ticker"].isin(STREAK).sum())
            resc_rows.append({"feature": feat, "split": lab, "n": len(s), "eps": len(eps),
                              "edge": s["tedge"].mean(), "ep_edge": means.mean(),
                              "P_neg": p_neg, "streak_leak": sh, "pnl": s["would_pnl"].sum()})
rf = pd.DataFrame(resc_rows)
if len(rf):
    good = rf[(rf["P_neg"] <= 0.10) & (rf["streak_leak"] == 0)].sort_values("ep_edge", ascending=False)
    print(f"rescue candidates (P(edge<=0)<=0.10, zero streak leak): {len(good)}")
    print(good.head(12).round(4).to_string(index=False))
else:
    print("none found")

# ── A5: YES-side mirror ───────────────────────────────────────────────────
pt = pd.read_csv("results/paper_trades_sol15m.csv", low_memory=False)
pt["logged_at_p"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["resolved_yes", "p_market", "would_pnl", "z_drift_6h"]:
    pt[c] = pd.to_numeric(pt[c], errors="coerce")
yes = pt[(pt["decision"] == "trade") & (pt["side"].str.lower() == "yes")].dropna(subset=["resolved_yes"]).copy()
yes["won"] = yes["resolved_yes"] == 1
yes["be"] = yes["p_market"]
yes["tedge"] = yes["won"].astype(float) - yes["be"]
ylow = yes[yes["z_drift_6h"] < THR]
yhigh = yes[yes["z_drift_6h"] >= THR]
print(f"\n=== A5: YES book mirror ===")
print(f"YES z_drift<{THR}:  n={len(ylow)}  edge={ylow['tedge'].mean():+.4f}  $={ylow['would_pnl'].sum():+.2f}")
print(f"YES z_drift>={THR}: n={len(yhigh)} edge={yhigh['tedge'].mean():+.4f}  $={yhigh['would_pnl'].sum():+.2f}")

# ── B: the oversold gate's rescue population ──────────────────────────────
print(f"\n=== B: sol_15m_no_stoch_oversold_gate rescue leak-through ===")
post = no[no["logged_at_p"] >= pd.Timestamp("2026-06-24", tz="UTC")]
osold = post[(post["stoch_k_15m"] < 20).fillna(False)]
print(f"post-06-24 taken NO with stoch_k_15m<20 (should all be chg_5m rescues): n={len(osold)}")
if len(osold):
    print(f"chg_5m distribution: min={osold['chg_5m'].min():+.3f} max={osold['chg_5m'].max():+.3f} "
          f"(all should be < -0.20)")
    frac_rescued = (osold["chg_5m"] < -0.20).mean()
    print(f"fraction with chg_5m<-0.20: {frac_rescued:.2f}")
    ne, ee, pp = ep_boot(osold)
    print(f"rescued population: WR={osold['won'].mean():.3f} BE={osold['be'].mean():.3f} "
          f"edge={osold['tedge'].mean():+.4f} ep_edge={ee:+.4f} P_pos={pp:.4f} $={osold['would_pnl'].sum():+.2f}")
    wk2 = osold.copy(); wk2["week"] = wk2["logged_at_p"].dt.to_period("W-FRI").astype(str)
    print(wk2.groupby("week").agg(n=("tedge","size"), edge=("tedge","mean"), pnl=("would_pnl","sum")).round(4).to_string())
# also the band [20,30) which the gate doesn't touch
band = post[(post["stoch_k_15m"] >= 20) & (post["stoch_k_15m"] < 30)]
if len(band) >= 20:
    ne, ee, pp = ep_boot(band)
    print(f"\nband stoch[20,30) post-06-24 (untouched by gate): n={len(band)} "
          f"edge={band['tedge'].mean():+.4f} ep_edge={ee:+.4f} P_pos={pp:.4f} $={band['would_pnl'].sum():+.2f}")
print("DONE_S2")
