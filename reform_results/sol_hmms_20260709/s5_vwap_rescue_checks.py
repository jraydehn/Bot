"""
S5 -- Two questions before wiring the VWAP-1h gates:

A) Do VWAP-1h HMM states rescue either of today's new gates' buckets?
   (i)  sol_15m_no_zdrift_gate bucket (559 NO trades, z<0.55) -- the ~4,600-
        subset search predates this model, so this is genuinely untested.
   (ii) sol_15m_cg_liq_yes_gate bucket (140 YES trades, CG S4).
   Rescue bar: ep-clustered P(edge<=0)<=0.05, n>=60, zero streak leak (for i).

B) Mandatory rescue search on the S2-NO hourly block bucket (n=49/28 eps)
   before implementing it as a pure block: sweep all usable logged columns of
   the hourly archive book + the CG flow states. Disclosed as under-powered
   at this n; bar: P<=0.05, n>=15, remainder>=15.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1103)
OUT = "reform_results/sol_hmms_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

sv = pd.read_csv(f"{OUT}/vwap_sol_1h_states.csv")
sv["effective"] = pd.to_datetime(sv["effective"], utc=True)
sv = sv.sort_values("effective")


def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 6:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()


# ── A(i): zdrift bucket ───────────────────────────────────────────────────
no = pd.read_csv("reform_results/sol15m_streak_20260709/no_book_reconstructed.csv", low_memory=False)
no["logged_at_p"] = pd.to_datetime(no["logged_at_p"], utc=True)
no["week"] = no["logged_at_p"].dt.to_period("W-FRI").astype(str)
bucket = no[(no["z_drift_6h"] < 0.55).fillna(False)].sort_values("logged_at_p")
bucket = pd.merge_asof(bucket, sv[["effective", "state"]].rename(columns={"state": "vwap1h"}),
                       left_on="logged_at_p", right_on="effective",
                       direction="backward", tolerance=pd.Timedelta("2h"))
print(f"=== A(i): VWAP-1h states within the zdrift NO bucket (n={len(bucket)}, "
      f"coverage {bucket['vwap1h'].notna().sum()}) ===")
smask = bucket["contract_ticker"].isin(STREAK)
for s in sorted(bucket["vwap1h"].dropna().unique()):
    d = bucket[bucket["vwap1h"] == s]
    if len(d) < 25:
        print(f"  S{int(s)}: n={len(d)} thin")
        continue
    ne, ee, pn = ep_stats(d)
    wk = d.groupby("week")["tedge"].mean()
    leak = int(((bucket["vwap1h"] == s) & smask).sum())
    print(f"  S{int(s)}: n={len(d)} eps={ne} ep_edge={ee:+.4f} P(<=0)={pn:.4f} "
          f"wk+={int((wk>0).sum())}/{len(wk)} streak_leak={leak} $={d['would_pnl'].sum():+.2f}")

# ── A(ii): CG S4 YES bucket ───────────────────────────────────────────────
pt = pd.read_csv("results/paper_trades_sol15m.csv", low_memory=False)
pt["logged_at_p"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["resolved_yes", "p_market", "would_pnl"]:
    pt[c] = pd.to_numeric(pt[c], errors="coerce")
t = pt[(pt["decision"] == "trade") & (pt["side"].str.lower() == "yes")].dropna(
    subset=["resolved_yes", "logged_at_p", "p_market"]).sort_values("logged_at_p").copy()
t["won"] = t["resolved_yes"] == 1
t["be"] = t["p_market"]
t["tedge"] = t["won"].astype(float) - t["be"]
gaps = t["logged_at_p"].diff().dt.total_seconds() / 60
t["episode"] = (gaps > 45).cumsum()
t["week"] = t["logged_at_p"].dt.to_period("W-FRI").astype(str)
cgs = pd.read_csv(f"{OUT}/cg_flow_sol_states.csv")
cgs["effective"] = pd.to_datetime(cgs["effective"], utc=True)
t = pd.merge_asof(t, cgs[["effective", "state"]].rename(columns={"state": "cgstate", "effective": "eff1"}),
                  left_on="logged_at_p", right_on="eff1", direction="backward", tolerance=pd.Timedelta("2h"))
s4b = t[t["cgstate"] == 4].sort_values("logged_at_p")
s4b = pd.merge_asof(s4b, sv[["effective", "state"]].rename(columns={"state": "vwap1h"}),
                    left_on="logged_at_p", right_on="effective",
                    direction="backward", tolerance=pd.Timedelta("2h"))
print(f"\n=== A(ii): VWAP-1h states within the CG-S4 YES bucket (n={len(s4b)}, "
      f"coverage {s4b['vwap1h'].notna().sum()}) ===")
for s in sorted(s4b["vwap1h"].dropna().unique()):
    d = s4b[s4b["vwap1h"] == s]
    if len(d) < 15:
        print(f"  S{int(s)}: n={len(d)} thin")
        continue
    ne, ee, pn = ep_stats(d)
    print(f"  S{int(s)}: n={len(d)} eps={ne} ep_edge={ee:+.4f} P(<=0)={pn:.4f} $={d['would_pnl'].sum():+.2f}")

# ── B: rescue search on the hourly S2-NO block bucket ────────────────────
print(f"\n=== B: rescue search, hourly S2-NO bucket ===")
frames = []
for p in ["results/paper_trades_sol_archive_20260707_2013_pre_contrarian_ls_gate.csv",
          "results/paper_trades_sol.csv"]:
    frames.append(pd.read_csv(p, low_memory=False))
raw = pd.concat(frames, ignore_index=True)
raw["logged_at_p"] = pd.to_datetime(raw["logged_at"], format="mixed", utc=True, errors="coerce")
raw = raw.drop_duplicates(subset=["logged_at_p", "contract_ticker"], keep="first")
for c in ["resolved_yes", "p_market", "would_pnl"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
h = raw[(raw["decision"] == "trade") & (raw["side"].str.lower() == "no")].dropna(
    subset=["resolved_yes", "logged_at_p", "p_market"]).sort_values("logged_at_p").copy()
h["won"] = h["resolved_yes"] == 0
h["be"] = 1 - h["p_market"]
h["tedge"] = h["won"].astype(float) - h["be"]
gaps = h["logged_at_p"].diff().dt.total_seconds() / 60
h["episode"] = (gaps > 90).cumsum()
h["week"] = h["logged_at_p"].dt.to_period("W-FRI").astype(str)
h = pd.merge_asof(h, sv[["effective", "state"]].rename(columns={"state": "vwap1h"}),
                  left_on="logged_at_p", right_on="effective",
                  direction="backward", tolerance=pd.Timedelta("2h"))
s2 = h[h["vwap1h"] == 2].copy()
print(f"S2-NO bucket: n={len(s2)}  eps={s2['episode'].nunique()}  "
      f"edge={s2['tedge'].mean():+.4f}  $={s2['would_pnl'].sum():+.2f}")

EXCLUDE = {"logged_at", "logged_at_p", "decision_time", "asset", "contract_ticker", "close_time",
           "close_ts", "side", "decision", "resolved_yes", "would_win", "would_pnl",
           "spot_at_expiry", "price_move_pct", "miss_pct", "won", "be", "tedge", "episode",
           "week", "kelly_fraction", "bet_fraction", "bet_amount", "bankroll", "spot",
           "floor_strike", "strike", "vwap1h", "effective", "eff1"}
n_tests, found = 0, []
for feat in s2.columns:
    if feat in EXCLUDE:
        continue
    col = pd.to_numeric(s2[feat], errors="coerce")
    if col.notna().sum() < 30:
        continue
    if col.dropna().nunique() <= 6:
        vals = col.dropna().unique()
        splits = [(f"== {v}", col == v) for v in vals]
    else:
        vv = col.dropna()
        splits = []
        for q in [0.25, 0.5, 0.75]:
            th = vv.quantile(q)
            splits += [(f">= {th:.4g}", col >= th), (f"< {th:.4g}", col < th)]
    for lab, mk in splits:
        n_tests += 1
        d = s2[mk.fillna(False)]
        if len(d) < 15 or len(s2) - len(d) < 15:
            continue
        if d["tedge"].mean() < 0.02:
            continue
        ne, ee, pn = ep_stats(d)
        found.append({"feature": feat, "split": lab, "n": len(d), "eps": ne,
                      "ep_edge": ee, "P_neg": pn, "pnl": d["would_pnl"].sum()})
print(f"{n_tests} splits tested (DISCLOSED: n=49 bucket -> under-powered; min cell 15)")
fd = pd.DataFrame(found)
if len(fd):
    surv = fd[fd["P_neg"] <= 0.05]
    print(f"survivors (P<=0.05): {len(surv)}")
    print(fd.sort_values("P_neg").head(10).round(4).to_string(index=False))
else:
    print("no positive-edge subsets at all")
print("DONE_S5")
