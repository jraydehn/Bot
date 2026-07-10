"""
S2 -- Test the user's cross-timeframe idea: should the BTC HOURLY model
require the BTC 15m model's directional agreement before entering?

Operationalization:
- 15m-model direction series: per 15m scan cycle (btc_scan_archive_15m.csv),
  take near-ATM rows (|offset_pct| <= 0.15%) and average (edge_yes - edge_no)
  -> dir15 = +1 (bull) / -1 (bear) / 0 (mixed, |mean| < 0.01). Logged at scan
  time = causal by construction.
- Join each real hourly taken trade (backward asof, 20min tolerance).
- agree  = (hourly YES & dir15=+1) or (hourly NO & dir15=-1)
- disagree = opposite; neutral = dir15=0.
- "Wait until agreement" == block-on-disagree (runner rescans ~75s).
Ticker-clustered bootstrap, weekly stability, side split, era split.
SIGN IS AN EMPIRICAL QUESTION: ETH/SOL agreement gates validated CONTRARIAN.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1151)
OUT = "reform_results/btc_vwap_fresh_20260709"


def parse_mixed(s):
    def _u(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_u(v) for v in s], utc=True)


# ── 15m model direction series ────────────────────────────────────────────
arch = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                   usecols=["logged_at", "p_market", "p_model_yes", "p_model_no", "offset_pct"])
for c in ["p_market", "p_model_yes", "p_model_no", "offset_pct"]:
    arch[c] = pd.to_numeric(arch[c], errors="coerce")
arch["ts"] = parse_mixed(arch["logged_at"])
arch = arch.dropna(subset=["ts", "p_market", "p_model_yes", "p_model_no", "offset_pct"])
near = arch[arch["offset_pct"].abs() <= 0.15].copy()
near["d"] = (near["p_model_yes"] - near["p_market"]) - (near["p_model_no"] - (1 - near["p_market"]))
cyc = near.groupby("ts")["d"].mean().reset_index().sort_values("ts")
cyc["dir15"] = np.where(cyc["d"] > 0.01, 1, np.where(cyc["d"] < -0.01, -1, 0))
print(f"15m direction series: {len(cyc)} scan cycles  {cyc['ts'].min()} -> {cyc['ts'].max()}")
print(f"dir15 distribution: {cyc['dir15'].value_counts().to_dict()}")

# ── hourly taken book ─────────────────────────────────────────────────────
h = pd.read_csv("results/paper_trades.csv", low_memory=False)
h["logged_at_p"] = pd.to_datetime(h["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["resolved_yes", "p_market", "would_pnl", "bet_amount"]:
    h[c] = pd.to_numeric(h[c], errors="coerce")
t = h[(h["bet_amount"] > 0)].dropna(subset=["resolved_yes", "logged_at_p", "p_market"]).copy()
t["side"] = t["side"].str.lower()
t["won"] = np.where(t["side"] == "yes", t["resolved_yes"] == 1, t["resolved_yes"] == 0)
t["be"] = np.where(t["side"] == "yes", t["p_market"], 1 - t["p_market"])
t["tedge"] = t["won"].astype(float) - t["be"]
t = t.sort_values("logged_at_p")
t = pd.merge_asof(t, cyc[["ts", "dir15", "d"]], left_on="logged_at_p", right_on="ts",
                  direction="backward", tolerance=pd.Timedelta("20min"))
t = t.dropna(subset=["dir15"])
t["week"] = t["logged_at_p"].dt.to_period("W-FRI").astype(str)
t["grp"] = np.where(((t["side"] == "yes") & (t["dir15"] == 1)) | ((t["side"] == "no") & (t["dir15"] == -1)),
                    "AGREE",
                    np.where(t["dir15"] == 0, "NEUTRAL", "DISAGREE"))
print(f"\nhourly taken trades with 15m direction: {len(t)}  "
      f"({t['logged_at_p'].min().date()} -> {t['logged_at_p'].max().date()})")
print(f"group sizes: {t['grp'].value_counts().to_dict()}")


def tk_boot(d, n_boot=4000):
    pt = d.groupby("contract_ticker")["tedge"].mean()
    e = pt.values
    if len(e) < 10:
        return len(e), np.nan, np.nan
    means = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)])
    return len(e), means.mean(), (means <= 0).mean()


print(f"\n=== agreement groups, full book ===")
for g in ["AGREE", "NEUTRAL", "DISAGREE"]:
    d = t[t["grp"] == g]
    if len(d) < 15:
        print(f"{g}: n={len(d)} thin")
        continue
    nt, ee, pn = tk_boot(d)
    wk = d.groupby("week")["tedge"].mean()
    print(f"{g}: n={len(d)} tickers={nt} WR={d['won'].mean():.3f} BE={d['be'].mean():.3f} "
          f"tk_edge={ee:+.4f} P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} "
          f"$={d['would_pnl'].sum():+.2f}")

print(f"\n=== by hourly side ===")
for side in ["yes", "no"]:
    for g in ["AGREE", "NEUTRAL", "DISAGREE"]:
        d = t[(t["grp"] == g) & (t["side"] == side)]
        if len(d) < 15:
            continue
        nt, ee, pn = tk_boot(d)
        wk = d.groupby("week")["tedge"].mean()
        print(f"{side.upper():3s} {g:<9}: n={len(d)} tickers={nt} tk_edge={ee:+.4f} "
              f"P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} $={d['would_pnl'].sum():+.2f}")

print(f"\n=== era split (pre/post 07-06 rollback) ===")
for lbl, m in [("pre-07-06", t["logged_at_p"] < pd.Timestamp("2026-07-06", tz="UTC")),
               ("post-07-06", t["logged_at_p"] >= pd.Timestamp("2026-07-06", tz="UTC"))]:
    for g in ["AGREE", "DISAGREE"]:
        d = t[m & (t["grp"] == g)]
        if len(d) < 15:
            print(f"{lbl} {g}: n={len(d)} thin")
            continue
        nt, ee, pn = tk_boot(d)
        print(f"{lbl} {g}: n={len(d)} tickers={nt} tk_edge={ee:+.4f} P(<=0)={pn:.4f} "
              f"$={d['would_pnl'].sum():+.2f}")

# strength-of-signal gradient: does deeper disagreement hurt more?
print(f"\n=== disagreement-strength gradient (signed d vs hourly side) ===")
t["signed_d"] = np.where(t["side"] == "yes", t["d"], -t["d"])   # + = 15m supports the side
for lab, m in [("strong support (d>+0.05)", t["signed_d"] > 0.05),
               ("mild support (0..0.05]", (t["signed_d"] > 0) & (t["signed_d"] <= 0.05)),
               ("mild oppose [-0.05..0)", (t["signed_d"] <= 0) & (t["signed_d"] > -0.05)),
               ("strong oppose (d<-0.05)", t["signed_d"] <= -0.05)]:
    d = t[m]
    if len(d) < 15:
        print(f"  {lab}: n={len(d)} thin")
        continue
    nt, ee, pn = tk_boot(d)
    print(f"  {lab}: n={len(d)} tickers={nt} tk_edge={ee:+.4f} P(<=0)={pn:.4f} "
          f"$={d['would_pnl'].sum():+.2f}")
t.to_csv(f"{OUT}/hourly_15m_agreement.csv", index=False)
print("DONE_S2")
