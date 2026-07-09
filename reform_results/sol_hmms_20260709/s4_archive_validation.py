"""
S4 -- Validate the SOL hourly VWAP MTF HMM (and, as a bonus, the SOL CG flow
HMM) against the PRE-RESET hourly archive: paper_trades_sol_archive_20260707
(559 taken trades, 04-15 -> 07-07) + the current post-reset file, deduped.
Joins use the causal effective-time state series decoded in s2/s3.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1091)
OUT = "reform_results/sol_hmms_20260709"

frames = []
for p in ["results/paper_trades_sol_archive_20260707_2013_pre_contrarian_ls_gate.csv",
          "results/paper_trades_sol.csv"]:
    df = pd.read_csv(p, low_memory=False)
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
raw["logged_at_p"] = pd.to_datetime(raw["logged_at"], format="mixed", utc=True, errors="coerce")
raw = raw.drop_duplicates(subset=["logged_at_p", "contract_ticker"], keep="first")
for c in ["resolved_yes", "p_market", "would_pnl"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
t = raw[raw["decision"] == "trade"].dropna(subset=["resolved_yes", "logged_at_p", "p_market"]).copy()
t["side"] = t["side"].str.lower()
t["won"] = np.where(t["side"] == "yes", t["resolved_yes"] == 1, t["resolved_yes"] == 0)
t["be"] = np.where(t["side"] == "yes", t["p_market"], 1 - t["p_market"])
t["tedge"] = t["won"].astype(float) - t["be"]
t = t.sort_values("logged_at_p")
gaps = t["logged_at_p"].diff().dt.total_seconds() / 60
t["episode"] = (gaps > 90).cumsum()
t["week"] = t["logged_at_p"].dt.to_period("W-FRI").astype(str)
print(f"SOL HOURLY combined book: {len(t)} taken  {t['logged_at_p'].min().date()} -> {t['logged_at_p'].max().date()}")


def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()


for statefile, label in [(f"{OUT}/vwap_sol_1h_states.csv", "VWAP-1h HMM"),
                         (f"{OUT}/cg_flow_sol_states.csv", "CG flow HMM")]:
    sv = pd.read_csv(statefile)
    sv["effective"] = pd.to_datetime(sv["effective"], utc=True)
    sv = sv.sort_values("effective")
    tt = pd.merge_asof(t, sv[["effective", "state"]], left_on="logged_at_p",
                       right_on="effective", direction="backward", tolerance=pd.Timedelta("2h"))
    tt = tt.dropna(subset=["state"])
    print(f"\n=== {label}: {len(tt)} hourly taken trades with state ===")
    base = tt["tedge"].mean()
    print(f"book baseline: edge={base:+.4f}  $={tt['would_pnl'].sum():+.2f}")
    for s in sorted(tt["state"].dropna().unique()):
        for side in ["yes", "no"]:
            d = tt[(tt["state"] == s) & (tt["side"] == side)]
            if len(d) < 15:
                continue
            ne, ee, pn = ep_stats(d)
            wk = d.groupby("week")["tedge"].mean()
            print(f"  S{int(s)} {side.upper():3s}: n={len(d)} eps={ne} edge={d['tedge'].mean():+.4f} "
                  f"ep_edge={ee:+.4f} P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} "
                  f"$={d['would_pnl'].sum():+.2f}")
print("DONE_S4")
