"""
S11 -- Full PnL breakdown (WR + PnL + breakeven, the standing reporting
standard for this project) using the clean 3-state merged labeling
(rising / neutral / crashing), for both books, split by side. Then an
applicability table: for each state x side cell, is there enough data
and enough week-to-week robustness to act on it, or not.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
states = pd.read_parquet(f"{OUT}/pup_v3_hmm_states.parquet")
bar_index = states.index
state3_series = states["state"].map({0: "neutral", 1: "rising", 2: "neutral", 3: "crashing"})


def lookup_state(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (logged_at - bar_index[idx]) > pd.Timedelta(hours=2):
        return np.nan
    return state3_series.iloc[idx]


def build(path, label, asset_filter=True):
    raw = pd.read_csv(path, low_memory=False)
    if asset_filter and "asset" in raw.columns:
        raw = raw[raw["asset"] == "BTC"].copy()
    taken = raw[raw["decision"] == "trade"].copy()
    taken["logged_at_parsed"] = pd.to_datetime(taken["logged_at"], format="mixed", utc=True, errors="coerce")
    taken["state3"] = taken["logged_at_parsed"].apply(lookup_state)
    taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
    taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
    pm_col = "p_market"
    taken[pm_col] = pd.to_numeric(taken[pm_col], errors="coerce")
    taken["side"] = taken["side"].str.lower()
    taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
    covered = taken.dropna(subset=["state3"]).copy()
    covered["be"] = np.where(covered["side"] == "yes", covered[pm_col], 1 - covered[pm_col])
    return covered


def report_table(covered, label):
    print(f"\n{'='*90}\n{label}  (n_covered={len(covered)})\n{'='*90}")
    print(f"{'state':10s} {'side':5s} {'n':>4s} {'WR':>6s} {'BE':>6s} {'edge':>7s} {'PnL':>9s}  {'weeks_seen':>10s} {'worst_wk_share':>14s}")
    rows = []
    for st in ["rising", "neutral", "crashing"]:
        for side in ["yes", "no"]:
            sub = covered[(covered["state3"] == st) & (covered["side"] == side)]
            n = len(sub)
            if n == 0:
                continue
            wr = sub["would_win"].mean()
            be = sub["be"].mean()
            pnl = sub["would_pnl"].sum()
            wk_pnl = sub.groupby("week")["would_pnl"].sum()
            n_weeks = len(wk_pnl)
            # worst-case: what % of total (abs) PnL comes from the single most extreme week
            worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
            print(f"{st:10s} {side:5s} {n:4d} {wr:6.3f} {be:6.3f} {wr-be:+7.3f} {pnl:9.2f}  {n_weeks:10d} {worst_share:13.1f}%")
            rows.append({"state": st, "side": side, "n": n, "wr": wr, "be": be, "pnl": pnl,
                        "n_weeks": n_weeks, "worst_wk_share": worst_share})
    return pd.DataFrame(rows)


hourly = build("results/paper_trades.csv", "hourly")
m15 = build("results/paper_trades_btc15m.csv", "15m")

r_hourly = report_table(hourly, "BTC HOURLY -- 3-state merged, by side")
r_15m = report_table(m15, "BTC 15m -- 3-state merged, by side")

print(f"\n{'='*90}\nAPPLICABILITY VERDICT (n>=30 AND n_weeks>=4 AND worst_wk_share<60% required to call it actionable)\n{'='*90}")
for label, r in [("HOURLY", r_hourly), ("15m", r_15m)]:
    for _, row in r.iterrows():
        ok_n = row["n"] >= 30
        ok_weeks = row["n_weeks"] >= 4
        ok_conc = row["worst_wk_share"] < 60
        verdict = "ACTIONABLE" if (ok_n and ok_weeks and ok_conc) else \
                 ("TOO THIN (n<30)" if not ok_n else
                  "TOO FEW WEEKS" if not ok_weeks else
                  "SINGLE-WEEK DRIVEN")
        print(f"{label:7s} {row['state']:10s} {row['side']:5s} n={row['n']:4d}  wks={row['n_weeks']:2d}  "
              f"worst_wk_share={row['worst_wk_share']:5.1f}%  ->  {verdict}")
