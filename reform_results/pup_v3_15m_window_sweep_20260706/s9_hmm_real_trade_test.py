"""
S9 -- Backfill the p_up_v3 regime HMM's states against REAL taken BTC
trades (both hourly and 15m books) and report WR+PnL+breakeven per state,
exactly the same rigor applied to the native-15m signal (which failed
this test despite decent standalone AUC) -- a good abstract correlation
isn't guaranteed to survive contact with the specific, already-gate-
selected population of trades we actually take.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
states = pd.read_parquet(f"{OUT}/pup_v3_hmm_states.parquet")
bar_index = states.index
state_series = states["state"]

STATE_LABEL = {0: "0-neutral", 1: "1-rising(bull)", 2: "2-steady", 3: "3-crashing(bear)"}


def lookup_state(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (logged_at - bar_index[idx]) > pd.Timedelta(hours=2):  # matches the runner's own staleness rule
        return np.nan
    return int(state_series.iloc[idx])


def breakeven(sub):
    be = np.where(sub["side"].str.lower() == "yes", sub["p_market"], 1 - sub["p_market"])
    return np.nanmean(be)


def test_book(path, label, pm_col="p_market"):
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["asset"] == "BTC"].copy() if "asset" in raw.columns else raw
    raw["logged_at_parsed"] = pd.to_datetime(raw["logged_at"], format="mixed", utc=True, errors="coerce")
    taken = raw[raw["decision"] == "trade"].copy()
    taken["state"] = taken["logged_at_parsed"].apply(lookup_state)
    taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
    taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
    taken[pm_col] = pd.to_numeric(taken[pm_col], errors="coerce")
    covered = taken.dropna(subset=["state"]).copy()
    print(f"\n=== {label} === taken={len(taken)}  covered_by_HMM={len(covered)}  "
          f"({covered['logged_at_parsed'].min()} -> {covered['logged_at_parsed'].max()})")
    for st in sorted(covered["state"].unique()):
        sub = covered[covered["state"] == st]
        if len(sub) < 5:
            print(f"  {STATE_LABEL[int(st)]:20s} n={len(sub):4d} (too thin)")
            continue
        wr = sub["would_win"].mean()
        pnl = sub["would_pnl"].sum()
        be = np.nanmean(np.where(sub["side"].str.lower() == "yes", sub[pm_col], 1 - sub[pm_col]))
        print(f"  {STATE_LABEL[int(st)]:20s} n={len(sub):4d}  WR={wr:.3f}  BE={be:.3f}  "
              f"edge={wr-be:+.3f}  PnL=${pnl:8.2f}")
    return covered


hourly = test_book("results/paper_trades.csv", "BTC HOURLY (live/dual)")
m15 = test_book("results/paper_trades_btc15m.csv", "BTC 15m")

print("\n=== Hourly, side-split (state 1 for YES bets, state 3 for NO bets -- the natural gate design) ===")
for side, st in [("yes", 1), ("no", 3)]:
    sub = hourly[(hourly["side"].str.lower() == side) & (hourly["state"] == st)]
    rest = hourly[(hourly["side"].str.lower() == side) & (hourly["state"] != st)]
    if len(sub) >= 5:
        print(f"  {side.upper()} in state {st}: n={len(sub):4d}  WR={sub['would_win'].mean():.3f}  "
              f"PnL=${sub['would_pnl'].sum():8.2f}   |   {side.upper()} elsewhere: n={len(rest):4d}  "
              f"WR={rest['would_win'].mean():.3f}  PnL=${rest['would_pnl'].sum():8.2f}")
