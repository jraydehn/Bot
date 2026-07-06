"""
Backfill p_up_v3 (honest rebuild, 2026-07-04) against actual historical taken
BTC trades in results/paper_trades.csv (hourly) and results/paper_trades_btc15m.csv
(15m), using the already-built extended_dataset.parquet feature matrix + the
trained model — NOT recomputing features from raw candles.

Read-only against the live paper_trades CSVs. Writes results to this directory
only; never touches the original files.

Causal alignment: model card says "row T = last completed 1h bar, decision
time T+1h" i.e. for any decision timestamp within hour H, the last completed
bar is H-1. bar_idx = searchsorted(H, side='left') - 1 where H = logged_at
floored to the hour.
"""
import pickle
import pandas as pd
import numpy as np

REBUILD_DIR = "reform_results/pup_v2_rebuild_20260704"

with open(f"{REBUILD_DIR}/btc_p_up_v3_20260704.pkl", "rb") as f:
    pkg = pickle.load(f)
clf = pkg["clf"]
features = pkg["features"]

ds = pd.read_parquet(f"{REBUILD_DIR}/extended_dataset.parquet")
X = ds[features].astype(float)
p_up_v3_by_bar = pd.Series(clf.predict_proba(X)[:, 1], index=ds.index)
print(f"Scored {len(p_up_v3_by_bar)} historical hourly bars "
      f"({p_up_v3_by_bar.index.min()} -> {p_up_v3_by_bar.index.max()})")

bar_index = p_up_v3_by_bar.index  # sorted, hourly, tz-aware UTC


def lookup_p_up_v3(logged_at: pd.Timestamp):
    if pd.isna(logged_at):
        return np.nan
    h = logged_at.floor("h")
    # out of dataset coverage entirely (before start, or after the last scored
    # bar) -> no prediction available, do NOT clamp to nearest edge row
    if h <= bar_index.min() or h > bar_index.max() + pd.Timedelta(hours=1):
        return np.nan
    idx = bar_index.searchsorted(h, side="left") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    return float(p_up_v3_by_bar.iloc[idx])


def v3_agree(p_v3, side):
    if p_v3 != p_v3 or side not in ("yes", "no"):
        return ""
    return int(p_v3 >= 0.50) if side == "yes" else int(p_v3 < 0.50)


def backfill_file(path, label, decision_col="decision", trade_value="trade"):
    df = pd.read_csv(path, low_memory=False)
    df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
    taken = df[df[decision_col] == trade_value].copy()
    taken["p_up_v3_backfill"] = taken["logged_at_parsed"].apply(lookup_p_up_v3)
    taken["v3_agree_backfill"] = taken.apply(
        lambda r: v3_agree(r["p_up_v3_backfill"], str(r["side"]).lower()), axis=1
    )
    n_covered = taken["p_up_v3_backfill"].notna().sum()
    print(f"\n{label}: {len(taken)} taken trades, {n_covered} covered by backfill "
          f"({taken['logged_at_parsed'].min()} -> {taken['logged_at_parsed'].max()})")
    out_cols = ["logged_at", "logged_at_parsed", "contract_ticker" if "contract_ticker" in taken.columns else "ticker",
                "side", "would_win", "would_pnl", "p_up_v3_backfill", "v3_agree_backfill"]
    out_cols = [c for c in out_cols if c in taken.columns]
    out = taken[out_cols]
    out_path = f"reform_results/pup_v3_backfill_20260706/{label}_backfilled.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path}")
    return out


hourly = backfill_file("results/paper_trades.csv", "btc_hourly")
m15 = backfill_file("results/paper_trades_btc15m.csv", "btc_15m", decision_col="decision", trade_value="trade")
