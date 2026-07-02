"""
backfill_flag_signals.py

Backfills bull/bear flag and pennant signal columns into BTC scan archives
using the trendline-confirmed flag/pennant detector from flag_pennant.py.

Detection runs on 1h close prices (log-transformed).

Output columns (added to each archive):
  flag_bull_bars_ago  : bars since last confirmed bull flag/pennant (-1 = none in 48h)
  flag_bear_bars_ago  : bars since last confirmed bear flag/pennant (-1 = none in 48h)
  flag_bull_tip_y     : price at top of bull pole (actual price, not log)
  flag_bear_tip_y     : price at bottom of bear pole
  flag_bull_pole_pct  : pole height as % of base price
  flag_bear_pole_pct  : pole depth as % of base price
  flag_signal         : +1 bull active, -1 bear active, 0 none

Usage:
  python3 backfill_flag_signals.py [--order 10] [--lookback 48] [--dry-run]

Output: results/btc_scan_archive_flags.parquet
  (based on btc_scan_archive_dc.parquet with flag columns added)
"""
import argparse, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from flag_pennant import build_signal_series

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--order",    type=int, default=10,
                    help="Rolling-window order for top/bottom detection (default 10)")
parser.add_argument("--lookback", type=int, default=48,
                    help="Max bars ago a confirmed flag counts as active (default 48h)")
parser.add_argument("--dry-run",  action="store_true")
args = parser.parse_args()


# ── Load 1h OHLCV ─────────────────────────────────────────────────────────────

print("Loading 1h parquet …", end=" ", flush=True)
pq_1h = max(DATA.glob("binanceus_BTCUSDT_1h_*.parquet"), key=os.path.getmtime)
df1h  = pd.read_parquet(pq_1h, columns=["close"])
df1h.index = pd.to_datetime(df1h.index, utc=True)
df1h  = df1h.sort_index()
# Drop obviously stale bars before exchange launched
df1h  = df1h[df1h.index >= "2020-01-01"]
print(f"{len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")


# ── Run signal builder ────────────────────────────────────────────────────────

print(f"Building flag signal series (order={args.order}, lookback={args.lookback}) …",
      end=" ", flush=True)
sig = build_signal_series(df1h["close"], order=args.order, lookback_bars=args.lookback)
print("done.")

bull_active = (sig["flag_bull_bars_ago"] >= 0).sum()
bear_active = (sig["flag_bear_bars_ago"] >= 0).sum()
total_bars  = len(sig)
print(f"  Bull active: {bull_active:,} bars ({bull_active/total_bars:.1%})")
print(f"  Bear active: {bear_active:,} bars ({bear_active/total_bars:.1%})")
print(f"  signal=+1:   {(sig['flag_signal']== 1).sum():,}  "
      f"signal=-1: {(sig['flag_signal']==-1).sum():,}  "
      f"signal=0:  {(sig['flag_signal']== 0).sum():,}")


# ── Load scan archive ─────────────────────────────────────────────────────────

print("\nLoading scan archive …", end=" ", flush=True)
sa = pd.read_parquet(RESULTS / "btc_scan_archive_dc.parquet")
sa = sa[sa["resolved_yes"].notna()].copy()
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
sa["spot"]      = pd.to_numeric(sa["spot"], errors="coerce")
print(f"{len(sa):,} rows  ({sa['logged_at'].min().date()} → {sa['logged_at'].max().date()})")


# ── Merge signal → archive using merge_asof ───────────────────────────────────

FLAG_COLS = [
    "flag_bull_bars_ago", "flag_bear_bars_ago",
    "flag_bull_tip_y",    "flag_bear_tip_y",
    "flag_bull_pole_pct", "flag_bear_pole_pct",
    "flag_signal",
]

sig_reset = sig[FLAG_COLS].copy()
sig_reset.index.name = "bar_ts"
sig_reset = sig_reset.reset_index()
sig_reset["bar_ts"] = pd.to_datetime(sig_reset["bar_ts"], utc=True)

sa = sa.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
sa["_i"] = sa.index

merged = pd.merge_asof(
    sa[["logged_at", "_i"]].sort_values("logged_at"),
    sig_reset.sort_values("bar_ts"),
    left_on="logged_at",
    right_on="bar_ts",
    direction="backward",
).set_index("_i").sort_index()

for col in FLAG_COLS:
    sa[col] = merged[col].values

sa = sa.drop(columns=["_i"])

# Coverage report
print("\nCoverage after merge:")
for col in FLAG_COLS:
    cov = sa[col].notna().mean()
    if col == "flag_signal":
        b = (sa[col]== 1).sum(); be = (sa[col]==-1).sum(); z = (sa[col]==0).sum()
        print(f"  {col:<25}: {cov:.1%}  (bull={b:,} bear={be:,} none={z:,})")
    elif "bars_ago" in col:
        active = (sa[col] >= 0).sum()
        print(f"  {col:<25}: {cov:.1%}  active={active:,} ({active/len(sa):.1%})")
    else:
        print(f"  {col:<25}: {cov:.1%}")


# ── Save ──────────────────────────────────────────────────────────────────────

out = RESULTS / "btc_scan_archive_flags.parquet"
if args.dry_run:
    print("\n[dry-run] Not saving.")
else:
    sa.to_parquet(out, index=False)
    print(f"\nSaved → {out.name}  ({len(sa):,} rows, {len(sa.columns)} cols)")

print("Done.")
