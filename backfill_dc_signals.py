"""
backfill_dc_signals.py

Backfills ATR Directional Change signals into the BTC 1h scan archive.

For each archive row (logged_at timestamp) we compute — for each ATR lookback:
  dc_direction_{lb}     : +1 (tracking high, last swing was a low)
                          -1 (tracking low, last swing was a high)
  dist_high_{lb}        : (spot - last_swing_high) / spot * 100  (negative = below swing high)
  dist_low_{lb}         : (spot - last_swing_low)  / spot * 100  (positive = above swing low)
  bars_since_swing_{lb} : 1m bars since the last confirmed swing (any direction)

Methodology:
  - Run ATRDirectionalChange once on the full 1m series per lookback.
  - Use conf_timestamp (when the swing was confirmed, not when it peaked) for merge-asof.
    A swing at bar T is only visible after it's confirmed at bar T+k, so this prevents lookahead.
  - pd.merge_asof matches each archive row to the last confirmed swing visible at logged_at.

Usage:
  python3 backfill_dc_signals.py [--lookbacks 60 240 1440] [--dry-run]

Output: results/btc_scan_archive_dc.parquet
"""
import argparse
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--lookbacks", type=int, nargs="+", default=[60, 240, 1440])
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()


# ── LocalExtreme dataclass (no external dependency) ──────────────────────────

@dataclass
class LocalExtreme:
    ext_type: int          # +1 = swing high, -1 = swing low
    index: int             # bar index of the extreme
    price: float           # price at the extreme (high or low)
    timestamp: pd.Timestamp
    conf_index: int        # bar index where the swing was confirmed
    conf_price: float      # close at confirmation bar
    conf_timestamp: pd.Timestamp


# ── ATR Directional Change (from user code, minor fixes) ─────────────────────

class ATRDirectionalChange:
    def __init__(self, atr_lookback: int):
        self._up_move     = True
        self._pend_max    = np.nan
        self._pend_min    = np.nan
        self._pend_max_i  = 0
        self._pend_min_i  = 0
        self._atr_lb      = atr_lookback
        self._atr_sum     = np.nan
        self._initialized = False
        self.extremes: List[LocalExtreme] = []

    def _create_ext(self, ext_type, ext_i, conf_i, time_index, high, low, close):
        arr = high if ext_type == "high" else low
        self.extremes.append(LocalExtreme(
            ext_type   = 1 if ext_type == "high" else -1,
            index      = ext_i,
            price      = arr[ext_i],
            timestamp  = time_index[ext_i],
            conf_index = conf_i,
            conf_price = close[conf_i],
            conf_timestamp = time_index[conf_i],
        ))

    def update(self, i, time_index, high, low, close):
        # ATR computation
        if i < self._atr_lb:
            return
        elif i == self._atr_lb:
            h_win = high[i - self._atr_lb + 1: i + 1]
            l_win = low [i - self._atr_lb + 1: i + 1]
            c_win = close[i - self._atr_lb: i]
            tr1 = h_win - l_win
            tr2 = np.abs(h_win - c_win)
            tr3 = np.abs(l_win - c_win)
            self._atr_sum = np.sum(np.max(np.stack([tr1, tr2, tr3]), axis=0))
        else:
            tr_curr = max(high[i] - low[i],
                          abs(high[i] - close[i-1]),
                          abs(low[i]  - close[i-1]))
            rm = i - self._atr_lb
            tr_rm   = max(high[rm] - low[rm],
                          abs(high[rm] - close[rm-1]),
                          abs(low[rm]  - close[rm-1]))
            self._atr_sum += tr_curr - tr_rm

        atr = self._atr_sum / self._atr_lb

        # First bar after ATR is ready
        if not self._initialized:
            self._pend_max   = high[i]
            self._pend_min   = low[i]
            self._pend_max_i = self._pend_min_i = i
            self._initialized = True
            return

        if self._up_move:
            if high[i] > self._pend_max:
                self._pend_max   = high[i]
                self._pend_max_i = i
            elif low[i] < self._pend_max - atr:
                self._create_ext("high", self._pend_max_i, i, time_index, high, low, close)
                self._up_move    = False
                self._pend_min   = low[i]
                self._pend_min_i = i
        else:
            if low[i] < self._pend_min:
                self._pend_min   = low[i]
                self._pend_min_i = i
            elif high[i] > self._pend_min + atr:
                self._create_ext("low", self._pend_min_i, i, time_index, high, low, close)
                self._up_move    = True
                self._pend_max   = high[i]
                self._pend_max_i = i


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading 1m parquet …", end=" ", flush=True)
pq_1m = max(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"), key=os.path.getmtime)
df1m  = pd.read_parquet(pq_1m, columns=["open", "high", "low", "close"])
if df1m.index.tz is None:
    df1m.index = df1m.index.tz_localize("UTC")
df1m  = df1m.sort_index()
print(f"{len(df1m):,} bars  ({df1m.index[0].date()} → {df1m.index[-1].date()})")

h = df1m["high"].to_numpy()
l = df1m["low"].to_numpy()
c = df1m["close"].to_numpy()
t = df1m.index

print("Loading scan archive …", end=" ", flush=True)
sa = pd.read_csv(RESULTS / "btc_scan_archive.csv", low_memory=False)
sa = sa[sa["resolved_yes"].notna()].copy()
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
sa["p_market"]  = pd.to_numeric(sa["p_market"],  errors="coerce")
sa["spot"]      = pd.to_numeric(sa["spot"],       errors="coerce")
sa["strike"]    = pd.to_numeric(sa["strike"],     errors="coerce")
print(f"{len(sa):,} resolved rows  ({sa['logged_at'].min().date()} → {sa['logged_at'].max().date()})")


# ── Run DC for each lookback, build feature columns ───────────────────────────

for lb in args.lookbacks:
    print(f"\n── ATR lookback = {lb} bars ({lb//60}h) ──────────────────────────────")

    dc = ATRDirectionalChange(lb)
    print(f"  Running DC on {len(h):,} bars …", end=" ", flush=True)
    for i in range(len(h)):
        dc.update(i, t, h, l, c)
    print(f"done.  {len(dc.extremes)} swings confirmed.")

    if len(dc.extremes) == 0:
        print("  No swings — skipping lookback.")
        continue

    ext_df = pd.DataFrame([{
        "ext_type":      e.ext_type,
        "price":         e.price,
        "timestamp":     e.timestamp,
        "conf_timestamp": e.conf_timestamp,
    } for e in dc.extremes])

    highs = ext_df[ext_df["ext_type"] ==  1].copy().sort_values("conf_timestamp")
    lows  = ext_df[ext_df["ext_type"] == -1].copy().sort_values("conf_timestamp")

    print(f"  Highs: {len(highs)}  Lows: {len(lows)}")
    print(f"  First swing confirmed: {ext_df['conf_timestamp'].min()}")

    # merge_asof: for each archive row find the last confirmed high/low visible at logged_at
    # Use a clean sorted frame with original index preserved for alignment
    sa = sa.sort_values("logged_at").reset_index(drop=True)
    sa_clean = sa.dropna(subset=["logged_at"]).copy()
    sa_clean["_orig_idx"] = sa_clean.index

    def _asof(left, right_ts_col, right_val_col, col_name):
        """merge_asof wrapper that returns aligned series indexed to sa_clean._orig_idx."""
        right = right_ts_col.rename("logged_at").to_frame()
        right[col_name] = right_val_col if isinstance(right_val_col, np.ndarray) else right_val_col.values
        merged = pd.merge_asof(
            left[["logged_at", "_orig_idx"]].sort_values("logged_at"),
            right.sort_values("logged_at"),
            on="logged_at", direction="backward",
        )
        return merged.set_index("_orig_idx")[col_name]

    col_h = _asof(sa_clean, highs["conf_timestamp"], highs["price"],              f"swing_high_{lb}")
    col_l = _asof(sa_clean, lows["conf_timestamp"],  lows["price"],               f"swing_low_{lb}")
    all_ext = ext_df[["conf_timestamp","ext_type"]].sort_values("conf_timestamp")
    col_d = _asof(sa_clean, all_ext["conf_timestamp"], all_ext["ext_type"],      f"dc_direction_{lb}")

    sa[f"swing_high_{lb}"]  = col_h
    sa[f"swing_low_{lb}"]   = col_l
    sa[f"dc_direction_{lb}"] = col_d

    # Distance features (pct of spot)
    sa[f"dist_high_{lb}"] = (sa["spot"] - sa[f"swing_high_{lb}"]) / sa["spot"] * 100
    sa[f"dist_low_{lb}"]  = (sa["spot"] - sa[f"swing_low_{lb}"])  / sa["spot"] * 100

    # Strike position relative to swings
    sa[f"strike_above_high_{lb}"] = sa["strike"] > sa[f"swing_high_{lb}"]
    sa[f"strike_below_low_{lb}"]  = sa["strike"] < sa[f"swing_low_{lb}"]

    cov_h = sa[f"swing_high_{lb}"].notna().mean()
    cov_l = sa[f"swing_low_{lb}"].notna().mean()
    print(f"  Coverage: highs={cov_h:.1%}  lows={cov_l:.1%}")
    print(f"  dist_high mean={sa[f'dist_high_{lb}'].mean():+.2f}%  "
          f"dist_low mean={sa[f'dist_low_{lb}'].mean():+.2f}%")


# ── Save ──────────────────────────────────────────────────────────────────────

out = RESULTS / "btc_scan_archive_dc.parquet"
if args.dry_run:
    print("\n[dry-run] Not saving.")
else:
    sa.to_parquet(out, index=False)
    print(f"\nSaved → {out.name}  ({len(sa):,} rows, {len(sa.columns)} columns)")

print("Done.")
