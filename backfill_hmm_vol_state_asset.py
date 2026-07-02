"""
backfill_hmm_vol_state_asset.py

Backfills HMM vol-regime columns into ETH or SOL scan archives.
Adds three columns using the asset-specific 2-state ergodic HMM:

  hmm_vol_state      : 0=R0 (low-vol) / 1=R1 (high-vol) — hard Viterbi state
  hmm_r1_prob        : P(R1|data) — soft posterior (shadow signal)
  hmm_time_in_state  : bars since last state transition (sojourn depth)

Uses 20-bar 15m log-return window at each scan timestamp (no lookahead).

Usage:
  python3 backfill_hmm_vol_state_asset.py --asset ETH [--dry-run]
  python3 backfill_hmm_vol_state_asset.py --asset SOL [--dry-run]

Output:
  results/eth_scan_archive_hmm.parquet
  results/sol_scan_archive_hmm.parquet
"""
import argparse, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
MODELS  = BASE / "models"
RESULTS = BASE / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--asset",   choices=["ETH", "SOL"], required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

ASSET   = args.asset
TICKER  = {"ETH": "ETHUSDT", "SOL": "SOLUSDT"}[ASSET]
ARCHIVE = {"ETH": "eth_scan_archive.csv", "SOL": "sol_scan_archive.csv"}[ASSET]
PKL     = MODELS / f"hmm_ergodic_2state_{ASSET.lower()}_15m.pkl"
OUT     = RESULTS / f"{ASSET.lower()}_scan_archive_hmm.parquet"

LOOKBACK = 20   # 20 × 15m = 5h context window


# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading HMM model: {PKL.name}")
with open(PKL, "rb") as f:
    pkg = pickle.load(f)

model   = pkg["model"]
order   = sorted(range(model.n_components),
                 key=lambda s: float(np.sqrt(model.covars_[s, 0, 0])))
rank_of = {s: i for i, s in enumerate(order)}
print(f"  States: {model.n_components}  "
      f"R0={order[0]} (low-vol)  R1={order[1]} (high-vol)")


# ── Load scan archive ─────────────────────────────────────────────────────────

print(f"\nLoading archive: {ARCHIVE}")
sa = pd.read_csv(RESULTS / ARCHIVE, low_memory=False)
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
sa = sa.reset_index(drop=True)
print(f"  {len(sa):,} rows  ({sa['logged_at'].min().date()} → {sa['logged_at'].max().date()})")

t_min = sa["logged_at"].min()
t_max = sa["logged_at"].max()


# ── Load 1m → resample to 15m ─────────────────────────────────────────────────

print(f"\nLoading 1m parquet for {ASSET} …", end=" ", flush=True)
import os
pq_1m = max(DATA.glob(f"binanceus_{TICKER}_1m_2024-01-01_*.parquet"),
            key=os.path.getmtime)
df_1m = pd.read_parquet(pq_1m, columns=["close"])
df_1m.index = pd.to_datetime(df_1m.index, utc=True)

warmup = pd.Timedelta(days=10)
df_1m  = df_1m[(df_1m.index >= t_min - warmup) & (df_1m.index <= t_max)]
c15    = df_1m["close"].resample("15min").last().dropna()
lr     = np.log(c15 / c15.shift(1)).dropna()
lr_idx = lr.index
lr_val = lr.values
print(f"{len(lr):,} 15m bars  ({lr_idx[0].date()} → {lr_idx[-1].date()})")


# ── Decode full 15m series once, track sojourn in bar-space ──────────────────
# Sojourn depth must be computed in 15m bar space (not archive row space),
# otherwise multiple archive rows at the same timestamp inflate the count.

print(f"\nDecoding full 15m series ({len(lr):,} bars) …", end=" ", flush=True)

bar_states   = []
bar_r1_probs = []
bar_sojourn  = []

prev_rank     = None
sojourn_depth = 0

for i in range(len(lr)):
    if i < LOOKBACK:
        bar_states.append(np.nan)
        bar_r1_probs.append(np.nan)
        bar_sojourn.append(np.nan)
        continue

    window    = lr_val[i - LOOKBACK + 1: i + 1].reshape(-1, 1)
    vit       = model.predict(window)
    raw_state = int(vit[-1])
    state_rank = rank_of[raw_state]

    # Soft posterior for R1
    log_post  = model.score_samples(window)[1]
    post_last = np.exp(log_post[-1])
    post_last = post_last / post_last.sum()
    r1_prob   = float(post_last[order[1]])

    # Sojourn depth in 15m bars
    if state_rank != prev_rank:
        prev_rank     = state_rank
        sojourn_depth = 1
    else:
        sojourn_depth += 1

    bar_states.append(float(state_rank))
    bar_r1_probs.append(r1_prob)
    bar_sojourn.append(float(sojourn_depth))

print("done.")

# Build a 15m-indexed DataFrame for merge_asof
bar_df = pd.DataFrame({
    "bar_ts":           lr_idx,
    "hmm_vol_state":    bar_states,
    "hmm_r1_prob":      bar_r1_probs,
    "hmm_time_in_state": bar_sojourn,
}).dropna(subset=["hmm_vol_state"]).sort_values("bar_ts")

# merge_asof: each archive row gets the most recent 15m bar's state
print(f"Merging into {len(sa):,} archive rows …", end=" ", flush=True)
sa_sorted = sa.sort_values("logged_at").copy()
sa_sorted["_orig_idx"] = sa_sorted.index

merged = pd.merge_asof(
    sa_sorted[["logged_at", "_orig_idx"]].sort_values("logged_at"),
    bar_df.sort_values("bar_ts"),
    left_on="logged_at",
    right_on="bar_ts",
    direction="backward",
).set_index("_orig_idx").sort_index()

sa["hmm_vol_state"]     = merged["hmm_vol_state"]
sa["hmm_r1_prob"]       = merged["hmm_r1_prob"]
sa["hmm_time_in_state"] = merged["hmm_time_in_state"]

filled = sa["hmm_vol_state"].notna().sum()
print(f"done.  {filled:,}/{len(sa):,} rows filled "
      f"({filled/len(sa):.1%})")

r0_rows = (sa["hmm_vol_state"] == 0).sum()
r1_rows = (sa["hmm_vol_state"] == 1).sum()
print(f"\n  R0 rows: {r0_rows:,} ({r0_rows/filled:.1%})  "
      f"R1 rows: {r1_rows:,} ({r1_rows/filled:.1%})")
print(f"  hmm_r1_prob  mean={sa['hmm_r1_prob'].mean():.3f}  "
      f"std={sa['hmm_r1_prob'].std():.3f}")
print(f"  hmm_time_in_state  R0 mean={sa.loc[sa['hmm_vol_state']==0,'hmm_time_in_state'].mean():.1f}  "
      f"R1 mean={sa.loc[sa['hmm_vol_state']==1,'hmm_time_in_state'].mean():.1f} bars")


# ── Save ──────────────────────────────────────────────────────────────────────

if args.dry_run:
    print("\n[dry-run] Not saving.")
else:
    sa.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT.name}  ({len(sa):,} rows, {len(sa.columns)} cols)")

print("Done.")
