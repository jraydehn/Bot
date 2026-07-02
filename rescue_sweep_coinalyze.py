"""
rescue_sweep_coinalyze.py

For every implemented gate in blocked_trades.csv:
  - Break blocked YES trades by: funding_bias, vpin_score, liq_score, liq_bias,
    ls_long_pct bucket, oi_chg_pct bucket, squeeze_1h (backfilled from paper_trades)
  - Report n, would_win%, total would_pnl per (gate, signal_bucket)
  - Flag rescue candidates: n>=15, would_win%>=60%, total would_pnl>0
  - Flag gate confirmations: n>=15, would_win%<=40%, total would_pnl<0

Usage: python3 rescue_sweep_coinalyze.py [--min-n 15] [--asset BTC]
"""

import pandas as pd
import numpy as np
import warnings
import argparse
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--min-n", type=int, default=15)
parser.add_argument("--asset", type=str, default=None, help="Filter to asset (BTC/ETH/SOL), or all")
args = parser.parse_args()

MIN_N   = args.min_n
ASSET   = args.asset

print("Loading data...")
bt = pd.read_csv("results/blocked_trades.csv", low_memory=False)
pt = pd.read_csv("results/paper_trades.csv",   low_memory=False)

# ── Backfill squeeze_1h from paper_trades via contract_ticker ──────────────────
print("Backfilling squeeze_1h from paper_trades...")
pt_squeeze = pt[["contract_ticker", "squeeze_1h"]].dropna(subset=["squeeze_1h"]).copy()
# Normalise to numeric
def _norm_squeeze(s):
    s = s.astype(str).str.lower().str.strip()
    out = pd.to_numeric(s, errors="coerce")
    out[s == "true"]  = 1.0
    out[s == "false"] = 0.0
    # extract first float from strings like '1.0'
    return out

pt_squeeze["squeeze_1h"] = _norm_squeeze(pt_squeeze["squeeze_1h"])
pt_squeeze = (pt_squeeze.dropna(subset=["squeeze_1h"])
                         .drop_duplicates("contract_ticker")
                         .rename(columns={"squeeze_1h": "squeeze_1h_fill"}))

bt = bt.merge(pt_squeeze, on="contract_ticker", how="left")
bt["squeeze_1h"] = bt["squeeze_1h_fill"]

# Also backfill ls_long_pct and oi_chg_pct and liq_score from scan archive if missing
print("Backfilling sparse signals from scan archive...")
arc_cols = ["contract_ticker", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct"]
arc = pd.read_csv("results/btc_scan_archive.csv", low_memory=False, usecols=arc_cols)
arc = arc.drop_duplicates("contract_ticker")

for col in ["liq_score","liq_bias","ls_long_pct","oi_chg_pct"]:
    mask = bt[col].isna()
    if mask.sum() > 0:
        filled = bt[mask][["contract_ticker"]].merge(arc[["contract_ticker",col]], on="contract_ticker", how="left")
        bt.loc[mask, col] = filled[col].values

# ── Filters ───────────────────────────────────────────────────────────────────
if ASSET:
    bt = bt[bt["asset"] == ASSET]

# Only YES bets with known outcome
bt_yes = bt[(bt["side"] == "yes") & bt["would_pnl"].notna()].copy()
bt_no  = bt[(bt["side"] == "no")  & bt["would_pnl"].notna()].copy()

print(f"\nBlocked YES trades: {len(bt_yes):,}   Blocked NO trades: {len(bt_no):,}")

# ── Coverage after backfill ───────────────────────────────────────────────────
signals = ["funding_bias","vpin_score","liq_score","liq_bias","ls_long_pct","oi_chg_pct","squeeze_1h"]
print("\nSignal coverage in blocked YES trades after backfill:")
for sig in signals:
    n = bt_yes[sig].notna().sum()
    print(f"  {sig:<18} {n:>6}/{len(bt_yes):>6} ({100*n/len(bt_yes):.1f}%)")

# ── Helper: bucket continuous signals ─────────────────────────────────────────
def bucket_continuous(s, col, q=5):
    """Bucket a continuous column into quintiles."""
    try:
        return pd.qcut(s[col], q=q, duplicates="drop", labels=False)
    except Exception:
        return None

# ── Core analysis function ────────────────────────────────────────────────────
def analyze_gate_signal(df_gate, gate_name, signal_col, signal_label,
                         bucket=False, q=5, min_n=MIN_N):
    """
    For a single (gate, signal) pair:
    Returns a DataFrame of rows: gate, signal, bucket_label, n, win_pct, total_pnl, flag
    """
    sub = df_gate[df_gate[signal_col].notna()].copy()
    if len(sub) < min_n:
        return None

    if bucket:
        sub["_bucket"] = bucket_continuous(sub, signal_col, q)
        if sub["_bucket"].isna().all():
            return None
        grp_col = "_bucket"
    else:
        sub["_bucket"] = sub[signal_col]
        grp_col = "_bucket"

    rows = []
    # Overall for gate
    total_n   = len(sub)
    total_win = (sub["would_pnl"] > 0).sum()
    total_pnl = sub["would_pnl"].sum()

    for bkt, grp in sub.groupby(grp_col):
        n   = len(grp)
        if n < min_n:
            continue
        win = (grp["would_pnl"] > 0).sum()
        pnl = grp["would_pnl"].sum()
        win_pct = 100 * win / n if n > 0 else 0
        # Flag
        flag = ""
        if win_pct >= 60 and pnl > 0:
            flag = "RESCUE_CANDIDATE"
        elif win_pct <= 40 and pnl < 0:
            flag = "GATE_CONFIRMED"
        elif win_pct >= 55 and pnl > 0:
            flag = "rescue_watch"
        rows.append({
            "gate":        gate_name,
            "signal":      signal_label,
            "bucket":      str(bkt),
            "n":           n,
            "win_pct":     round(win_pct, 1),
            "total_pnl":   round(pnl, 2),
            "flag":        flag,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)

# ── Gates to skip (mechanical floors, not real logic gates) ───────────────────
SKIP_GATES = {"no_pm_floor"}

# ── Run sweep ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("RESCUE SWEEP — BLOCKED YES TRADES")
print("="*80)

# Discrete signals (take values as-is)
discrete_sigs = {
    "funding_bias": "funding_bias",
    "vpin_score":   "vpin_score",
    "liq_score":    "liq_score",
    "liq_bias":     "liq_bias",
    "squeeze_1h":   "squeeze_1h",
}
# Continuous (bucket into quintiles)
continuous_sigs = {
    "ls_long_pct": "ls_long_pct",
    "oi_chg_pct":  "oi_chg_pct",
}

all_results = []

gates = [g for g in bt_yes["gate_name"].unique() if g not in SKIP_GATES]

for gate in sorted(gates):
    df_gate = bt_yes[bt_yes["gate_name"] == gate]
    if len(df_gate) < MIN_N:
        continue

    gate_pnl = df_gate["would_pnl"].sum()
    gate_win  = 100*(df_gate["would_pnl"]>0).sum()/len(df_gate)

    print(f"\n{'─'*70}")
    print(f"GATE: {gate}  |  n={len(df_gate):,}  would_win={gate_win:.1f}%  total_would_pnl=${gate_pnl:+.2f}")
    print(f"{'─'*70}")

    gate_had_rescue = False

    for col, label in {**discrete_sigs, **continuous_sigs}.items():
        is_cont = col in continuous_sigs
        res = analyze_gate_signal(df_gate, gate, col, label,
                                   bucket=is_cont, q=5, min_n=MIN_N)
        if res is None:
            continue

        flagged = res[res["flag"].str.contains("RESCUE|rescue", na=False)]
        confirmed = res[res["flag"].str.contains("GATE_CONFIRMED", na=False)]

        # Print all rows for this signal
        print(f"\n  Signal: {label}")
        print(f"  {'bucket':>12}  {'n':>6}  {'win%':>6}  {'total_pnl':>10}  flag")
        for _, row in res.sort_values("bucket").iterrows():
            flag_str = f"  *** {row['flag']} ***" if row['flag'] else ""
            print(f"  {row['bucket']:>12}  {row['n']:>6}  {row['win_pct']:>5.1f}%  ${row['total_pnl']:>9.2f}{flag_str}")

        all_results.append(res)
        if not flagged.empty:
            gate_had_rescue = True

# ── Summary table of all rescue candidates ───────────────────────────────────
print("\n\n" + "="*80)
print("RESCUE CANDIDATES SUMMARY (would_win>=60%, pnl>0, n>=" + str(MIN_N) + ")")
print("="*80)

if all_results:
    full = pd.concat(all_results, ignore_index=True)
    rescues = full[full["flag"].str.contains("RESCUE_CANDIDATE", na=False)].sort_values("total_pnl", ascending=False)
    if not rescues.empty:
        print(rescues[["gate","signal","bucket","n","win_pct","total_pnl"]].to_string(index=False))
    else:
        print("No strong rescue candidates found (win%>=60% AND pnl>0).")

    print("\n" + "="*80)
    print("GATE CONFIRMATIONS SUMMARY (would_win<=40%, pnl<0)")
    print("="*80)
    confirms = full[full["flag"] == "GATE_CONFIRMED"].sort_values("total_pnl")
    if not confirms.empty:
        print(confirms[["gate","signal","bucket","n","win_pct","total_pnl"]].to_string(index=False))
    else:
        print("No strong gate confirmations found.")

    # Save full results
    full.to_csv("results/rescue_sweep_coinalyze.csv", index=False)
    print(f"\nFull results saved to results/rescue_sweep_coinalyze.csv")
else:
    print("No results generated.")

print("\nDone.")
