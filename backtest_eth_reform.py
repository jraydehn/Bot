"""
backtest_eth_reform.py

Gate-aware comparison: ETH direct_p_model (current) vs score_to_p_model YES/NO
branches (proposed reform) on 761 resolved paper trades (Apr 2 – May 8 2026).

For each trade the backtest:
  1. Uses actual Kalshi p_market prices (logged in paper_trades CSVs)
  2. Applies the existing gate stack using logged signal columns
  3. Reconstructs 30m composite scores from Binance data to compute blended p_up
  4. Computes new p_yes / p_no from score_to_p_model (k=0.80) and
     score_to_p_no_model (k=0.30) with tau-blended p_up
  5. Reports: which trades would be taken, edge, WR, P&L vs current model

Reform parameters:
  k_drift_yes = 0.80  (confirmed by calibration sweep)
  k_drift_no  = 0.30  (conservative regime-robust start)
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from composite_scorer import (
    score_to_p_model, score_to_p_no_model,
    lookup_p_up_blended, lookup_p_up,
)
from composite_scorer import compute_scores_30m

BASE  = Path(__file__).parent
DATA  = BASE / "data"
SYM   = "ETHUSDT"
ASSET = "ETH"

K_DRIFT_YES = 0.80
K_DRIFT_NO  = 0.30
MIN_EDGE    = 0.02

ARCHIVE_FILES = [
    BASE / "results" / "paper_trades_eth_archive_20260407_122844.csv",
    BASE / "results" / "paper_trades_eth_archive_20260407_152310.csv",
    BASE / "results" / "paper_trades_eth_archive_20260415_1342_precal.csv",
    BASE / "results" / "paper_trades_eth_archive_20260420_0431_pre_counter_tape.csv",
    BASE / "results" / "paper_trades_eth_archive_20260420_2328_pre_direct_model.csv",
    BASE / "results" / "paper_trades_eth.csv",
]

# ── load + consolidate ────────────────────────────────────────────────────────
print("Loading paper trade archives …")
dfs = []
for f in ARCHIVE_FILES:
    df = pd.read_csv(f, low_memory=False)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df["decision_time"] = pd.to_datetime(all_df["decision_time"], utc=True, errors="coerce")

resolved = all_df[
    all_df["resolved_yes"].notna() &
    (all_df["decision"] == "trade")
].copy()

resolved = resolved.drop_duplicates(subset=["contract_ticker", "decision_time", "side"])
resolved = resolved.sort_values("decision_time").reset_index(drop=True)
print(f"  {len(resolved)} unique resolved paper trades  "
      f"({resolved['decision_time'].min().date()} → {resolved['decision_time'].max().date()})")

# ── load Binance data for 30m score reconstruction ───────────────────────────
print("Loading ETH Binance data for 30m score reconstruction …")

def latest(pat):
    files = sorted(DATA.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    return files[-1]

df_1h = pd.read_parquet(latest(f"binanceus_{SYM}_1h_2024-01-01_*.parquet"))
df_1m = pd.read_parquet(latest(f"binanceus_{SYM}_1m_2024-01-01_*.parquet"))
for df in (df_1h, df_1m):
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

# Build 15m from 1m for 30m score computation
df_15m_r = df_1m.resample("15min").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()
df_15m_r.index = df_15m_r.index.tz_localize("UTC") \
    if df_15m_r.index.tz is None else df_15m_r.index

# Filter to relevant period (saves memory during per-trade slicing)
cutoff_1m  = pd.Timestamp("2026-03-01", tz="UTC")
cutoff_15m = pd.Timestamp("2026-03-01", tz="UTC")
df_1m_cut  = df_1m[df_1m.index >= cutoff_1m]
df_15m_cut = df_15m_r[df_15m_r.index >= cutoff_15m]

# ── reconstruct 30m scores per trade ─────────────────────────────────────────
print(f"Reconstructing 30m composite scores for {len(resolved)} trades …")

trend_30m_list = []
rev_30m_list   = []
failed = 0

for i, row in resolved.iterrows():
    ts = row["decision_time"]

    h1_slice  = df_1h[df_1h.index < ts.floor("1h")].tail(120)
    m15_slice = df_15m_cut[df_15m_cut.index < ts].tail(300)
    m1_slice  = df_1m_cut[df_1m_cut.index < ts].tail(500)

    try:
        if len(h1_slice) < 60 or len(m15_slice) < 20 or len(m1_slice) < 62:
            raise ValueError("insufficient bars")

        ts_30m = pd.DatetimeIndex([ts.floor("30min")])
        tr30, rv30 = compute_scores_30m(
            h1_slice["close"], h1_slice["high"], h1_slice["low"], h1_slice["volume"],
            m15_slice["close"], m15_slice["high"], m15_slice["low"],
            m1_slice["close"], m1_slice["volume"],
            ts_30m=ts_30m,
        )
        trend_30m_list.append(int(tr30.iloc[-1]))
        rev_30m_list.append(int(rv30.iloc[-1]))
    except Exception:
        trend_30m_list.append(0)
        rev_30m_list.append(0)
        failed += 1

resolved["trend_30m"] = trend_30m_list
resolved["rev_30m"]   = rev_30m_list
print(f"  Done. Failures (fallback 0): {failed}/{len(resolved)}")

# ── compute new model probabilities ──────────────────────────────────────────
print("Computing score_to_p_model probabilities …")

new_p_yes_list = []
new_p_no_list  = []

for _, row in resolved.iterrows():
    tau    = float(row.get("tau_minutes", 60) or 60)
    vol_pm = float(row.get("vol_60m", 0.001) or 0.001)
    if pd.isna(vol_pm) or vol_pm <= 0:
        vol_pm = 0.001

    sigma_tau = vol_pm * math.sqrt(tau)
    spot   = float(row["spot"])
    strike = float(row["strike"])

    # 1h composite scores (from logged columns)
    trend_1h = int(row.get("composite_trend", 0) if pd.notna(row.get("composite_trend")) else 0)
    rev_1h   = int(row.get("composite_rev",   0) if pd.notna(row.get("composite_rev"))   else 0)
    trend_30 = int(row["trend_30m"])
    rev_30   = int(row["rev_30m"])

    # Tau-blended p_up (now re-enabled for ETH)
    p_up_blend = lookup_p_up_blended(
        trend_1h, rev_1h, trend_30, rev_30, tau, asset=ASSET
    )

    if sigma_tau > 0:
        p_yes = score_to_p_model(
            trend_1h, rev_1h, spot, strike, sigma_tau,
            asset=ASSET, p_up_override=p_up_blend,
        )
        p_no = score_to_p_no_model(
            trend_1h, rev_1h, spot, strike, sigma_tau,
            asset=ASSET, p_up_override=p_up_blend,
        )
    else:
        p_yes = 0.5
        p_no  = 0.5

    new_p_yes_list.append(p_yes)
    new_p_no_list.append(p_no)

resolved["new_p_yes"] = new_p_yes_list
resolved["new_p_no"]  = new_p_no_list

# ── gate stack (mirror existing live gates for ETH) ───────────────────────────
# Using logged signal columns. Gates applied identically to live runner.
def apply_gates(row, side, p_model, p_mkt):
    """Return True if the trade should be blocked by gates."""
    ema  = str(row.get("ema_alignment", "neutral") or "neutral")
    p_up = float(row.get("composite_p_up", 0.5) or 0.5)
    off  = float(row.get("offset_pct", 0) or 0)
    pm   = float(p_mkt)

    # OTM YES gate (existing live gate)
    if side == "yes" and off > 0 and pm < 0.45 and p_up <= 0.50:
        return True, "otm_yes"

    # OTM NO block when bullish (existing live gate)
    if side == "no" and off > 0 and p_up > 0.50:
        return True, "otm_no_bullish"

    return False, ""

# ── evaluate new model decisions ──────────────────────────────────────────────
new_decisions = []
for _, row in resolved.iterrows():
    pm     = float(row["p_market"])
    side   = str(row["side"])
    p_yes  = float(row["new_p_yes"])
    p_no   = float(row["new_p_no"])
    outcome = int(row["resolved_yes"])  # 1 = YES resolved

    # Determine new model edge for each side
    edge_yes = p_yes - pm
    edge_no  = p_no - (1.0 - pm)

    # Choose side: prefer the side with higher edge if both positive
    if edge_yes > edge_no and edge_yes > MIN_EDGE:
        chosen_side = "yes"
        p_model_chosen = p_yes
        pm_chosen = pm
        blocked, gate_reason = apply_gates(row, "yes", p_yes, pm)
    elif edge_no > MIN_EDGE:
        chosen_side = "no"
        p_model_chosen = p_no
        pm_chosen = 1.0 - pm
        blocked, gate_reason = apply_gates(row, "no", p_no, pm)
    else:
        chosen_side = None
        blocked = False
        gate_reason = "no_edge"

    if chosen_side is None or blocked:
        new_decisions.append({
            "new_decision": "no_trade",
            "new_side": chosen_side,
            "new_edge": max(edge_yes, edge_no),
            "gate_reason": gate_reason,
            "new_win": None,
            "new_pnl": 0.0,
        })
    else:
        win = (outcome == 1) if chosen_side == "yes" else (outcome == 0)
        bet = float(row.get("bet_amount", 5.0) or 5.0)
        if chosen_side == "yes":
            pnl = (1.0 - pm) * bet / pm if win else -bet
        else:
            pnl = pm * bet / (1.0 - pm) if win else -bet
        new_decisions.append({
            "new_decision": "trade",
            "new_side": chosen_side,
            "new_edge": max(edge_yes, edge_no),
            "gate_reason": "",
            "new_win": int(win),
            "new_pnl": round(pnl, 2),
        })

new_df = pd.DataFrame(new_decisions)
resolved = pd.concat([resolved.reset_index(drop=True), new_df], axis=1)

# ── helpers ───────────────────────────────────────────────────────────────────
def bew(sub, pnl_col, win_col):
    w = sub[sub[win_col] == 1][pnl_col]
    l = sub[sub[win_col] == 0][pnl_col]
    if w.empty or l.empty:
        return float("nan")
    return abs(l.mean()) / (w.mean() + abs(l.mean()))

def stats(sub, label, pnl_col="would_pnl", win_col="would_win"):
    if sub.empty:
        return
    n   = len(sub)
    wr  = sub[win_col].mean()
    pnl = sub[pnl_col].sum()
    b   = bew(sub, pnl_col, win_col)
    pp  = (wr - b) * 100 if b == b else float("nan")
    print(f"  {label}: n={n}  WR={wr:.1%}  P&L={pnl:+.2f}  BEW={b:.1%}  vs={pp:+.1f}pp")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CURRENT MODEL baseline
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 1 — CURRENT MODEL (direct_p_model) baseline")
print("=" * 72)
stats(resolved, "ALL ", "would_pnl", "would_win")
stats(resolved[resolved["side"] == "yes"], "YES ", "would_pnl", "would_win")
stats(resolved[resolved["side"] == "no"],  "NO  ", "would_pnl", "would_win")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NEW MODEL overview
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"SECTION 2 — NEW MODEL (score_to_p_model k_yes={K_DRIFT_YES} k_no={K_DRIFT_NO} + 30m blend)")
print("=" * 72)

new_traded = resolved[resolved["new_decision"] == "trade"]
new_no_trade = resolved[resolved["new_decision"] == "no_trade"]

print(f"  Trades taken: {len(new_traded)}/{len(resolved)}  "
      f"({len(new_no_trade)} dropped — no_edge or gated)")

stats(new_traded, "ALL ", "new_pnl", "new_win")
stats(new_traded[new_traded["new_side"] == "yes"], "YES ", "new_pnl", "new_win")
stats(new_traded[new_traded["new_side"] == "no"],  "NO  ", "new_pnl", "new_win")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPARISON: trades current took but new skips / new adds
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 3 — COMPARISON BREAKDOWN")
print("=" * 72)

# Map original side to new decision
both_trade = resolved[resolved["new_decision"] == "trade"]
old_only   = resolved[resolved["new_decision"] == "no_trade"]   # current took, new skips

print(f"\n  [A] Both models trade: n={len(both_trade)}")
stats(both_trade, "  current", "would_pnl", "would_win")
stats(both_trade, "  new    ", "new_pnl",   "new_win")

print(f"\n  [B] Current model took — new model SKIPS: n={len(old_only)}")
stats(old_only, "  skipped impact", "would_pnl", "would_win")

# Summary delta
curr_pnl = resolved["would_pnl"].sum()
new_pnl  = new_traded["new_pnl"].sum()
print(f"\n  Current total P&L : {curr_pnl:+.2f}")
print(f"  New model P&L     : {new_pnl:+.2f}")
print(f"  Delta             : {new_pnl - curr_pnl:+.2f}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — P_UP BLEND IMPACT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 4 — 30m P_UP BLEND IMPACT")
print("=" * 72)

def _safe_int(v, default=0):
    try:
        return int(v) if pd.notna(v) else default
    except Exception:
        return default

resolved["p_up_1h"]    = resolved.apply(
    lambda r: lookup_p_up(_safe_int(r.get("composite_trend")),
                          _safe_int(r.get("composite_rev")), asset=ASSET), axis=1)
resolved["p_up_blend"] = resolved.apply(
    lambda r: lookup_p_up_blended(
        _safe_int(r.get("composite_trend")), _safe_int(r.get("composite_rev")),
        int(r["trend_30m"]), int(r["rev_30m"]),
        float(r.get("tau_minutes", 60) or 60), asset=ASSET), axis=1)

resolved["p_up_delta"] = resolved["p_up_blend"] - resolved["p_up_1h"]

print(f"  Mean p_up delta (blend − 1h): {resolved['p_up_delta'].mean():+.4f}")
print(f"  Std:  {resolved['p_up_delta'].std():.4f}")
print(f"  Non-zero deltas: {(resolved['p_up_delta'] != 0).sum()}/{len(resolved)}")

for lo, hi in [(0, 30), (30, 60), (60, 120)]:
    sub = resolved[(resolved["tau_minutes"] >= lo) & (resolved["tau_minutes"] < hi)]
    if not sub.empty:
        print(f"  tau=[{lo},{hi})  n={len(sub)}  mean_delta={sub['p_up_delta'].mean():+.4f}")

print("\nDone.")
