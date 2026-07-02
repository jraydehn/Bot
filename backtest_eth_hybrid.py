"""
backtest_eth_hybrid.py

Hybrid ETH model simulation on 761 resolved paper trades (Apr 2 – May 8 2026).

  YES side : score_to_p_model (log-drift, k=0.80, tau-blended p_up)
  NO side  : direct_p_model  (ML, logged p_yes_model → p_no = 1 − p_yes_model)

Since NO trades were already selected by direct_p_model using the same edge
formula, the NO book is preserved almost entirely. The key change is YES:
replacing the miscalibrated ML YES branch with the regime-adaptive log-drift formula.

Methodology
-----------
  1. Load 761 consolidated paper trades (same as backtest_eth_reform.py)
  2. Reconstruct tau-blended 30m p_up for YES branch
  3. For each YES trade : compute score_to_p_model edge; take if > MIN_EDGE
  4. For each NO  trade : compute direct_p_model edge (p_market − p_yes_model);
                          take if > MIN_EDGE (same rule used live, almost all pass)
  5. Use logged would_pnl / would_win (actual outcomes) — no P&L recomputation needed
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from composite_scorer import (
    score_to_p_model,
    lookup_p_up_blended, lookup_p_up,
    compute_scores_30m,
)

BASE  = Path(__file__).parent
DATA  = BASE / "data"
SYM   = "ETHUSDT"
ASSET = "ETH"

K_DRIFT_YES = 0.80
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
print(f"  {len(resolved)} unique resolved trades  "
      f"({resolved['decision_time'].min().date()} → {resolved['decision_time'].max().date()})")

# ── load Binance data ─────────────────────────────────────────────────────────
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

df_15m_r = df_1m.resample("15min").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()
df_15m_r.index = df_15m_r.index.tz_localize("UTC") \
    if df_15m_r.index.tz is None else df_15m_r.index

cutoff    = pd.Timestamp("2026-03-01", tz="UTC")
df_1m_cut = df_1m[df_1m.index >= cutoff]
df_15m_cut = df_15m_r[df_15m_r.index >= cutoff]

# ── reconstruct 30m scores ────────────────────────────────────────────────────
print(f"Reconstructing 30m composite scores for {len(resolved)} trades …")
trend_30m_list = []
rev_30m_list   = []
failed = 0

for i, row in resolved.iterrows():
    ts        = row["decision_time"]
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

# ── helpers ───────────────────────────────────────────────────────────────────
def _safe_int(v, default=0):
    try:
        return int(v) if pd.notna(v) else default
    except Exception:
        return default

def _safe_float(v, default=0.0):
    try:
        return float(v) if pd.notna(v) else default
    except Exception:
        return default

def bew(sub, pnl_col, win_col):
    w = sub[sub[win_col] == 1][pnl_col]
    l = sub[sub[win_col] == 0][pnl_col]
    if w.empty or l.empty:
        return float("nan")
    return abs(l.mean()) / (w.mean() + abs(l.mean()))

def stats(sub, label, pnl_col="would_pnl", win_col="would_win"):
    if sub.empty:
        print(f"  {label}: n=0")
        return
    n   = len(sub)
    wr  = sub[win_col].mean()
    pnl = sub[pnl_col].sum()
    ppt = pnl / n
    b   = bew(sub, pnl_col, win_col)
    pp  = (wr - b) * 100 if b == b else float("nan")
    bstr = f"{b:.1%}" if b == b else "  —  "
    ppstr = f"{pp:+.1f}pp" if b == b else "  —  "
    print(f"  {label}: n={n}  WR={wr:.1%}  P&L={pnl:+.2f}  (${ppt:+.2f}/t)  BEW={bstr}  vs={ppstr}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BASELINE (direct_p_model for all trades)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 1 — BASELINE: direct_p_model for YES + NO")
print("=" * 72)

yes_base = resolved[resolved["side"] == "yes"]
no_base  = resolved[resolved["side"] == "no"]

stats(resolved,  "ALL")
stats(yes_base,  "YES")
stats(no_base,   "NO ")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HYBRID MODEL evaluation
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"SECTION 2 — HYBRID: score_to_p_model YES (k={K_DRIFT_YES}) + direct_p_model NO")
print("=" * 72)

hybrid_rows = []

for _, row in resolved.iterrows():
    side    = str(row["side"])
    pm      = _safe_float(row["p_market"], 0.5)
    w_pnl   = _safe_float(row.get("would_pnl"), 0.0)
    w_win   = _safe_int(row.get("would_win"), 0)

    if side == "yes":
        # score_to_p_model YES branch
        tau      = _safe_float(row.get("tau_minutes"), 60) or 60
        vol_pm   = _safe_float(row.get("vol_60m"), 0.001) or 0.001
        sigma_tau = vol_pm * math.sqrt(tau)
        spot     = _safe_float(row.get("spot"), 0.0)
        strike   = _safe_float(row.get("strike"), 0.0)
        trend_1h = _safe_int(row.get("composite_trend"))
        rev_1h   = _safe_int(row.get("composite_rev"))
        tr30     = int(row["trend_30m"])
        rv30     = int(row["rev_30m"])

        p_up_blend = lookup_p_up_blended(trend_1h, rev_1h, tr30, rv30, tau, asset=ASSET)

        if sigma_tau > 0 and spot > 0 and strike > 0:
            new_p_yes = score_to_p_model(
                trend_1h, rev_1h, spot, strike, sigma_tau,
                asset=ASSET, p_up_override=p_up_blend,
            )
        else:
            new_p_yes = 0.5

        edge = new_p_yes - pm
        take = edge > MIN_EDGE

        hybrid_rows.append({
            "side": "yes",
            "take": take,
            "model_p": new_p_yes,
            "edge": edge,
            "would_pnl": w_pnl if take else 0.0,
            "would_win": w_win if take else None,
            "p_market": pm,
        })

    else:
        # direct_p_model NO branch: p_no = 1 - p_yes_model_direct
        p_yes_direct = _safe_float(row.get("p_yes_model"), 0.5)
        p_no_direct  = 1.0 - p_yes_direct
        edge         = p_no_direct - (1.0 - pm)   # = pm - p_yes_direct

        take = edge > MIN_EDGE

        hybrid_rows.append({
            "side": "no",
            "take": take,
            "model_p": p_no_direct,
            "edge": edge,
            "would_pnl": w_pnl if take else 0.0,
            "would_win": w_win if take else None,
            "p_market": pm,
        })

hdf = pd.DataFrame(hybrid_rows)
resolved["hyb_take"]   = hdf["take"].values
resolved["hyb_model_p"] = hdf["model_p"].values
resolved["hyb_edge"]   = hdf["edge"].values
resolved["hyb_pnl"]    = hdf["would_pnl"].values
resolved["hyb_win"]    = hdf["would_win"].values

hyb_traded = resolved[resolved["hyb_take"] == True].copy()
hyb_yes    = hyb_traded[hyb_traded["side"] == "yes"]
hyb_no     = hyb_traded[hyb_traded["side"] == "no"]

print(f"  Trades taken: {len(hyb_traded)}/{len(resolved)}  "
      f"(YES: {len(hyb_yes)}, NO: {len(hyb_no)}, "
      f"dropped: {len(resolved) - len(hyb_traded)})")

def stats_hyb(sub, label):
    if sub.empty:
        print(f"  {label}: n=0")
        return
    n   = len(sub)
    wr  = sub["hyb_win"].mean()
    pnl = sub["hyb_pnl"].sum()
    ppt = pnl / n
    wins  = sub[sub["hyb_win"] == 1]["hyb_pnl"]
    loses = sub[sub["hyb_win"] == 0]["hyb_pnl"]
    if not wins.empty and not loses.empty:
        b  = abs(loses.mean()) / (wins.mean() + abs(loses.mean()))
        pp = (wr - b) * 100
        bstr  = f"{b:.1%}"
        ppstr = f"{pp:+.1f}pp"
    else:
        bstr = ppstr = "  —  "
    print(f"  {label}: n={n}  WR={wr:.1%}  P&L={pnl:+.2f}  (${ppt:+.2f}/t)  BEW={bstr}  vs={ppstr}")

stats_hyb(hyb_traded, "ALL")
stats_hyb(hyb_yes,    "YES")
stats_hyb(hyb_no,     "NO ")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DELTA SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 3 — DELTA: baseline vs hybrid")
print("=" * 72)

base_pnl = resolved["would_pnl"].sum()
hyb_pnl  = resolved["hyb_pnl"].sum()

base_yes_pnl = resolved[resolved["side"] == "yes"]["would_pnl"].sum()
base_no_pnl  = resolved[resolved["side"] == "no"]["would_pnl"].sum()
hyb_yes_pnl  = resolved[resolved["side"] == "yes"]["hyb_pnl"].sum()
hyb_no_pnl   = resolved[resolved["side"] == "no"]["hyb_pnl"].sum()

print(f"  {'':25s}  {'Baseline':>12}  {'Hybrid':>12}  {'Delta':>10}")
print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}")
print(f"  {'YES P&L':25s}  {base_yes_pnl:>+12.2f}  {hyb_yes_pnl:>+12.2f}  {hyb_yes_pnl-base_yes_pnl:>+10.2f}")
print(f"  {'NO  P&L':25s}  {base_no_pnl:>+12.2f}  {hyb_no_pnl:>+12.2f}  {hyb_no_pnl-base_no_pnl:>+10.2f}")
print(f"  {'TOTAL P&L':25s}  {base_pnl:>+12.2f}  {hyb_pnl:>+12.2f}  {hyb_pnl-base_pnl:>+10.2f}")

# YES trades dropped by hybrid
yes_dropped = resolved[(resolved["side"] == "yes") & (resolved["hyb_take"] == False)]
yes_kept    = resolved[(resolved["side"] == "yes") & (resolved["hyb_take"] == True)]
no_dropped  = resolved[(resolved["side"] == "no")  & (resolved["hyb_take"] == False)]

print(f"\n  YES trades: {len(yes_base)} total → {len(yes_kept)} kept, {len(yes_dropped)} dropped")
if not yes_dropped.empty:
    print(f"    Dropped YES P&L (what we avoid): {yes_dropped['would_pnl'].sum():+.2f}  "
          f"WR={yes_dropped['would_win'].mean():.1%}")
if not yes_kept.empty:
    print(f"    Kept YES P&L:                    {yes_kept['would_pnl'].sum():+.2f}  "
          f"WR={yes_kept['would_win'].mean():.1%}")

print(f"\n  NO trades: {len(no_base)} total → {len(hyb_no)} kept, {len(no_dropped)} dropped")
if not no_dropped.empty:
    print(f"    Dropped NO  P&L (what we avoid): {no_dropped['would_pnl'].sum():+.2f}  "
          f"WR={no_dropped['would_win'].mean():.1%}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — YES KEPT vs DROPPED breakdown
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 4 — HYBRID YES: kept vs dropped by p_market band")
print("=" * 72)

yes_base_copy = resolved[resolved["side"] == "yes"].copy()

for lo, hi in [(0.05, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.85), (0.85, 0.97)]:
    band = yes_base_copy[(yes_base_copy["p_market"] >= lo) & (yes_base_copy["p_market"] < hi)]
    if band.empty:
        continue
    kept_b = band[band["hyb_take"] == True]
    drop_b = band[band["hyb_take"] == False]
    k_pnl  = kept_b["would_pnl"].sum()
    d_pnl  = drop_b["would_pnl"].sum()
    k_wr   = kept_b["would_win"].mean() if not kept_b.empty else float("nan")
    d_wr   = drop_b["would_win"].mean() if not drop_b.empty else float("nan")
    print(f"  pm=[{lo:.2f},{hi:.2f})  total={len(band)}  "
          f"kept={len(kept_b)} P&L={k_pnl:+.2f} WR={k_wr:.1%}  |  "
          f"dropped={len(drop_b)} P&L={d_pnl:+.2f} WR={d_wr:.1%}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HYBRID YES by composite score (p_up) and edge band
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 5 — HYBRID YES kept: score_to_p_model edge distribution")
print("=" * 72)

for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)]:
    band = hyb_yes[(hyb_yes["hyb_edge"] >= lo) & (hyb_yes["hyb_edge"] < hi)]
    if band.empty:
        continue
    n   = len(band)
    wr  = band["hyb_win"].mean()
    pnl = band["hyb_pnl"].sum()
    ppt = pnl / n
    print(f"  edge=[{lo:.2f},{hi:.2f})  n={n:3d}  WR={wr:.1%}  P&L={pnl:+.2f}  (${ppt:+.2f}/t)")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Direct model era only (Apr 15 onward — cleanest comparison)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 6 — Direct model era only (Apr 15–May 8, cleanest apples-to-apples)")
print("=" * 72)

era_start = pd.Timestamp("2026-04-15", tz="UTC")
era = resolved[resolved["decision_time"] >= era_start].copy()
era_yes = era[era["side"] == "yes"]
era_no  = era[era["side"] == "no"]

era_hyb = era[era["hyb_take"] == True]
era_hyb_yes = era_hyb[era_hyb["side"] == "yes"]
era_hyb_no  = era_hyb[era_hyb["side"] == "no"]

print(f"  Era trades: {len(era)}  (YES={len(era_yes)}, NO={len(era_no)})")
print()
print(f"  BASELINE (direct_p_model):")
stats(era,     "    ALL")
stats(era_yes, "    YES")
stats(era_no,  "    NO ")

print()
print(f"  HYBRID (score_to_p YES + direct_p NO):")
def stats_hyb_sub(sub, label):
    if sub.empty:
        print(f"    {label}: n=0")
        return
    n   = len(sub)
    wr  = sub["hyb_win"].mean()
    pnl = sub["hyb_pnl"].sum()
    ppt = pnl / n
    wins  = sub[sub["hyb_win"] == 1]["hyb_pnl"]
    loses = sub[sub["hyb_win"] == 0]["hyb_pnl"]
    if not wins.empty and not loses.empty:
        b  = abs(loses.mean()) / (wins.mean() + abs(loses.mean()))
        pp = (wr - b) * 100
        bstr  = f"{b:.1%}"
        ppstr = f"{pp:+.1f}pp"
    else:
        bstr = ppstr = "  —  "
    print(f"    {label}: n={n}  WR={wr:.1%}  P&L={pnl:+.2f}  (${ppt:+.2f}/t)  BEW={bstr}  vs={ppstr}")

stats_hyb_sub(era_hyb,     "ALL")
stats_hyb_sub(era_hyb_yes, "YES")
stats_hyb_sub(era_hyb_no,  "NO ")

era_base_pnl = era["would_pnl"].sum()
era_hyb_pnl  = era["hyb_pnl"].sum()
print(f"\n  Era delta: {era_hyb_pnl - era_base_pnl:+.2f}  "
      f"(baseline {era_base_pnl:+.2f} → hybrid {era_hyb_pnl:+.2f})")

print("\nDone.")
