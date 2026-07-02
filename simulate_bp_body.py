#!/usr/bin/env python3
"""
simulate_bp_body.py

Simulate 5m buying pressure (bp_5m) and 15m body ratio (body_15m) signals
against the full BTC YES paper trade archive.

Definitions:
  bp_5m     = (close - low) / (high - low) on last completed 5m candle
  body_15m  = |close - open| / (high - low) on last completed 15m candle

Analysis plan:
  1. Reliability tables (decile buckets)
  2. Cross-tab: bp × body
  3. Interaction with ema_stack_bias, offset_pct, stoch_k, composite_p_up
  4. Gate candidate analysis (which buckets are natural blocks/rescues)
  5. Integration option comparison

Run: python3 simulate_bp_body.py
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PARQUET = max(
    Path("data").glob("binanceus_BTCUSDT_1m_2024-01-01_*.parquet"),
    key=lambda p: p.stem.split("_")[-1],
)
SEP  = "=" * 80
SEP2 = "-" * 80


# ── data loading ──────────────────────────────────────────────────────────────

def load_candles() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading candles from {PARQUET.name}...", end=" ", flush=True)
    df1m = pd.read_parquet(PARQUET)
    if not isinstance(df1m.index, pd.DatetimeIndex):
        df1m.index = pd.to_datetime(df1m.index, utc=True)
    elif df1m.index.tz is None:
        df1m.index = df1m.index.tz_localize("UTC")
    print(f"{len(df1m):,} bars")

    # 5m: bp = (close - low) / (high - low)
    df5 = df1m.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    rng5 = df5["high"] - df5["low"]
    df5["bp_5m"] = np.where(rng5 > 0, (df5["close"] - df5["low"]) / rng5, 0.5)

    # 15m: body_ratio = |close - open| / (high - low)
    df15 = df1m.resample("15min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    rng15 = df15["high"] - df15["low"]
    df15["body_15m"] = np.where(rng15 > 0,
                                (df15["close"] - df15["open"]).abs() / rng15, 0.0)
    df15["dir_15m"]  = np.where(df15["close"] >= df15["open"], 1, -1)  # 1=bullish, -1=bearish

    return df5[["bp_5m"]], df15[["body_15m", "dir_15m"]]


def load_trades() -> pd.DataFrame:
    frames = []
    for path in glob.glob("results/paper_trades*.csv"):
        try:
            chunk = pd.read_csv(path)
            chunk = chunk[chunk["would_win"].notna()].copy()
            if "contract_ticker" not in chunk.columns:
                continue
            chunk = chunk[chunk["contract_ticker"].str.startswith("KXBTC", na=False)]
            frames.append(chunk)
        except Exception:
            pass
    df = pd.concat(frames).drop_duplicates(subset=["contract_ticker", "decision_time"]).copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"], format="mixed", utc=True)
    num_cols = ["offset_pct", "p_market", "p_yes_model", "composite_p_up",
                "composite_rev", "composite_trend", "stoch_k", "ema_stack_bias",
                "vwap_stretch_score", "chg_30m", "chg_5m", "vol_score",
                "tau_minutes", "vpin_raw", "vpin_score", "would_pnl",
                "confirmation_score", "ema_stretch_score"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def join_signals(trades: pd.DataFrame,
                 df5: pd.DataFrame,
                 df15: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()

    # Use last COMPLETED bar → floor to bar start, then subtract one bar
    bp_floor    = trades["decision_time"].dt.floor("5min") - pd.Timedelta(minutes=5)
    body_floor  = trades["decision_time"].dt.floor("15min") - pd.Timedelta(minutes=15)

    trades["bp_5m"]    = bp_floor.map(df5["bp_5m"])
    trades["body_15m"] = body_floor.map(df15["body_15m"])
    trades["dir_15m"]  = body_floor.map(df15["dir_15m"])

    # Fallback: one bar earlier if missing
    bp_prev   = bp_floor   - pd.Timedelta(minutes=5)
    body_prev = body_floor - pd.Timedelta(minutes=15)
    trades["bp_5m"]    = trades["bp_5m"].fillna(bp_prev.map(df5["bp_5m"]))
    trades["body_15m"] = trades["body_15m"].fillna(body_prev.map(df15["body_15m"]))
    trades["dir_15m"]  = trades["dir_15m"].fillna(body_prev.map(df15["dir_15m"]))

    matched = trades["bp_5m"].notna().sum()
    print(f"  bp_5m matched: {matched}/{len(trades)} trades "
          f"({matched/len(trades):.1%})")
    return trades


# ── stats helpers ─────────────────────────────────────────────────────────────

def _stats(sub: pd.DataFrame, label: str = "", min_n: int = 0):
    n = len(sub)
    if n < min_n:
        return
    wr  = sub["would_win"].mean()
    be  = sub["p_market"].mean()
    pnl = sub["would_pnl"].sum()
    d   = wr - be
    w   = int(sub["would_win"].sum())
    flag = " ★" if abs(d) > 0.05 and n >= 15 else ""
    print(f"  {label:<50}  n={n:>4}  W={w:>3}  WR={wr:.1%}  BE={be:.1%}  "
          f"Δ={d:>+.1%}  P&L=${pnl:>+,.0f}{flag}")


# ── Section 1: reliability tables ─────────────────────────────────────────────

def section_reliability(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 1 — Reliability tables (YES trades only)")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    print(f"\n  Total BTC YES trades with bp_5m: {yes['bp_5m'].notna().sum()}")

    # bp_5m decile buckets
    print(f"\n  bp_5m buckets (0=all-sellers, 1=all-buyers):")
    bp_bins  = [0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.01]
    bp_lbls  = ["0–0.10","0.10–0.20","0.20–0.30","0.30–0.40","0.40–0.45",
                "0.45–0.50","0.50–0.55","0.55–0.60","0.60–0.70","0.70–0.80","0.80–1.0"]
    yes["bp_bucket"] = pd.cut(yes["bp_5m"], bins=bp_bins, labels=bp_lbls, right=False)
    for lbl in bp_lbls:
        sub = yes[yes["bp_bucket"] == lbl]
        _stats(sub, f"  bp={lbl}", min_n=5)

    # body_15m buckets
    print(f"\n  body_15m buckets (0=doji, 1=full-body marubozu):")
    bd_bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    bd_lbls = ["0–0.10","0.10–0.20","0.20–0.30","0.30–0.40",
               "0.40–0.50","0.50–0.60","0.60–0.70","0.70–0.80","0.80–1.0"]
    yes["bd_bucket"] = pd.cut(yes["body_15m"], bins=bd_bins, labels=bd_lbls, right=False)
    for lbl in bd_lbls:
        sub = yes[yes["bd_bucket"] == lbl]
        _stats(sub, f"  body={lbl}", min_n=5)

    # 15m direction
    print(f"\n  15m candle direction (YES trades):")
    for d, lbl in [(1, "bullish (close>open)"), (-1, "bearish (close<open)")]:
        sub = yes[yes["dir_15m"] == d]
        _stats(sub, f"  dir15m={lbl}", min_n=5)


# ── Section 2: cross-tab bp × body ────────────────────────────────────────────

def section_crosstab(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 2 — Cross-tab: bp_5m × body_15m (YES only, n≥10)")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    yes["bp_q"]  = pd.cut(yes["bp_5m"],    bins=[0, 0.40, 0.60, 1.01],
                          labels=["bear(<0.40)","neutral(0.40-0.60)","bull(>0.60)"], right=False)
    yes["bd_q"]  = pd.cut(yes["body_15m"], bins=[0, 0.25, 0.50, 1.01],
                          labels=["doji(<0.25)","medium(0.25-0.50)","strong(>0.50)"], right=False)

    for bp_lbl in ["bear(<0.40)", "neutral(0.40-0.60)", "bull(>0.60)"]:
        for bd_lbl in ["doji(<0.25)", "medium(0.25-0.50)", "strong(>0.50)"]:
            sub = yes[(yes["bp_q"] == bp_lbl) & (yes["bd_q"] == bd_lbl)]
            _stats(sub, f"  bp={bp_lbl:<25} body={bd_lbl}", min_n=10)


# ── Section 3: interaction with existing model features ───────────────────────

def section_interactions(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 3 — Interactions with ema_stack_bias, offset_pct, stoch_k")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()

    for bp_lo, bp_hi, bp_lbl in [(0.0, 0.40, "bear"),
                                  (0.40, 0.60, "neutral"),
                                  (0.60, 1.01, "bull")]:
        sub = yes[(yes["bp_5m"] >= bp_lo) & (yes["bp_5m"] < bp_hi)]
        if len(sub) < 10:
            continue
        print(f"\n  bp_5m={bp_lbl} ({bp_lo:.2f}–{bp_hi:.2f})  [n={len(sub)}]")
        for ema in [-1, 0, 1]:
            _stats(sub[sub["ema_stack_bias"] == ema], f"    ema={ema:+d}", min_n=8)

    print()
    for bp_lo, bp_hi, bp_lbl in [(0.0, 0.40, "bear"),
                                  (0.40, 0.60, "neutral"),
                                  (0.60, 1.01, "bull")]:
        sub = yes[(yes["bp_5m"] >= bp_lo) & (yes["bp_5m"] < bp_hi)]
        if len(sub) < 10:
            continue
        print(f"\n  bp_5m={bp_lbl}  ×  offset (ITM/OTM):")
        _stats(sub[sub["offset_pct"] <= 0],  f"    ITM (offset<=0)", min_n=8)
        _stats(sub[sub["offset_pct"] >  0],  f"    OTM (offset>0)",  min_n=8)

    print()
    print("  body_15m × ema_stack_bias:")
    for bd_lo, bd_hi, bd_lbl in [(0.0, 0.25, "doji"),
                                   (0.25, 0.50, "medium"),
                                   (0.50, 1.01, "strong")]:
        sub = yes[(yes["body_15m"] >= bd_lo) & (yes["body_15m"] < bd_hi)]
        if len(sub) < 10:
            continue
        print(f"\n  body={bd_lbl}  [n={len(sub)}]")
        for ema in [-1, 0, 1]:
            _stats(sub[sub["ema_stack_bias"] == ema], f"    ema={ema:+d}", min_n=8)


# ── Section 4: gate candidate analysis ────────────────────────────────────────

def section_gate_candidates(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 4 — Gate candidate analysis")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()

    # High bp_5m YES gate candidate
    print("\n  A) Block YES when bp_5m >= 0.60 (buyers just dominated — fade risk):")
    block = yes[yes["bp_5m"] >= 0.60]
    _stats(block, "  Flat block (bp>=0.60)", min_n=5)

    # Rescue search within high bp block
    print("  Rescue within bp>=0.60:")
    for col, op, thresholds in [
        ("ema_stack_bias", "==", [-1, 0, 1]),
        ("offset_pct",     "<=", [0, -0.05]),
        ("stoch_k",        "<=", [30, 40, 50]),
        ("composite_p_up", ">=", [0.60, 0.65, 0.70]),
        ("body_15m",       "<=", [0.25, 0.30]),
    ]:
        if col not in block.columns:
            continue
        for t in thresholds:
            if op == ">=":
                mask = block[col] >= t
            elif op == "<=":
                mask = block[col] <= t
            else:
                mask = block[col] == t
            sub = block[mask]
            if len(sub) < 8:
                continue
            wr = sub["would_win"].mean()
            be = sub["p_market"].mean()
            pnl = sub["would_pnl"].sum()
            d = wr - be
            if d > 0.03:
                print(f"    {col}{op}{t}: n={len(sub)} WR={wr:.1%} BE={be:.1%} "
                      f"Δ={d:>+.1%} P&L=${pnl:>+,.0f} ★")

    # Low bp_5m YES gate candidate
    print("\n  B) Block YES when bp_5m <= 0.30 (sellers dominated — bearish):")
    block_low = yes[yes["bp_5m"] <= 0.30]
    _stats(block_low, "  Flat block (bp<=0.30)", min_n=5)

    # Strong body bearish gate
    print("\n  C) Block YES when body_15m >= 0.60 AND dir_15m == -1 (strong bearish candle):")
    block_strong_bear = yes[(yes["body_15m"] >= 0.60) & (yes["dir_15m"] == -1)]
    _stats(block_strong_bear, "  body>=0.60 bearish", min_n=5)

    # Doji bullish vs bearish
    print("\n  D) Doji candle breakdown:")
    doji = yes[yes["body_15m"] < 0.20]
    _stats(doji[doji["dir_15m"] ==  1], "  Doji + bullish direction", min_n=5)
    _stats(doji[doji["dir_15m"] == -1], "  Doji + bearish direction", min_n=5)


# ── Section 5: integration comparison ────────────────────────────────────────

def section_integration(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 5 — Integration option comparison")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    yes = yes[yes["bp_5m"].notna() & yes["body_15m"].notna()].copy()

    print(f"\n  n={len(yes)} YES trades with both signals")

    # Option A: bp_5m as additive to composite_p_up
    # Map: bear→-0.05, neutral→0, bull→+0.05
    yes["bp_adj"] = pd.cut(yes["bp_5m"], bins=[0, 0.40, 0.60, 1.01],
                           labels=[-0.05, 0.0, 0.05]).astype(float)
    yes["body_adj"] = pd.cut(yes["body_15m"], bins=[0, 0.25, 0.50, 1.01],
                              labels=[0.02, 0.0, -0.02]).astype(float)  # doji=slight UP, strong=slight DOWN

    yes["p_up_adj"] = (yes["composite_p_up"].fillna(0.5)
                       + yes["bp_adj"].fillna(0)
                       + yes["body_adj"].fillna(0)).clip(0.3, 0.9)

    print("\n  Option A: additive bp_adj + body_adj to composite_p_up")
    print(f"  Mean p_up before: {yes['composite_p_up'].mean():.3f}  "
          f"after: {yes['p_up_adj'].mean():.3f}")
    print(f"  p_up change distribution:")
    delta = yes["p_up_adj"] - yes["composite_p_up"]
    for v in sorted(delta.unique()):
        cnt = (delta == v).sum()
        print(f"    Δ={v:>+.2f}: {cnt:>4} trades")

    # Simulate what gates would change
    # We don't re-run the full gate stack but can see how many trades shift
    # from neutral (0.45-0.60) into higher/lower zones
    before_dead = yes[(yes["composite_p_up"] >= 0.45) & (yes["composite_p_up"] < 0.60)]
    after_above  = yes[yes["p_up_adj"] >= 0.60]
    after_below  = yes[yes["p_up_adj"] < 0.45]
    print(f"\n  Trades in p_up dead zone [0.45-0.60]: {len(before_dead)}")
    print(f"  After adjustment — trades pushed above 0.60: {(yes['p_up_adj'] >= 0.60).sum()}")
    print(f"  After adjustment — trades pushed below 0.45: {(yes['p_up_adj'] < 0.45).sum()}")

    # Option B: simple gate (block YES when bp >= 0.65 AND body >= 0.50 bearish)
    gate_b = (yes["bp_5m"] >= 0.65) & (yes["body_15m"] >= 0.50) & (yes["dir_15m"] == -1)
    sub_b  = yes[gate_b]
    print(f"\n  Option B: gate — block YES when bp>=0.65 AND body>=0.50 AND bearish-15m")
    _stats(sub_b, "  Block sub-set", min_n=5)
    if len(sub_b) >= 5:
        pnl_impact = -sub_b["would_pnl"].sum()
        print(f"  Net P&L impact (blocking these): ${pnl_impact:>+,.0f}")

    # Option C: gate — block YES when bp >= 0.60 AND ema_stack != 1 (no bullish EMA to support)
    gate_c = (yes["bp_5m"] >= 0.60) & (yes["ema_stack_bias"] != 1)
    sub_c  = yes[gate_c]
    print(f"\n  Option C: gate — block YES when bp>=0.60 AND ema_stack!=1")
    _stats(sub_c, "  Block sub-set", min_n=5)
    if len(sub_c) >= 5:
        pnl_impact = -sub_c["would_pnl"].sum()
        print(f"  Net P&L impact: ${pnl_impact:>+,.0f}")
        # Rescue within this gate
        rescue = sub_c[sub_c["stoch_k"] <= 40]
        if len(rescue) >= 5:
            print(f"  Rescue stoch<=40: ", end="")
            _stats(rescue, "stoch<=40", min_n=5)

    # Show existing gate interaction
    print(f"\n  Existing gate overlap check:")
    print(f"  Trades already blocked by vol_score==1 gate that also have bp>=0.60:")
    vs1 = yes[yes["vol_score"] == 1]
    overlap = vs1[vs1["bp_5m"] >= 0.60]
    _stats(overlap, "  vol=1 AND bp>=0.60", min_n=5)


# ── Section 6: time-split stability ──────────────────────────────────────────

def section_time_split(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 6 — Time-split stability (early vs recent)")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    yes = yes[yes["bp_5m"].notna()].copy()

    cutoff = yes["decision_time"].median()
    early  = yes[yes["decision_time"] <= cutoff]
    recent = yes[yes["decision_time"] >  cutoff]
    print(f"  Split at {cutoff.date()}  early={len(early)}  recent={len(recent)}")

    for bp_lo, bp_hi, lbl in [(0.0, 0.40, "bear"), (0.40, 0.60, "neutral"), (0.60, 1.01, "bull")]:
        print(f"\n  bp={lbl} ({bp_lo}–{bp_hi}):")
        e_sub = early[(early["bp_5m"] >= bp_lo) & (early["bp_5m"] < bp_hi)]
        r_sub = recent[(recent["bp_5m"] >= bp_lo) & (recent["bp_5m"] < bp_hi)]
        _stats(e_sub,  f"    early  ({early['decision_time'].min().date()}→{cutoff.date()})", min_n=8)
        _stats(r_sub,  f"    recent ({cutoff.date()}→{recent['decision_time'].max().date()})", min_n=8)

    for bd_lo, bd_hi, lbl in [(0.0, 0.25, "doji"), (0.25, 0.50, "medium"), (0.50, 1.01, "strong")]:
        print(f"\n  body={lbl} ({bd_lo}–{bd_hi}):")
        e_sub = early[(early["body_15m"] >= bd_lo) & (early["body_15m"] < bd_hi)]
        r_sub = recent[(recent["body_15m"] >= bd_lo) & (recent["body_15m"] < bd_hi)]
        _stats(e_sub,  f"    early", min_n=8)
        _stats(r_sub,  f"    recent", min_n=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  BTC YES — bp_5m + body_15m Signal Analysis")
    print(SEP)

    df5, df15 = load_candles()

    print("Loading BTC paper trades...", end=" ", flush=True)
    trades = load_trades()
    yes_total = (trades["side"] == "yes").sum()
    print(f"{len(trades)} resolved  ({yes_total} YES)")

    trades = join_signals(trades, df5, df15)

    section_reliability(trades)
    section_crosstab(trades)
    section_interactions(trades)
    section_gate_candidates(trades)
    section_integration(trades)
    section_time_split(trades)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
