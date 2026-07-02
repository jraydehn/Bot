#!/usr/bin/env python3
"""
simulate_eth_bp_body.py

Simulate 5m buying pressure (bp_5m) and 15m body ratio (body_15m) signals
against the full ETH paper trade archive.

Same methodology as simulate_bp_body.py (BTC). ETH-specific thresholds may
differ — this script finds them.

Run: python3 simulate_eth_bp_body.py
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
PARQUET = max(
    DATA_DIR.glob("binanceus_ETHUSDT_1m_2024-01-01_*.parquet"),
    key=lambda p: p.stem.split("_")[-1],
)
SEP  = "=" * 80
SEP2 = "-" * 80


# ── data loading ───────────────────────────────────────────────────────────────

def load_candles() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading ETH candles from {PARQUET.name}...", end=" ", flush=True)
    df1m = pd.read_parquet(PARQUET)
    if not isinstance(df1m.index, pd.DatetimeIndex):
        df1m.index = pd.to_datetime(df1m.index, utc=True)
    elif df1m.index.tz is None:
        df1m.index = df1m.index.tz_localize("UTC")
    print(f"{len(df1m):,} bars")

    df5 = df1m.resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna()
    rng5 = df5["high"] - df5["low"]
    df5["bp_5m"] = np.where(rng5 > 0, (df5["close"] - df5["low"]) / rng5, 0.5)

    df15 = df1m.resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna()
    rng15 = df15["high"] - df15["low"]
    df15["body_15m"] = np.where(rng15 > 0,
                                (df15["close"] - df15["open"]).abs() / rng15, 0.0)
    df15["dir_15m"]  = np.where(df15["close"] >= df15["open"], 1, -1)

    return df5[["bp_5m"]], df15[["body_15m", "dir_15m"]]


def load_trades() -> pd.DataFrame:
    frames = []
    for path in glob.glob("results/paper_trades*.csv"):
        try:
            chunk = pd.read_csv(path, low_memory=False)
            chunk = chunk[chunk["would_win"].notna()].copy()
            if "contract_ticker" not in chunk.columns:
                continue
            chunk = chunk[chunk["contract_ticker"].str.startswith("KXETH", na=False)]
            if len(chunk):
                frames.append(chunk)
        except Exception:
            pass
    df = pd.concat(frames).drop_duplicates(subset=["contract_ticker", "decision_time"]).copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"], format="mixed", utc=True)
    df = df.sort_values("decision_time").reset_index(drop=True)
    num_cols = [
        "offset_pct", "p_market", "p_yes_model", "composite_p_up",
        "composite_rev", "composite_trend", "stoch_k", "ema_stack_bias",
        "vwap_stretch_score", "vwap_distance_pct", "chg_30m", "chg_10m", "chg_5m",
        "vol_score", "tau_minutes", "vpin_score", "vpin_raw",
        "would_pnl", "would_win", "confirmation_score", "ema_stretch_score",
        "funding_bias", "smc_1h", "smc_4h",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def join_signals(trades: pd.DataFrame,
                 df5: pd.DataFrame,
                 df15: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    bp_floor   = trades["decision_time"].dt.floor("5min")  - pd.Timedelta(minutes=5)
    body_floor = trades["decision_time"].dt.floor("15min") - pd.Timedelta(minutes=15)

    trades["bp_5m"]    = bp_floor.map(df5["bp_5m"])
    trades["body_15m"] = body_floor.map(df15["body_15m"])
    trades["dir_15m"]  = body_floor.map(df15["dir_15m"])

    bp_prev   = bp_floor   - pd.Timedelta(minutes=5)
    body_prev = body_floor - pd.Timedelta(minutes=15)
    trades["bp_5m"]    = trades["bp_5m"].fillna(bp_prev.map(df5["bp_5m"]))
    trades["body_15m"] = trades["body_15m"].fillna(body_prev.map(df15["body_15m"]))
    trades["dir_15m"]  = trades["dir_15m"].fillna(body_prev.map(df15["dir_15m"]))

    matched = trades["bp_5m"].notna().sum()
    print(f"  bp_5m matched: {matched}/{len(trades)} ({matched/len(trades):.1%})")
    return trades


# ── stats helpers ──────────────────────────────────────────────────────────────

def _stats(sub: pd.DataFrame, label: str = "", min_n: int = 0):
    n = len(sub)
    if n < min_n:
        return
    if n == 0:
        print(f"  {label:<55}  n=   0")
        return
    wr  = sub["would_win"].mean()
    be  = sub["p_market"].mean()
    pnl = sub["would_pnl"].sum()
    d   = wr - be
    w   = int(sub["would_win"].sum())
    flag = " ★" if abs(d) > 0.05 and n >= 15 else ""
    print(f"  {label:<55}  n={n:>4}  W={w:>3}  WR={wr:.1%}  BE={be:.1%}  "
          f"Δ={d:>+.1%}  P&L=${pnl:>+,.0f}{flag}")


# ── Section 1: reliability tables ─────────────────────────────────────────────

def section_reliability(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 1 — Reliability tables")
    print(SEP)

    for side_label, side_val in [("YES", "yes"), ("NO", "no")]:
        sub = df[df["side"] == side_val].copy()
        print(f"\n  {side_label} trades ({len(sub)}) — bp_5m buckets:")
        bp_bins = [0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.01]
        bp_lbls = ["0–0.10","0.10–0.20","0.20–0.30","0.30–0.40","0.40–0.45",
                   "0.45–0.50","0.50–0.55","0.55–0.60","0.60–0.70","0.70–0.80","0.80–1.0"]
        sub["bp_bucket"] = pd.cut(sub["bp_5m"], bins=bp_bins, labels=bp_lbls, right=False)
        for lbl in bp_lbls:
            _stats(sub[sub["bp_bucket"] == lbl], f"  bp={lbl}", min_n=5)

        print(f"\n  {side_label} trades — body_15m buckets:")
        bd_bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
        bd_lbls = ["0–0.10","0.10–0.20","0.20–0.30","0.30–0.40",
                   "0.40–0.50","0.50–0.60","0.60–0.70","0.70–0.80","0.80–1.0"]
        sub["bd_bucket"] = pd.cut(sub["body_15m"], bins=bd_bins, labels=bd_lbls, right=False)
        for lbl in bd_lbls:
            _stats(sub[sub["bd_bucket"] == lbl], f"  body={lbl}", min_n=5)

        print(f"\n  {side_label} trades — 15m candle direction:")
        _stats(sub[sub["dir_15m"] ==  1], "  dir15m=bullish", min_n=5)
        _stats(sub[sub["dir_15m"] == -1], "  dir15m=bearish", min_n=5)


# ── Section 2: cross-tab bp × body ────────────────────────────────────────────

def section_crosstab(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 2 — Cross-tab: bp_5m × body_15m")
    print(SEP)

    for side_label, side_val in [("YES", "yes"), ("NO", "no")]:
        sub = df[df["side"] == side_val].copy()
        sub["bp_q"] = pd.cut(sub["bp_5m"], bins=[0, 0.40, 0.60, 1.01],
                             labels=["bear(<0.40)","neutral(0.40-0.60)","bull(>0.60)"], right=False)
        sub["bd_q"] = pd.cut(sub["body_15m"], bins=[0, 0.25, 0.50, 1.01],
                             labels=["doji(<0.25)","medium(0.25-0.50)","strong(>0.50)"], right=False)
        print(f"\n  {side_label} — bp × body cross-tab (n≥10):")
        for bp_lbl in ["bear(<0.40)", "neutral(0.40-0.60)", "bull(>0.60)"]:
            for bd_lbl in ["doji(<0.25)", "medium(0.25-0.50)", "strong(>0.50)"]:
                cell = sub[(sub["bp_q"] == bp_lbl) & (sub["bd_q"] == bd_lbl)]
                _stats(cell, f"  bp={bp_lbl:<25} body={bd_lbl}", min_n=10)


# ── Section 3: interactions with model features ────────────────────────────────

def section_interactions(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 3 — Interactions: ema_stack_bias, offset_pct, stoch_k")
    print(SEP)

    for side_label, side_val in [("YES", "yes"), ("NO", "no")]:
        sub = df[df["side"] == side_val].copy()
        print(f"\n  {side_label} — bp_5m × ema_stack_bias:")
        for bp_lo, bp_hi, bp_lbl in [(0.0, 0.40, "bear"), (0.40, 0.60, "neutral"), (0.60, 1.01, "bull")]:
            grp = sub[(sub["bp_5m"] >= bp_lo) & (sub["bp_5m"] < bp_hi)]
            if len(grp) < 10:
                continue
            print(f"\n    bp={bp_lbl} ({bp_lo:.2f}–{bp_hi:.2f})  [n={len(grp)}]")
            for ema in [-1, 0, 1]:
                _stats(grp[grp["ema_stack_bias"] == ema], f"      ema={ema:+d}", min_n=8)

        print(f"\n  {side_label} — bp_5m × offset (ITM/OTM):")
        for bp_lo, bp_hi, bp_lbl in [(0.0, 0.40, "bear"), (0.40, 0.60, "neutral"), (0.60, 1.01, "bull")]:
            grp = sub[(sub["bp_5m"] >= bp_lo) & (sub["bp_5m"] < bp_hi)]
            if len(grp) < 10:
                continue
            print(f"    bp={bp_lbl}:")
            _stats(grp[grp["offset_pct"] <= 0], f"      ITM (offset<=0)", min_n=8)
            _stats(grp[grp["offset_pct"] >  0], f"      OTM (offset>0)",  min_n=8)

        print(f"\n  {side_label} — body_15m × ema_stack_bias:")
        for bd_lo, bd_hi, bd_lbl in [(0.0, 0.25, "doji"), (0.25, 0.50, "medium"), (0.50, 1.01, "strong")]:
            grp = sub[(sub["body_15m"] >= bd_lo) & (sub["body_15m"] < bd_hi)]
            if len(grp) < 10:
                continue
            print(f"\n    body={bd_lbl} [n={len(grp)}]")
            for ema in [-1, 0, 1]:
                _stats(grp[grp["ema_stack_bias"] == ema], f"      ema={ema:+d}", min_n=8)


# ── Section 4: gate candidate analysis ────────────────────────────────────────

def section_gate_candidates(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 4 — Gate candidate analysis")
    print(SEP)

    # ---- YES side ----
    yes = df[df["side"] == "yes"].copy()
    yes = yes[yes["bp_5m"].notna() & yes["body_15m"].notna()].copy()
    print(f"\n  YES trades with both signals: n={len(yes)}")
    _stats(yes, "  Baseline (all YES)", min_n=0)

    # BTC gate thresholds applied to ETH — check if they transfer
    print(f"\n  A) BTC gate thresholds on ETH: body∈[0.50,0.60) + bp<0.55")
    btc_block = yes[(yes["body_15m"] >= 0.50) & (yes["body_15m"] < 0.60) & (yes["bp_5m"] < 0.55)]
    _stats(btc_block, "  body[0.50,0.60) + bp<0.55  [BTC gate]", min_n=0)
    if len(btc_block) > 0:
        print(f"  Rescue (bp>=0.55 within that range):")
        rescue = yes[(yes["body_15m"] >= 0.50) & (yes["body_15m"] < 0.60) & (yes["bp_5m"] >= 0.55)]
        _stats(rescue, "  body[0.50,0.60) + bp>=0.55  [rescue]", min_n=0)

    # Low bp (bearish pressure) — YES block candidates
    print(f"\n  B) Block YES when bp_5m < 0.40 (sellers dominated):")
    bp_low = yes[yes["bp_5m"] < 0.40]
    _stats(bp_low, "  Flat block bp<0.40", min_n=5)
    print("  Rescue search within bp<0.40:")
    for col, vals in [("ema_stack_bias", [1]), ("stoch_k", [">50", ">60"]),
                       ("offset_pct", ["<=0"])]:
        if col not in bp_low.columns:
            continue
        for v in vals:
            if isinstance(v, str) and v.startswith(">"):
                t = float(v[1:])
                mask = bp_low[col] > t
                lbl  = f"{col}>{t}"
            elif isinstance(v, str) and v.startswith("<="):
                t = float(v[2:])
                mask = bp_low[col] <= t
                lbl  = f"{col}<={t}"
            else:
                mask = bp_low[col] == v
                lbl  = f"{col}={v}"
            sub = bp_low[mask]
            if len(sub) >= 8:
                wr = sub["would_win"].mean()
                be = sub["p_market"].mean()
                if wr - be > 0.02:
                    _stats(sub, f"    rescue {lbl}", min_n=0)

    # body > 0.50 bearish — YES block candidate
    print(f"\n  C) Block YES when body_15m>0.50 AND dir_15m==-1 (strong bearish candle):")
    strong_bear = yes[(yes["body_15m"] > 0.50) & (yes["dir_15m"] == -1)]
    _stats(strong_bear, "  body>0.50 bearish", min_n=5)
    if len(strong_bear) >= 10:
        print("  Rescue search within strong bearish:")
        for ema in [-1, 0, 1]:
            sub = strong_bear[strong_bear["ema_stack_bias"] == ema]
            _stats(sub, f"    ema={ema:+d}", min_n=8)
        _stats(strong_bear[strong_bear["bp_5m"] >= 0.55], "    bp>=0.55 rescue", min_n=5)
        _stats(strong_bear[strong_bear["bp_5m"] >= 0.60], "    bp>=0.60 rescue", min_n=5)

    # Granular body scan for ETH-specific death zone
    print(f"\n  D) Fine-grained body_15m × bp_5m scan (YES only):")
    for bd_lo, bd_hi in [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]:
        bd_sub = yes[(yes["body_15m"] >= bd_lo) & (yes["body_15m"] < bd_hi)]
        if len(bd_sub) < 10:
            continue
        print(f"\n  body=[{bd_lo},{bd_hi}) n={len(bd_sub)}:")
        _stats(bd_sub, f"  All", min_n=0)
        for bp_lo2, bp_hi2, bp_lbl2 in [(0.0, 0.40, "bp<0.40"), (0.40, 0.55, "bp 0.40-0.55"),
                                          (0.55, 0.70, "bp 0.55-0.70"), (0.70, 1.01, "bp>0.70")]:
            sub = bd_sub[(bd_sub["bp_5m"] >= bp_lo2) & (bd_sub["bp_5m"] < bp_hi2)]
            _stats(sub, f"  {bp_lbl2}", min_n=5)

    # ---- NO side ----
    no = df[df["side"] == "no"].copy()
    no = no[no["bp_5m"].notna() & no["body_15m"].notna()].copy()
    print(f"\n{SEP2}")
    print(f"  NO side analysis  n={len(no)}")
    print(SEP2)
    _stats(no, "  Baseline (all NO)", min_n=0)

    print(f"\n  E) Block NO when bp_5m > 0.60 (buyers dominated — fade risk for NO):")
    bp_high_no = no[no["bp_5m"] > 0.60]
    _stats(bp_high_no, "  Flat block bp>0.60", min_n=5)
    print("  Rescue within bp>0.60:")
    for ema in [-1, 0, 1]:
        sub = bp_high_no[bp_high_no["ema_stack_bias"] == ema]
        _stats(sub, f"    ema={ema:+d}", min_n=8)
    _stats(bp_high_no[bp_high_no["stoch_k"] > 70], "    stoch>70", min_n=5)

    print(f"\n  F) Block NO when body>0.50 AND dir_15m==1 (bullish marubozu):")
    strong_bull_no = no[(no["body_15m"] > 0.50) & (no["dir_15m"] == 1)]
    _stats(strong_bull_no, "  body>0.50 bullish", min_n=5)
    if len(strong_bull_no) >= 10:
        _stats(strong_bull_no[strong_bull_no["bp_5m"] < 0.45], "    bp<0.45 rescue", min_n=5)


# ── Section 5: ETH-specific gate simulation ───────────────────────────────────

def section_gate_simulation(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 5 — Gate simulation: proposed ETH gates vs BTC gates")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    no  = df[df["side"] == "no"].copy()
    yes = yes[yes["bp_5m"].notna() & yes["body_15m"].notna()].copy()
    no  = no[no["bp_5m"].notna() & no["body_15m"].notna()].copy()

    print(f"\n  YES baseline: n={len(yes)}  WR={yes['would_win'].mean():.1%}  "
          f"P&L=${yes['would_pnl'].sum():>+,.0f}")
    print(f"  NO  baseline: n={len(no)}   WR={no['would_win'].mean():.1%}  "
          f"P&L=${no['would_pnl'].sum():>+,.0f}\n")

    # Simulate BTC exact gate on ETH
    def sim_gate(sub, mask, mask_rescue, label):
        blocked  = sub[mask & ~mask_rescue]
        rescued  = sub[mask & mask_rescue]
        kept     = sub[~mask]
        print(f"\n  Gate: {label}")
        _stats(blocked, "  Blocked (would skip)", min_n=0)
        _stats(rescued, "  Rescued (would keep)", min_n=0)
        _stats(kept,    "  Untouched",            min_n=0)
        pnl_saved = -blocked["would_pnl"].sum()
        pnl_kept  = rescued["would_pnl"].sum() + kept["would_pnl"].sum()
        print(f"  P&L saved by block: ${pnl_saved:>+,.0f}   "
              f"P&L in rescued+kept: ${pnl_kept:>+,.0f}")
        net = pnl_saved + (pnl_kept - sub[~blocked.index.isin(sub.index)]["would_pnl"].sum()
                           if False else 0)
        print(f"  Net gate impact: ${pnl_saved:>+,.0f} (losses avoided from blocked trades)")
        return blocked, rescued

    # BTC gate (body[0.50,0.60) + bp<0.55 block; rescue bp>=0.55)
    gate_btc = (yes["body_15m"] >= 0.50) & (yes["body_15m"] < 0.60) & (yes["bp_5m"] < 0.55)
    rescue_btc = (yes["body_15m"] >= 0.50) & (yes["body_15m"] < 0.60) & (yes["bp_5m"] >= 0.55)
    blocked_btc, rescued_btc = sim_gate(yes, gate_btc, rescue_btc, "BTC exact gate (body[0.50,0.60)+bp<0.55)")

    # Candidate gate A: bp<0.40 block (broader)
    gate_a = yes["bp_5m"] < 0.40
    rescue_a = (yes["bp_5m"] < 0.40) & (yes["ema_stack_bias"] == 1)
    blocked_a, rescued_a = sim_gate(yes, gate_a, rescue_a, "ETH Gate A: bp<0.40, rescue ema=+1")

    # Candidate gate B: body>0.50 bearish + bp<0.55
    gate_b = (yes["body_15m"] > 0.50) & (yes["dir_15m"] == -1) & (yes["bp_5m"] < 0.55)
    rescue_b = (yes["body_15m"] > 0.50) & (yes["dir_15m"] == -1) & (yes["bp_5m"] >= 0.55)
    blocked_b, rescued_b = sim_gate(yes, gate_b, rescue_b, "ETH Gate B: body>0.50 bearish + bp<0.55")

    # Candidate gate C: body[0.40,0.65) + bp<0.50
    gate_c = (yes["body_15m"] >= 0.40) & (yes["body_15m"] < 0.65) & (yes["bp_5m"] < 0.50)
    rescue_c = ((yes["body_15m"] >= 0.40) & (yes["body_15m"] < 0.65)
                & (yes["bp_5m"] < 0.50) & (yes["ema_stack_bias"] == 1))
    blocked_c, rescued_c = sim_gate(yes, gate_c, rescue_c, "ETH Gate C: body[0.40,0.65)+bp<0.50, rescue ema=+1")

    # NO side: mirror gate
    print(f"\n  NO side — Gate mirror: bp>0.60 + body>0.50 bullish block:")
    gate_no = (no["bp_5m"] > 0.60) & (no["body_15m"] > 0.50) & (no["dir_15m"] == 1)
    rescue_no = gate_no & (no["ema_stack_bias"] == -1)
    blocked_no = no[gate_no & ~rescue_no]
    rescued_no = no[gate_no & rescue_no]
    _stats(blocked_no, "  Blocked", min_n=0)
    _stats(rescued_no, "  Rescued", min_n=0)
    print(f"  P&L saved: ${-blocked_no['would_pnl'].sum():>+,.0f}")


# ── Section 6: time-split stability ──────────────────────────────────────────

def section_time_split(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  SECTION 6 — Time-split stability")
    print(SEP)

    yes = df[df["side"] == "yes"].copy()
    yes = yes[yes["bp_5m"].notna()].copy()

    cutoff = yes["decision_time"].median()
    early  = yes[yes["decision_time"] <= cutoff]
    recent = yes[yes["decision_time"] >  cutoff]
    print(f"  Split at {cutoff.date()}  early=n{len(early)}  recent=n{len(recent)}")

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
    print("  ETH — bp_5m + body_15m Signal Analysis")
    print(SEP)

    df5, df15 = load_candles()

    print("Loading ETH paper trades...", end=" ", flush=True)
    trades = load_trades()
    yes_n = (trades["side"] == "yes").sum()
    no_n  = (trades["side"] == "no").sum()
    print(f"{len(trades)} resolved  (YES={yes_n}  NO={no_n})")
    print(f"  Date range: {trades['decision_time'].min().date()} → {trades['decision_time'].max().date()}")
    print(f"  Win rate:  YES={trades[trades['side']=='yes']['would_win'].mean():.1%}  "
          f"NO={trades[trades['side']=='no']['would_win'].mean():.1%}")
    print(f"  P&L:       YES=${trades[trades['side']=='yes']['would_pnl'].sum():>+,.0f}  "
          f"NO=${trades[trades['side']=='no']['would_pnl'].sum():>+,.0f}")

    trades = join_signals(trades, df5, df15)

    section_reliability(trades)
    section_crosstab(trades)
    section_interactions(trades)
    section_gate_candidates(trades)
    section_gate_simulation(trades)
    section_time_split(trades)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
