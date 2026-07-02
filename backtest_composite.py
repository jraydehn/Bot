#!/usr/bin/env python3
"""
backtest_composite.py — Comprehensive backtest of the composite scorer + reformed gate logic.

For each hourly bar:
  1. Compute composite scores (trend + reversion) from historical OHLCV
  2. Generate 12 Kalshi-style strike prices at offsets -2% to +2%
  3. Simulate p_market via log-normal with realized 60-minute vol
  4. Apply reformed gates: Gate CS, Gate NS, Gate 3, Gate R:R
  5. Select highest net-edge qualifying bet per hour
  6. Half-Kelly size, track PnL and bankroll

Two windows reported:
  - 2024           : OUT-OF-SAMPLE  (predates composite calibration)
  - 2025-01→2026-04: IN-SAMPLE      (composite calibration period — benchmark only)
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import compute_scores, lookup_p_up, score_to_p_model
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

BANKROLL_0  = 1_000.0
KELLY_MULT  = 0.50
KELLY_CAP   = 0.05       # max 5% of bankroll per bet
MIN_NET_EDGE = 0.03
SLIPPAGE    = DEFAULT_SLIPPAGE
SPREAD      = DEFAULT_SPREAD
TAU_MINUTES = 60.0       # contracts evaluated at bar open, 60 min to expiry

# Strike offsets relative to spot
OFFSETS = [-0.020, -0.015, -0.010, -0.0075, -0.005, -0.0025,
            0.0025,  0.005,  0.0075,  0.010,  0.015,  0.020]

# Gate thresholds (composite path)
GATE_CS_MIN = 0.55   # Gate CS: OTM YES requires p_up ≥ 0.55
GATE_NS_MAX = 0.40   # Gate NS: OTM NO  requires p_up ≤ 0.40
P_MODEL_MIN = 0.04
P_MODEL_MAX = 0.96
RR_MAX_NO   = 4.0
RR_MIN_NO   = 0.33

WINDOWS = {
    "2024 (out-of-sample)":   ("2024-01-01", "2025-01-01"),
    "2025-2026 (in-sample)":  ("2025-01-01", "2026-04-09"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vol_per_min(close_1h: pd.Series) -> pd.Series:
    """Rolling 60-bar realised vol on 1h returns, converted to per-minute units."""
    log_ret = np.log(close_1h / close_1h.shift(1))
    vol_1h  = log_ret.rolling(60).std()
    return (vol_1h / math.sqrt(60)).clip(lower=1e-6)


def _p_mkt(spot: float, strike: float, vol_pm: float) -> float:
    """Log-normal P(close > strike) with 60 min to expiry."""
    sigma_tau = vol_pm * math.sqrt(TAU_MINUTES)
    if sigma_tau <= 0:
        return 0.5
    z = math.log(strike / spot) / sigma_tau
    return float(np.clip(1 - norm.cdf(z), 0.02, 0.98))


def _gates(side: str, p_model: float, p_market: float, p_up: float, offset: float) -> tuple:
    """
    Returns (passes: bool, blocker: str, net_edge: float).
    offset = (strike - spot) / spot: positive = OTM YES / ITM NO, negative = ITM YES / OTM NO.
    Gate CS and NS use offset (not p_market) to determine OTM/ITM correctly.
    """
    # Gate 0a: model saturation
    if not (P_MODEL_MIN <= p_model <= P_MODEL_MAX):
        return False, "Gate0-sat", 0.0
    # Gate 0b: market liquidity
    if not (0.04 <= p_market <= 0.96):
        return False, "Gate0-liq", 0.0

    fee = kalshi_fee(p_market)
    if side == "yes":
        raw = p_model - p_market
        net = raw - fee - SLIPPAGE - SPREAD
        # Gate CS: OTM YES (strike above spot) requires bullish calibration
        if offset > 0 and p_up < GATE_CS_MIN:
            return False, "GateCS", net
    else:
        raw = p_market - p_model
        net = raw - fee - SLIPPAGE - SPREAD
        # Gate NS: OTM NO (strike below spot) requires bearish calibration
        if offset < 0 and p_up > GATE_NS_MAX:
            return False, "GateNS", net
        # Gate R:R for NO
        rr = (1 - p_market) / p_market if p_market > 0 else 999
        if rr < RR_MIN_NO or rr > RR_MAX_NO:
            return False, "GateRR", net

    # Gate 3: minimum net edge
    if net < MIN_NET_EDGE:
        return False, "Gate3", net

    return True, "", net


def _kelly_amount(p_model: float, p_market: float, side: str, bankroll: float) -> float:
    """Half-Kelly bet size capped at KELLY_CAP of bankroll."""
    if side == "yes":
        b = (1 - p_market) / p_market
        p, q = p_model, 1 - p_model
    else:
        p_no = 1 - p_model
        b    = p_market / (1 - p_market)
        p, q = p_no, 1 - p_no
    kelly_f  = max(0.0, (b * p - q) / b)
    fraction = min(kelly_f * KELLY_MULT, KELLY_CAP)
    return round(bankroll * fraction, 2)


def _pnl(bet: float, side: str, p_market: float, won: bool) -> float:
    """Realised PnL including Kalshi fee deducted from winnings."""
    fee_rate = kalshi_fee(p_market)
    if side == "yes":
        if won:
            n_contracts = bet / p_market
            gross       = bet * (1 - p_market) / p_market
            fee         = fee_rate * n_contracts
            return gross - fee
        return -bet
    else:
        if won:
            n_contracts = bet / (1 - p_market)
            gross       = bet * p_market / (1 - p_market)
            fee         = fee_rate * n_contracts
            return gross - fee
        return -bet


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_window(label: str, start: str, end: str,
               df_1h, trend_s, rev_s, vol_pm_s) -> pd.DataFrame:
    mask     = (df_1h.index >= pd.Timestamp(start, tz="UTC")) & \
               (df_1h.index <  pd.Timestamp(end,   tz="UTC"))
    test_idx = df_1h.index[mask]
    if len(test_idx) == 0:
        print(f"  [!] No data for window {label}")
        return pd.DataFrame()

    bankroll = BANKROLL_0
    records  = []

    for ts in test_idx:
        spot    = float(df_1h.loc[ts, "close"])
        next_ts = ts + pd.Timedelta(hours=1)
        if next_ts not in df_1h.index:
            continue
        next_close = float(df_1h.loc[next_ts, "close"])

        trend  = int(trend_s.get(ts, 0))
        rev    = int(rev_s.get(ts, 0))
        p_up   = lookup_p_up(trend, rev)
        vol_pm = float(vol_pm_s.get(ts, 3e-4))
        if vol_pm <= 0:
            vol_pm = 3e-4

        sigma_tau = vol_pm * math.sqrt(TAU_MINUTES)

        candidates = []
        for offset in OFFSETS:
            strike  = spot * (1 + offset)
            pm      = _p_mkt(spot, strike, vol_pm)
            p_model = score_to_p_model(trend, rev, spot, strike, sigma_tau)

            for side in ("yes", "no"):
                passes, blocker, net = _gates(side, p_model, pm, p_up, offset)
                if passes:
                    candidates.append(dict(
                        offset=offset, strike=strike, side=side,
                        p_model=p_model, p_market=pm, net_edge=net,
                    ))

        outcome_yes = int(next_close > spot)   # used for direction tracking
        row_base = dict(ts=ts, spot=spot, next_close=next_close,
                        trend=trend, rev=rev, p_up=p_up,
                        bankroll=round(bankroll, 2))

        if not candidates:
            records.append({**row_base, "decision": "no_trade",
                            "side": None, "offset": None, "strike": None,
                            "p_model": None, "p_market": None, "net_edge": None,
                            "outcome_yes": outcome_yes, "win": None,
                            "bet": 0, "pnl": 0})
            continue

        best = max(candidates, key=lambda x: x["net_edge"])
        bet  = _kelly_amount(best["p_model"], best["p_market"], best["side"], bankroll)

        actual_yes = int(next_close > best["strike"])
        won = (actual_yes == 1) if best["side"] == "yes" else (actual_yes == 0)
        pnl = _pnl(bet, best["side"], best["p_market"], won)
        bankroll = max(1.0, bankroll + pnl)

        records.append({**row_base,
            "decision": "trade",
            "side":     best["side"],
            "offset":   round(best["offset"] * 100, 3),
            "strike":   round(best["strike"], 2),
            "p_model":  round(best["p_model"], 4),
            "p_market": round(best["p_market"], 4),
            "net_edge": round(best["net_edge"], 4),
            "outcome_yes": actual_yes,
            "win":      won,
            "bet":      bet,
            "pnl":      round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    return pd.DataFrame(records)


# ── Summary printer ───────────────────────────────────────────────────────────

def summarise(label: str, df: pd.DataFrame):
    print(f"\n{'═'*58}")
    print(f"  {label}")
    print(f"{'═'*58}")

    trades = df[df["decision"] == "trade"].copy()
    n_hrs  = len(df)
    n_tr   = len(trades)
    if n_tr == 0:
        print("  No trades taken.")
        return

    wins      = trades["win"].sum()
    win_rate  = wins / n_tr
    total_pnl = trades["pnl"].sum()
    final_br  = df["bankroll"].iloc[-1]
    roi       = (final_br - BANKROLL_0) / BANKROLL_0

    print(f"  Period         : {df['ts'].min().date()} → {df['ts'].max().date()}")
    print(f"  Hours evaluated: {n_hrs:,}")
    print(f"  Trades taken   : {n_tr:,}  ({n_tr/n_hrs*100:.1f}% of hours)")
    print(f"  Win rate       : {wins:.0f}/{n_tr}  ({win_rate:.1%})")
    print(f"  Total PnL      : ${total_pnl:+,.2f}")
    print(f"  Final bankroll : ${final_br:,.2f}  (started ${BANKROLL_0:,.0f})")
    print(f"  ROI            : {roi:+.1%}")

    # Average bet and edge
    print(f"  Avg bet size   : ${trades['bet'].mean():.2f}")
    print(f"  Avg net edge   : {trades['net_edge'].mean():+.1%}")

    # By side
    print(f"\n  ── By side ──────────────────────────────────────")
    for side in ("yes", "no"):
        s = trades[trades["side"] == side]
        if len(s) == 0:
            continue
        wr = s["win"].mean()
        print(f"  {side.upper():3s}  n={len(s):4d}  win={wr:.1%}  "
              f"pnl=${s['pnl'].sum():+,.2f}  avg_edge={s['net_edge'].mean():+.1%}")

    # By p_up bin
    print(f"\n  ── By composite_p_up bin ────────────────────────")
    trades["p_up_bin"] = pd.cut(trades["p_up"],
                                bins=[0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
                                labels=["≤0.40","0.40-0.45","0.45-0.50",
                                        "0.50-0.55","0.55-0.60","0.60-0.65",
                                        "0.65-0.70","≥0.70"])
    gb = trades.groupby("p_up_bin", observed=True).agg(
        n=("win","count"), wins=("win","sum"), pnl=("pnl","sum")
    )
    gb["wr"] = gb["wins"] / gb["n"]
    for bin_lbl, row in gb.iterrows():
        if row["n"] == 0:
            continue
        print(f"  p_up {str(bin_lbl):10s}  n={int(row.n):4d}  "
              f"win={row.wr:.1%}  pnl=${row.pnl:+,.0f}")

    # Top (trend, rev) combos
    print(f"\n  ── Top (trend, rev) combos  (n≥15) ─────────────")
    grp = (trades.groupby(["trend","rev"], observed=True)
           .agg(n=("win","count"), wr=("win","mean"), pnl=("pnl","sum"))
           .reset_index()
           .query("n >= 15")
           .sort_values("wr", ascending=False)
           .head(10))
    for _, r in grp.iterrows():
        print(f"  trend={int(r.trend):+d}  rev={int(r.rev):+3d}  "
              f"n={int(r.n):3d}  win={r.wr:.1%}  pnl=${r.pnl:+,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading 1h and 15m data...")
    f_1h  = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_2024-01-01_*.parquet"))[-1]
    f_15m = sorted(DATA_DIR.glob("binanceus_BTCUSDT_15m_2024-01-01_*.parquet"))[-1]
    print(f"  1h  : {f_1h.name}")
    print(f"  15m : {f_15m.name}")

    df_1h  = pd.read_parquet(f_1h)
    df_15m = pd.read_parquet(f_15m)
    df_1h.index  = pd.to_datetime(df_1h.index, utc=True)
    df_15m.index = pd.to_datetime(df_15m.index, utc=True)

    # Approximate 1m from 15m for VWAP (volume distributed evenly across 15 min)
    close_1m  = df_15m["close"].resample("1min").ffill()
    volume_1m = (df_15m["volume"] / 15).resample("1min").ffill().fillna(0)

    # 4h aggregation
    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["close"])

    print("\nComputing composite scores for full history (this takes ~30s)...")
    trend_s, rev_s = compute_scores(
        df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"],
        df_4h["close"], df_4h["high"], df_4h["low"], df_4h["volume"],
        df_15m["close"], df_15m["high"], df_15m["low"],
        close_1m, volume_1m,
        df_1h.index,
    )
    vol_pm_s = _vol_per_min(df_1h["close"])
    print("  Done.")

    all_dfs = {}
    for label, (start, end) in WINDOWS.items():
        print(f"\nRunning window: {label}  ({start} → {end}) ...")
        df_w = run_window(label, start, end, df_1h, trend_s, rev_s, vol_pm_s)
        all_dfs[label] = df_w
        summarise(label, df_w)

    # Save combined results
    combined = []
    for label, df_w in all_dfs.items():
        if len(df_w):
            df_w["window"] = label
            combined.append(df_w)
    if combined:
        out = pd.concat(combined, ignore_index=True)
        out_path = RESULTS_DIR / "backtest_composite.csv"
        out.to_csv(out_path, index=False)
        print(f"\n  Results saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
