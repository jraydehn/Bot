#!/usr/bin/env python3
"""
backtest_real_kalshi.py — Backtest using real Kalshi scan data.

Uses actual scanned contract rows from paper trade archives (real Kalshi strikes
and p_market values). Recomputes composite scores from OHLCV for each decision
hour, then re-applies reformed gate logic.

Up to 4 bets per hour selected by highest net edge. Individual Kelly sizing
capped at 5% of bankroll per bet.
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

BANKROLL_0       = 1_000.0
KELLY_MULT       = 0.50
KELLY_CAP        = 0.05
MAX_BETS_PER_HOUR = 4
MIN_NET_EDGE     = 0.03
SLIPPAGE         = DEFAULT_SLIPPAGE
SPREAD           = DEFAULT_SPREAD

# Gate thresholds
GATE_CS_MIN  = 0.55
GATE_NS_MAX  = 0.40
P_MODEL_MIN  = 0.04
P_MODEL_MAX  = 0.96
P_MARKET_MIN = 0.04
P_MARKET_MAX = 0.96
RR_MAX_NO    = 4.0
RR_MIN_NO    = 0.33

ARCHIVE_GLOB = [
    "results/paper_trades_archive*.csv",
    "results/paper_trades.csv",
    "results/paper_trades_all.csv",
]


# ── Load real scan data ───────────────────────────────────────────────────────

def load_scan_data() -> pd.DataFrame:
    import glob as _glob
    dfs = []
    for pattern in ARCHIVE_GLOB:
        for f in sorted(_glob.glob(pattern)):
            try:
                df = pd.read_csv(f, low_memory=False)
                if all(c in df.columns for c in
                       ["decision_time", "contract_ticker", "strike", "offset_pct",
                        "p_market", "resolved_yes", "spot", "vol_eff", "tau_minutes"]):
                    df["source"] = Path(f).name
                    dfs.append(df)
            except Exception:
                pass

    all_df = pd.concat(dfs, ignore_index=True)

    # BTC only
    all_df = all_df[all_df["contract_ticker"].astype(str).str.startswith("KXBTC")].copy()

    # Resolved only
    all_df = all_df[all_df["resolved_yes"].notna()].copy()

    # Deduplicate: same contract at same decision_time — keep latest source
    all_df = all_df.sort_values("source").drop_duplicates(
        subset=["decision_time", "contract_ticker"], keep="last"
    )

    # Cast types
    for col in ["strike", "offset_pct", "p_market", "spot", "vol_eff",
                "tau_minutes", "resolved_yes"]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    all_df["decision_time"] = pd.to_datetime(all_df["decision_time"], utc=True)
    all_df = all_df.dropna(subset=["strike", "p_market", "spot", "resolved_yes"])
    all_df = all_df.sort_values("decision_time").reset_index(drop=True)
    return all_df


# ── Compute composite scores for all decision hours ───────────────────────────

def build_composite_index(decision_times: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns DataFrame with (decision_time, trend, rev, p_up) for all hours."""
    print("  Loading OHLCV...")
    f_1h  = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_2024-01-01_*.parquet"))[-1]
    f_15m = sorted(DATA_DIR.glob("binanceus_BTCUSDT_15m_2024-01-01_*.parquet"))[-1]

    df_1h  = pd.read_parquet(f_1h)
    df_15m = pd.read_parquet(f_15m)
    df_1h.index  = pd.to_datetime(df_1h.index, utc=True)
    df_15m.index = pd.to_datetime(df_15m.index, utc=True)

    close_1m  = df_15m["close"].resample("1min").ffill()
    volume_1m = (df_15m["volume"] / 15).resample("1min").ffill().fillna(0)

    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["close"])

    print("  Computing composite scores...")
    trend_s, rev_s = compute_scores(
        df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"],
        df_4h["close"], df_4h["high"], df_4h["low"], df_4h["volume"],
        df_15m["close"], df_15m["high"], df_15m["low"],
        close_1m, volume_1m,
        df_1h.index,
    )

    rows = []
    for dt in decision_times:
        trend = int(trend_s.get(dt, 0))
        rev   = int(rev_s.get(dt, 0))
        p_up  = lookup_p_up(trend, rev)
        rows.append({"decision_time": dt, "trend": trend, "rev": rev, "p_up": p_up})

    return pd.DataFrame(rows)


# ── Gate logic ────────────────────────────────────────────────────────────────

def _apply_gates(side: str, p_model: float, p_market: float,
                 p_up: float, offset: float) -> tuple:
    """Returns (passes, blocker, net_edge)."""
    fee = kalshi_fee(p_market)
    raw = (p_model - p_market) if side == "yes" else (p_market - p_model)
    net = raw - fee - SLIPPAGE - SPREAD

    # Gate 0a: model saturation
    if not (P_MODEL_MIN <= p_model <= P_MODEL_MAX):
        return False, "Gate0-sat", net
    # Gate 0b: market liquidity
    if not (P_MARKET_MIN <= p_market <= P_MARKET_MAX):
        return False, "Gate0-liq", net

    if side == "yes":
        # Gate CS: OTM YES (offset > 0) requires bullish composite
        if offset > 0 and p_up < GATE_CS_MIN:
            return False, "GateCS", net
    else:
        # Gate NS: OTM NO (offset < 0) requires bearish composite
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


# ── Sizing + PnL ─────────────────────────────────────────────────────────────

def _kelly_bet(p_model: float, p_market: float, side: str, bankroll: float) -> float:
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
    fee_rate = kalshi_fee(p_market)
    if side == "yes":
        if won:
            n  = bet / p_market
            return bet * (1 - p_market) / p_market - fee_rate * n
        return -bet
    else:
        if won:
            n  = bet / (1 - p_market)
            return bet * p_market / (1 - p_market) - fee_rate * n
        return -bet


# ── Main backtest ─────────────────────────────────────────────────────────────

def main():
    print("Loading real Kalshi scan data...")
    scan = load_scan_data()
    print(f"  {len(scan):,} resolved BTC rows  |  "
          f"{scan['decision_time'].nunique()} unique hours  |  "
          f"{scan['decision_time'].min().date()} → {scan['decision_time'].max().date()}")

    decision_times = scan["decision_time"].dt.tz_convert("UTC").unique()
    comp_idx = build_composite_index(pd.DatetimeIndex(decision_times))
    comp_map = comp_idx.set_index("decision_time")

    bankroll = BANKROLL_0
    hour_records = []
    trade_records = []

    for dt, group in scan.groupby("decision_time"):
        dt_utc = pd.Timestamp(dt).tz_convert("UTC") if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt, tz="UTC")
        comp = comp_map.loc[dt_utc] if dt_utc in comp_map.index else None
        if comp is None:
            continue

        trend = int(comp["trend"])
        rev   = int(comp["rev"])
        p_up  = float(comp["p_up"])

        candidates = []
        for _, row in group.iterrows():
            spot      = float(row["spot"])
            strike    = float(row["strike"])
            p_market  = float(row["p_market"])
            offset    = float(row["offset_pct"])
            vol_eff   = float(row.get("vol_eff", 3e-4)) or 3e-4
            tau       = float(row.get("tau_minutes", 60)) or 60
            sigma_tau = vol_eff * math.sqrt(tau)
            resolved_yes = int(row["resolved_yes"])

            p_model = score_to_p_model(trend, rev, spot, strike, sigma_tau)

            for side in ("yes", "no"):
                passes, blocker, net = _apply_gates(side, p_model, p_market, p_up, offset)
                if passes:
                    won = (resolved_yes == 1) if side == "yes" else (resolved_yes == 0)
                    candidates.append(dict(
                        dt=dt, ticker=row["contract_ticker"],
                        trend=trend, rev=rev, p_up=p_up,
                        offset=offset, strike=strike, spot=spot,
                        side=side, p_model=round(p_model, 4),
                        p_market=p_market, net_edge=net,
                        resolved_yes=resolved_yes, won=won,
                    ))

        # Pick top 4 by net edge
        top = sorted(candidates, key=lambda x: x["net_edge"], reverse=True)[:MAX_BETS_PER_HOUR]

        hour_pnl = 0
        for bet_info in top:
            bet = _kelly_bet(bet_info["p_model"], bet_info["p_market"],
                             bet_info["side"], bankroll)
            pnl = _pnl(bet, bet_info["side"], bet_info["p_market"], bet_info["won"])
            bankroll = max(1.0, bankroll + pnl)
            hour_pnl += pnl
            trade_records.append({**bet_info, "bet": bet, "pnl": round(pnl, 2),
                                   "bankroll": round(bankroll, 2)})

        hour_records.append({"dt": dt, "n_bets": len(top),
                              "hour_pnl": round(hour_pnl, 2),
                              "bankroll": round(bankroll, 2)})

    trades = pd.DataFrame(trade_records)
    hours  = pd.DataFrame(hour_records)

    # ── Summary ────────────────────────────────────────────────────────────────
    n_hours  = len(hours)
    n_trades = len(trades)
    if n_trades == 0:
        print("No trades taken.")
        return

    wins     = trades["won"].sum()
    win_rate = wins / n_trades
    total_pnl = trades["pnl"].sum()
    final_br  = trades["bankroll"].iloc[-1]
    roi       = (final_br - BANKROLL_0) / BANKROLL_0

    print(f"\n{'═'*58}")
    print(f"  REAL KALSHI DATA BACKTEST")
    print(f"  {trades['dt'].min()} → {trades['dt'].max()}")
    print(f"{'═'*58}")
    print(f"  Hours with trades  : {hours[hours['n_bets']>0].shape[0]} / {n_hours}")
    print(f"  Total bets         : {n_trades}  (max {MAX_BETS_PER_HOUR}/hour)")
    print(f"  Win rate           : {wins:.0f}/{n_trades}  ({win_rate:.1%})")
    print(f"  Total PnL          : ${total_pnl:+,.2f}")
    print(f"  Final bankroll     : ${final_br:,.2f}  (started ${BANKROLL_0:,.0f})")
    print(f"  ROI                : {roi:+.1%}")
    print(f"  Avg bet size       : ${trades['bet'].mean():.2f}")
    print(f"  Avg net edge       : {trades['net_edge'].mean():+.1%}")

    print(f"\n  ── By side ──────────────────────────────────────")
    for side in ("yes", "no"):
        s = trades[trades["side"] == side]
        if not len(s): continue
        print(f"  {side.upper():3s}  n={len(s):3d}  win={s['won'].mean():.1%}"
              f"  pnl=${s['pnl'].sum():+,.2f}  avg_edge={s['net_edge'].mean():+.1%}")

    print(f"\n  ── By offset bucket ─────────────────────────────")
    trades["off_bin"] = pd.cut(trades["offset"],
        bins=[-5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 5],
        labels=["<-1%","-1:-0.5%","-0.5:-0.25%","ITM-near",
                "OTM-near","0.25:0.5%","0.5:1%",">1%"])
    gb = trades.groupby("off_bin", observed=True).agg(
        n=("won","count"), wins=("won","sum"), pnl=("pnl","sum"))
    gb["wr"] = gb["wins"]/gb["n"]
    for lbl, r in gb.iterrows():
        if r["n"] == 0: continue
        print(f"  {str(lbl):12s}  n={int(r.n):3d}  win={r.wr:.1%}  pnl=${r.pnl:+,.0f}")

    print(f"\n  ── By composite_p_up bin ────────────────────────")
    trades["p_up_bin"] = pd.cut(trades["p_up"],
        bins=[0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=["≤0.40","0.40-0.45","0.45-0.50","0.50-0.55",
                "0.55-0.60","0.60-0.65","0.65-0.70","≥0.70"])
    gb2 = trades.groupby("p_up_bin", observed=True).agg(
        n=("won","count"), wins=("won","sum"), pnl=("pnl","sum"))
    gb2["wr"] = gb2["wins"]/gb2["n"]
    for lbl, r in gb2.iterrows():
        if r["n"] == 0: continue
        print(f"  p_up {str(lbl):10s}  n={int(r.n):3d}  win={r.wr:.1%}  pnl=${r.pnl:+,.0f}")

    print(f"\n  ── Top (trend, rev) combos (n≥5) ────────────────")
    grp = (trades.groupby(["trend","rev"])
           .agg(n=("won","count"), wr=("won","mean"), pnl=("pnl","sum"))
           .reset_index().query("n >= 5")
           .sort_values("wr", ascending=False).head(10))
    for _, r in grp.iterrows():
        print(f"  trend={int(r.trend):+d}  rev={int(r.rev):+3d}  "
              f"n={int(r.n):3d}  win={r.wr:.1%}  pnl=${r.pnl:+,.0f}")

    out_path = RESULTS_DIR / "backtest_real_kalshi.csv"
    trades.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
