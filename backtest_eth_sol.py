"""
Comprehensive ETH and SOL Kalshi hourly model backtest.

FIXED: p_market is ALWAYS p_yes_market for both YES and NO bets.
Previously, NO bets incorrectly used pm_no as p_market, which caused
all NO gate checks to fail (pm_no was 0.60-0.96, never <= 0.35).

Sweeps gate configurations (p_market thresholds, offset minimums, EMA regime,
side selection) to find the most profitable configuration for each asset.

Outputs:
  - results/backtest_eth_full.csv
  - results/backtest_sol_full.csv
  - Printed tables: §6 raw offset/side, §7 vol regime, §8 EMA regime,
                    §4-5 configs A-K, §9 p_market bucket, §10 EV per dollar
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probability_engine import estimate_probability
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TAU              = 60    # minutes to expiry
EMA_FAST         = 20
EMA_SLOW         = 50
EMA_CONFIRM_BARS = 3
BANKROLL         = 250   # $250 bankroll per spec

OFFSETS = [-0.005, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]

# p_yes_market by offset (ALWAYS use this as p_market for both YES and NO bets)
# OTM (strike above spot, offset > 0): YES is cheap, NO is expensive
OTM_TABLE = {
    0.001: 0.42,
    0.002: 0.30,
    0.003: 0.22,
    0.005: 0.15,
    0.008: 0.08,
    0.010: 0.05,
}
# ITM YES (strike below spot, offset < 0): YES is expensive
ITM_TABLE = {
    0.001: 0.58,
    0.002: 0.65,
    0.003: 0.72,
    0.005: 0.80,
    0.008: 0.87,
}

OTM_KEYS = sorted(OTM_TABLE.keys())
ITM_KEYS = sorted(ITM_TABLE.keys())


def get_pmarket(offset: float) -> float:
    """
    Return p_yes_market for a given signed offset.
    This is ALWAYS used as p_market for both YES and NO bets.
    """
    abs_off = abs(offset)
    if offset < 0:
        bucket = min(ITM_KEYS, key=lambda k: abs(k - abs_off))
        return ITM_TABLE[bucket]
    else:
        bucket = min(OTM_KEYS, key=lambda k: abs(k - abs_off))
        return OTM_TABLE[bucket]


# ── EMA alignment ─────────────────────────────────────────────────────────────

def compute_ema_alignment(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()

    e_f = ema_fast.values
    e_s = ema_slow.values
    cl  = close.values
    n   = len(cl)
    result = ["neutral"] * n

    for i in range(n):
        if i < EMA_CONFIRM_BARS - 1:
            result[i] = "neutral"
            continue
        sl = slice(i - EMA_CONFIRM_BARS + 1, i + 1)
        lc  = cl[sl]
        l_f = e_f[sl]
        l_s = e_s[sl]
        golden_cross = bool(np.all(l_f > l_s))
        death_cross  = bool(np.all(l_f < l_s))
        price_above  = bool(np.all(lc > l_f))
        price_below  = bool(np.all(lc < l_s))
        if golden_cross and price_above:
            result[i] = "bullish"
        elif death_cross or price_below:
            result[i] = "bearish"
        else:
            result[i] = "neutral"

    return pd.Series(result, index=close.index)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_asset_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load latest 1h and 1m parquet files for the given symbol."""
    def latest_parquet(pattern):
        matches = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        matches = [m for m in matches if ".ckpt." not in m.name]
        return matches[-1] if matches else None

    p_1h = latest_parquet(f"*{symbol}_1h_2024-01-01*.parquet")
    p_1m = latest_parquet(f"*{symbol}_1m_2024-01-01*.parquet")
    if not p_1h or not p_1m:
        raise FileNotFoundError(f"Missing parquet for {symbol}")

    print(f"  1h: {p_1h.name}")
    print(f"  1m: {p_1m.name}")

    df_1h = pd.read_parquet(p_1h)
    df_1m = pd.read_parquet(p_1m)

    for df in [df_1h, df_1m]:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.columns = df.columns.str.lower()

    return df_1h, df_1m


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_asset(asset: str, symbol: str) -> pd.DataFrame:
    """
    Build the full simulation DataFrame for one asset.
    Each row = (bar, offset, side) triple.

    KEY FIX: p_market = p_yes_market for BOTH yes and no sides.
    """
    print(f"\n{'='*68}")
    print(f"  Loading {asset} ({symbol})...")
    print(f"{'='*68}")

    df_1h, df_1m = load_asset_data(symbol)

    # EMA alignment on full 1h series
    print("  Computing EMA alignment...")
    ema_series = compute_ema_alignment(df_1h["close"])
    ema_map    = ema_series.to_dict()

    # vol_60m: rolling 60-bar std of 1m log returns, sampled at each 1h bar
    print("  Computing vol_60m...")
    log_ret_1m = np.log(df_1m["close"] / df_1m["close"].shift(1)).dropna()
    vol_1m     = log_ret_1m.rolling(60).std()
    vol_1h     = vol_1m.resample("1h", closed="right", label="right").last()
    vol_map    = vol_1h.to_dict()

    bars = df_1h.reset_index()
    n    = len(bars) - 1  # last bar has no next close

    print(f"  Bars: {n:,}  ({bars.iloc[200]['open_time'].date()} → {bars.iloc[n-1]['open_time'].date()})")
    print(f"  Simulating {n-200:,} bars × {len(OFFSETS)} offsets × 2 sides...")

    records = []
    for i in range(200, n):
        row      = bars.iloc[i]
        next_row = bars.iloc[i + 1]
        ts       = row["open_time"]

        spot       = float(row["close"])
        next_close = float(next_row["close"])

        ema  = ema_map.get(ts, "neutral")
        vol  = vol_map.get(ts, None)
        if vol is None or np.isnan(vol) or vol <= 0:
            continue

        for offset in OFFSETS:
            strike  = spot * (1 + offset)
            outcome = int(next_close > strike)

            try:
                prob    = estimate_probability(spot, strike, TAU, vol)
                p_model = max(0.001, min(0.999, prob.p_yes))
            except Exception:
                continue

            # p_market is ALWAYS the YES price regardless of side
            p_market = get_pmarket(offset)

            for side in ["yes", "no"]:
                fee = kalshi_fee(p_market)

                # Edge calculation (using p_yes_market as p_market throughout)
                if side == "yes":
                    raw_edge = p_model - p_market
                else:
                    # NO edge: model says NO wins with prob (1-p_model),
                    # market implies NO wins with prob (1-p_market)
                    raw_edge = (1 - p_model) - (1 - p_market)  # = p_market - p_model

                net_edge = raw_edge - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD
                win      = outcome if side == "yes" else (1 - outcome)

                # half-Kelly bet sizing (cap 5%)
                # Kelly formula: f* = (b*p - q) / b * 0.5
                if side == "yes":
                    p_win = p_model
                    b     = (1 - p_market) / p_market if p_market < 1 else 1
                else:
                    # NO: p_win = prob NO wins = 1 - p_model
                    # payout ratio: win $p_market per $1 risked on NO bet
                    p_win = 1 - p_model
                    b     = p_market / (1 - p_market) if p_market < 1 else 1
                q_win  = 1 - p_win
                kelly  = max(0.0, (b * p_win - q_win) / b) * 0.5
                kelly  = min(kelly, 0.05)
                bet    = kelly * BANKROLL

                # PnL for this trade
                # p_market is p_yes_market throughout
                if side == "yes":
                    if outcome == 1:
                        pnl = bet * (1 - p_market) / p_market - fee * bet
                    else:
                        pnl = -bet - fee * bet
                else:
                    # NO bet: win if outcome == 0 (price did NOT exceed strike)
                    if outcome == 0:
                        pnl = bet * p_market / (1 - p_market) - fee * bet
                    else:
                        pnl = -bet - fee * bet

                # Flat $5 pnl (for raw analysis, no gates)
                flat_bet = 5.0
                if side == "yes":
                    if outcome == 1:
                        flat_pnl = flat_bet * (1 - p_market) / p_market - fee * flat_bet
                    else:
                        flat_pnl = -flat_bet - fee * flat_bet
                else:
                    if outcome == 0:
                        flat_pnl = flat_bet * p_market / (1 - p_market) - fee * flat_bet
                    else:
                        flat_pnl = -flat_bet - fee * flat_bet

                records.append({
                    "ts":          ts,
                    "asset":       asset,
                    "spot":        spot,
                    "offset":      offset,
                    "strike":      strike,
                    "next_close":  next_close,
                    "ema":         ema,
                    "vol_60m":     vol,
                    "p_model":     p_model,
                    "p_market":    p_market,   # always p_yes_market
                    "side":        side,
                    "raw_edge":    raw_edge,
                    "net_edge":    net_edge,
                    "outcome":     outcome,
                    "win":         win,
                    "bet":         bet,
                    "pnl":         pnl,
                    "flat_pnl":    flat_pnl,
                    "kelly":       kelly,
                })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} rows")
    return df


# ── Separator ─────────────────────────────────────────────────────────────────

def sep(title: str = "", width: int = 68):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n── {title} {'─'*pad}")
    else:
        print("─" * width)


# ── Stats helper ──────────────────────────────────────────────────────────────

def compute_stats(trades: pd.DataFrame, date_range_years: float) -> dict:
    if trades.empty:
        return {"n": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0,
                "pnl_per_year": 0, "sharpe": 0, "max_dd": 0,
                "pct_yes": 0, "pct_no": 0}

    pnl   = trades["pnl"].values
    n     = len(pnl)
    total = pnl.sum()
    avg   = pnl.mean()
    std   = pnl.std() if n > 1 else 1e-9
    sharpe = avg / std * math.sqrt(8760) if std > 0 else 0

    # Max drawdown
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    max_dd = dd.min() if len(dd) > 0 else 0

    pct_yes = (trades["side"] == "yes").mean() * 100
    pct_no  = 100 - pct_yes

    return {
        "n":           n,
        "win_rate":    trades["win"].mean(),
        "total_pnl":   total,
        "avg_pnl":     avg,
        "pnl_per_year": total / date_range_years,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "pct_yes":     pct_yes,
        "pct_no":      pct_no,
    }


def print_stats(name: str, stats: dict):
    if stats["n"] == 0:
        print(f"  {name:<28}  n=0  (no trades)")
        return
    print(f"  {name:<28}  "
          f"n={stats['n']:>5}  "
          f"win={stats['win_rate']:>5.1%}  "
          f"tot=${stats['total_pnl']:>+8.2f}  "
          f"avg=${stats['avg_pnl']:>+6.3f}  "
          f"$/yr=${stats['pnl_per_year']:>+8.2f}  "
          f"sharpe={stats['sharpe']:>5.2f}  "
          f"maxDD=${stats['max_dd']:>+7.2f}  "
          f"Y%={stats['pct_yes']:>5.1f}%")


# ── Gate configs ──────────────────────────────────────────────────────────────

def apply_config(df: pd.DataFrame, config: str) -> pd.DataFrame:
    """
    Apply gate config and return filtered trades.

    IMPORTANT: df["p_market"] is ALWAYS p_yes_market.
    - For NO bets: p_market <= 0.35 means "YES costs <= 35c" → deep OTM NO
    - For YES bets: p_market >= 0.55 means "YES costs >= 55c" → ITM YES
    """
    base_edge = df["net_edge"] >= 0.03

    if config == "A":
        # NO strict: OTM NO, p_market (YES price) <= 0.35, net_edge >= 0.03
        # p_market <= 0.35 means offset >= +0.2% (YES is cheap -> NO is deep OTM from YES side)
        mask = (
            (df["side"] == "no") &
            (df["offset"] > 0) &
            (df["p_market"] <= 0.35) &
            base_edge
        )

    elif config == "B":
        # NO moderate: OTM NO, p_market <= 0.40, net_edge >= 0.03
        mask = (
            (df["side"] == "no") &
            (df["offset"] > 0) &
            (df["p_market"] <= 0.40) &
            base_edge
        )

    elif config == "C":
        # YES strict: ITM YES (offset < 0) OR high-confidence OTM (p_market >= 0.55)
        itm_yes = (df["side"] == "yes") & (df["offset"] < 0)
        otm_yes = (df["side"] == "yes") & (df["offset"] > 0) & (df["p_market"] >= 0.55)
        mask = (itm_yes | otm_yes) & base_edge

    elif config == "D":
        # YES moderate: p_market >= 0.40, net_edge >= 0.03
        mask = (
            (df["side"] == "yes") &
            (df["p_market"] >= 0.40) &
            base_edge
        )

    elif config == "E":
        # Mixed, EMA-filtered
        # YES: p_market >= 0.50, EMA != bearish
        # NO: offset > 0, p_market <= 0.35, EMA != bullish
        yes_mask = (df["side"] == "yes") & (df["p_market"] >= 0.50) & (df["ema"] != "bearish")
        no_mask  = (df["side"] == "no")  & (df["offset"] > 0) & (df["p_market"] <= 0.35) & (df["ema"] != "bullish")
        mask = (yes_mask | no_mask) & base_edge

    elif config == "F":
        # YES + EMA: allow bullish + neutral, p_market >= 0.40
        mask = (
            (df["side"] == "yes") &
            (df["ema"].isin(["bullish", "neutral"])) &
            (df["p_market"] >= 0.40) &
            base_edge
        )

    elif config == "G":
        # NO + EMA: bearish/neutral only, offset >= 0.2%, p_market <= 0.35
        mask = (
            (df["side"] == "no") &
            (df["ema"].isin(["bearish", "neutral"])) &
            (df["offset"] >= 0.002) &
            (df["p_market"] <= 0.35) &
            base_edge
        )

    elif config == "H":
        # Mixed, strict EMA alignment
        # YES: EMA == bullish, p_market >= 0.40
        # NO: EMA == bearish, offset >= 0.2%, p_market <= 0.35
        yes_mask = (df["side"] == "yes") & (df["ema"] == "bullish") & (df["p_market"] >= 0.40)
        no_mask  = (df["side"] == "no")  & (df["ema"] == "bearish") & (df["offset"] >= 0.002) & (df["p_market"] <= 0.35)
        mask = (yes_mask | no_mask) & base_edge

    elif config == "I":
        # NO + min offset 0.20%: offset >= 0.002, p_market <= 0.35
        mask = (
            (df["side"] == "no") &
            (df["offset"] >= 0.002) &
            (df["p_market"] <= 0.35) &
            base_edge
        )

    elif config == "J":
        # NO + min offset 0.30%: offset >= 0.003, p_market <= 0.35
        mask = (
            (df["side"] == "no") &
            (df["offset"] >= 0.003) &
            (df["p_market"] <= 0.35) &
            base_edge
        )

    elif config == "K":
        # Best mixed, data-driven
        # YES: p_market >= 0.55 (high-confidence YES only)
        # NO: p_market <= 0.22 (= +0.3%+ OTM), offset >= 0.003
        yes_mask = (df["side"] == "yes") & (df["p_market"] >= 0.55)
        no_mask  = (df["side"] == "no")  & (df["p_market"] <= 0.22) & (df["offset"] >= 0.003)
        mask = (yes_mask | no_mask) & base_edge

    else:
        raise ValueError(f"Unknown config: {config}")

    return df[mask].copy()


# ── Analysis sections ─────────────────────────────────────────────────────────

def section_raw_offset_profitability(df: pd.DataFrame, asset: str):
    sep(f"§6  {asset} RAW OFFSET/SIDE PROFITABILITY (flat $5 bet, no gates)")
    print(f"  {'offset':>7}  {'side':<4}  {'n':>6}  {'win%':>6}  {'avg_net_edge':>13}  "
          f"{'total_pnl($5)':>14}  {'breakeven_win%':>14}")
    print("  " + "-" * 78)

    for offset in sorted(df["offset"].unique()):
        for side in ["yes", "no"]:
            sub = df[(df["offset"] == offset) & (df["side"] == side)]
            if sub.empty:
                continue
            n         = len(sub)
            win_rate  = sub["win"].mean()
            avg_edge  = sub["net_edge"].mean()
            total_pnl = sub["flat_pnl"].sum()
            pm        = sub["p_market"].iloc[0]   # p_yes_market
            # Breakeven win rate
            # YES: need win_rate >= p_market to break even (cost is p_market per $1)
            # NO:  need win_rate >= (1-p_market) (cost is 1-p_market per $1)
            beven = pm if side == "yes" else (1 - pm)
            print(f"  {offset:>+6.1%}  {side:<4}  {n:>6}  {win_rate:>5.1%}  "
                  f"{avg_edge:>+12.4f}  {total_pnl:>+13.2f}  {beven:>13.1%}")


def section_vol_regime(df: pd.DataFrame, asset: str):
    sep(f"§7  {asset} VOLATILITY REGIME ANALYSIS")
    df2 = df.copy()
    df2["vol_q"] = pd.qcut(df2["vol_60m"], q=4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])

    tbl = df2.groupby(["vol_q", "side"], observed=True).agg(
        n=("win", "count"),
        win_rate=("win", "mean"),
        avg_pnl=("flat_pnl", "mean"),
    ).reset_index()

    print(f"  {'vol_q':<10}  {'side':<4}  {'n':>6}  {'win%':>6}  {'avg_pnl($5)':>12}")
    print("  " + "-" * 46)
    for _, row in tbl.iterrows():
        print(f"  {str(row['vol_q']):<10}  {row['side']:<4}  {int(row['n']):>6}  "
              f"{row['win_rate']:>5.1%}  {row['avg_pnl']:>+11.4f}")


def section_ema_regime(df: pd.DataFrame, asset: str):
    sep(f"§8  {asset} EMA REGIME TABLE")
    tbl = df.groupby(["ema", "side"]).agg(
        n=("win", "count"),
        win_rate=("win", "mean"),
        avg_pnl=("flat_pnl", "mean"),
    ).reset_index()

    print(f"  {'ema':<10}  {'side':<4}  {'n':>6}  {'win%':>6}  {'avg_pnl($5)':>12}")
    print("  " + "-" * 46)
    for _, row in tbl.iterrows():
        print(f"  {row['ema']:<10}  {row['side']:<4}  {int(row['n']):>6}  "
              f"{row['win_rate']:>5.1%}  {row['avg_pnl']:>+11.4f}")


def section_pmarket_bucket(df: pd.DataFrame, asset: str):
    """
    §9 - Profitability by p_market bucket (raw, no edge gate)
    Flat $5 bet. Reports n, win%, actual EV per $5, breakeven win%, profitable?
    p_market is always p_yes_market.
    """
    sep(f"§9  {asset} PROFITABILITY BY p_market BUCKET (flat $5, no gates)")
    print(f"  {'side':<4}  {'p_mkt_bucket':<18}  {'n':>6}  {'win%':>6}  "
          f"{'EV/bet($5)':>11}  {'breakeven%':>11}  {'profitable':>10}")
    print("  " + "-" * 74)

    # YES buckets: [0.40-0.50, 0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90]
    yes_buckets = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90)]
    # NO buckets (p_market = YES price, so NO is OTM when YES is cheap):
    # [0.05-0.10, 0.10-0.15, 0.15-0.22, 0.22-0.30, 0.30-0.35, 0.35-0.42]
    no_buckets  = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.22), (0.22, 0.30),
                   (0.30, 0.35), (0.35, 0.42)]

    for side, buckets in [("yes", yes_buckets), ("no", no_buckets)]:
        sub_side = df[df["side"] == side]
        for lo, hi in buckets:
            sub = sub_side[(sub_side["p_market"] >= lo) & (sub_side["p_market"] < hi)]
            if sub.empty:
                continue
            n        = len(sub)
            win_rate = sub["win"].mean()
            ev       = sub["flat_pnl"].mean()
            pm_mid   = (lo + hi) / 2
            beven    = pm_mid if side == "yes" else (1 - pm_mid)
            prof     = "YES" if ev > 0 else "NO"
            label    = f"{lo:.2f}-{hi:.2f}"
            print(f"  {side:<4}  {label:<18}  {n:>6}  {win_rate:>5.1%}  "
                  f"{ev:>+10.4f}  {beven:>10.1%}  {prof:>10}")


def section_ev_per_dollar(df: pd.DataFrame, asset: str):
    """
    §10 - EV per dollar bet by offset and side.
    EV/$ = EV per $1 invested.
    YES: EV = win_rate*(1-p_market)/p_market - (1-win_rate) - fee
    NO:  EV = win_rate*p_market/(1-p_market) - (1-win_rate) - fee
    """
    sep(f"§10  {asset} EV PER DOLLAR BET BY OFFSET (no gates)")
    print(f"  {'offset':>7}  {'side':<4}  {'n':>6}  {'win%':>6}  {'p_mkt':>6}  "
          f"{'EV/$':>8}  {'EV_rank':>7}")
    print("  " + "-" * 58)

    rows = []
    for offset in sorted(df["offset"].unique()):
        for side in ["yes", "no"]:
            sub = df[(df["offset"] == offset) & (df["side"] == side)]
            if sub.empty:
                continue
            n        = len(sub)
            win_rate = sub["win"].mean()
            pm       = sub["p_market"].iloc[0]
            fee      = kalshi_fee(pm)

            if side == "yes":
                ev_per_dollar = win_rate * (1 - pm) / pm - (1 - win_rate) - fee
            else:
                # win_rate here is rate of NO wins = 1 - outcome_mean
                ev_per_dollar = win_rate * pm / (1 - pm) - (1 - win_rate) - fee

            rows.append((offset, side, n, win_rate, pm, ev_per_dollar))

    # Rank by EV per dollar
    rows_sorted = sorted(rows, key=lambda x: x[5], reverse=True)
    for rank, (offset, side, n, win_rate, pm, ev_d) in enumerate(rows_sorted, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {offset:>+6.1%}  {side:<4}  {n:>6}  {win_rate:>5.1%}  "
              f"{pm:>5.2f}  {ev_d:>+7.4f}  #{rank:<4}{marker}")


def section_config_sweep(df: pd.DataFrame, asset: str, date_range_years: float):
    sep(f"§4-5  {asset} GATE CONFIG SWEEP (A-K)")
    configs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    config_labels = {
        "A": "NO strict (p_mkt<=0.35, OTM)",
        "B": "NO moderate (p_mkt<=0.40, OTM)",
        "C": "YES strict (ITM/p_mkt>=0.55)",
        "D": "YES moderate (p_mkt>=0.40)",
        "E": "Mixed EMA-filtered",
        "F": "YES+EMA (bullish/neutral)",
        "G": "NO+EMA (bear/neut,off>=0.2%)",
        "H": "Mixed+EMA strict",
        "I": "NO+off>=0.2%,p_mkt<=0.35",
        "J": "NO+off>=0.3%,p_mkt<=0.35",
        "K": "Best mixed (YES>=0.55,NO<=0.22)",
    }

    header = (f"  {'Config':<4}  {'Description':<32}  "
              f"{'n':>5}  {'win%':>5}  {'$/trade':>7}  {'$/yr':>8}  "
              f"{'sharpe':>6}  {'maxDD':>7}  {'Y%':>5}")
    print(header)
    print("  " + "-" * 96)

    results = {}
    for cfg_name in configs:
        filtered = apply_config(df, cfg_name)
        stats    = compute_stats(filtered, date_range_years)
        results[cfg_name] = (stats, config_labels[cfg_name])

        if stats["n"] == 0:
            print(f"  {cfg_name:<4}  {config_labels[cfg_name]:<32}  n=0")
            continue

        print(f"  {cfg_name:<4}  {config_labels[cfg_name]:<32}  "
              f"{stats['n']:>5}  {stats['win_rate']:>4.1%}  "
              f"{stats['avg_pnl']:>+6.3f}  {stats['pnl_per_year']:>+8.2f}  "
              f"{stats['sharpe']:>5.2f}  {stats['max_dd']:>+6.2f}  "
              f"{stats['pct_yes']:>4.1f}%")

    # Breakdown by offset for best config (by $/yr)
    ranked = sorted(
        [(cfg, s["pnl_per_year"]) for cfg, (s, _) in results.items() if s["n"] > 0],
        key=lambda x: x[1], reverse=True
    )

    if ranked:
        best_cfg = ranked[0][0]
        sep(f"  {asset} Config {best_cfg} — OFFSET BREAKDOWN")
        best_trades = apply_config(df, best_cfg)
        tbl = best_trades.groupby(["offset", "side"]).agg(
            n=("win", "count"),
            win_rate=("win", "mean"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        ).reset_index()
        print(f"  {'offset':>7}  {'side':<4}  {'n':>5}  {'win%':>5}  {'total_pnl':>10}  {'avg_pnl':>8}")
        print("  " + "-" * 52)
        for _, row in tbl.iterrows():
            print(f"  {row['offset']:>+6.1%}  {row['side']:<4}  {int(row['n']):>5}  "
                  f"{row['win_rate']:>4.1%}  {row['total_pnl']:>+9.2f}  {row['avg_pnl']:>+7.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_asset(asset: str, symbol: str):
    df = simulate_asset(asset, symbol)

    # Date range in years
    dates = pd.to_datetime(df["ts"])
    date_range_years = (dates.max() - dates.min()).days / 365.25
    print(f"  Date range: {date_range_years:.2f} years")

    # Save full CSV
    out = RESULTS_DIR / f"backtest_{asset.lower()}_full.csv"
    df.to_csv(out, index=False)
    print(f"  Full data saved → {out}")

    sep(f"{'='*20} {asset} ANALYSIS {'='*20}")

    section_raw_offset_profitability(df, asset)
    section_vol_regime(df, asset)
    section_ema_regime(df, asset)
    section_pmarket_bucket(df, asset)
    section_ev_per_dollar(df, asset)
    results = section_config_sweep(df, asset, date_range_years)

    return df, results, date_range_years


def main():
    print("\n" + "=" * 68)
    print("  ETH + SOL COMPREHENSIVE KALSHI HOURLY MODEL BACKTEST")
    print("  FIXED: p_market = p_yes_market for ALL bets (YES and NO)")
    print("=" * 68)
    print(f"  Bankroll: ${BANKROLL}  |  tau={TAU}min  |  half-Kelly capped 5%")
    print(f"  Offsets: {[f'{o:+.1%}' for o in OFFSETS]}")
    print(f"  Configs: A-K  (see spec)")
    print(f"  Analyses: §6 raw offset, §7 vol, §8 EMA, §9 p_mkt bucket, §10 EV/$")

    # ETH
    eth_df, eth_results, eth_years = run_asset("ETH", "ETHUSDT")

    # SOL
    sol_df, sol_results, sol_years = run_asset("SOL", "SOLUSDT")

    # ── Final recommendations ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  FINAL RECOMMENDATIONS")
    print("=" * 68)

    config_labels = {
        "A": "NO strict (OTM NO, p_mkt<=0.35, net_edge>=0.03)",
        "B": "NO moderate (OTM NO, p_mkt<=0.40, net_edge>=0.03)",
        "C": "YES strict (ITM or p_mkt>=0.55, net_edge>=0.03)",
        "D": "YES moderate (p_mkt>=0.40, net_edge>=0.03)",
        "E": "Mixed EMA-filtered (YES>=0.50 non-bear, NO<=0.35 non-bull)",
        "F": "YES+EMA (bullish/neutral, p_mkt>=0.40)",
        "G": "NO+EMA (bearish/neutral, off>=0.2%, p_mkt<=0.35)",
        "H": "Mixed+EMA aligned (bullish YES, bearish NO)",
        "I": "NO+off>=0.2%, p_mkt<=0.35, net_edge>=0.03",
        "J": "NO+off>=0.3%, p_mkt<=0.35, net_edge>=0.03",
        "K": "Best mixed (YES p_mkt>=0.55, NO p_mkt<=0.22+off>=0.3%)",
    }

    scale_factor = 10_000 / BANKROLL  # for $10k bankroll projection

    for asset, results, years in [("ETH", eth_results, eth_years), ("SOL", sol_results, sol_years)]:
        ranked = sorted(
            [(cfg, s["pnl_per_year"], s) for cfg, (s, _) in results.items() if s["n"] >= 10],
            key=lambda x: x[1], reverse=True
        )
        print(f"\n{'─'*68}")
        if not ranked:
            print(f"  BEST {asset} CONFIG: insufficient trades in all configs")
            continue

        best_cfg, best_pyr, best_stats = ranked[0]
        projected_10k = best_pyr * scale_factor

        print(f"  BEST {asset} CONFIG: {best_cfg}")
        print(f"  Gates: {config_labels[best_cfg]}")
        print(f"  n={best_stats['n']}  win={best_stats['win_rate']:.1%}  "
              f"sharpe={best_stats['sharpe']:.2f}  maxDD=${best_stats['max_dd']:.2f}")
        print(f"  Expected annual PnL on ${BANKROLL:,} bankroll: ${best_pyr:+.2f}/year")
        print(f"  Expected annual PnL on $10,000 bankroll:   ${projected_10k:+,.2f}/year")

        if len(ranked) > 1:
            print(f"  Runner-up configs:")
            for cfg, pyr, stats in ranked[1:5]:
                proj10k = pyr * scale_factor
                print(f"    {cfg}: {config_labels[cfg][:50]}")
                print(f"       $/yr=${pyr:+.2f} (${proj10k:+,.2f} on $10k)  "
                      f"sharpe={stats['sharpe']:.2f}  n={stats['n']}")

    print("\n  Results saved to:")
    print(f"    {RESULTS_DIR}/backtest_eth_full.csv")
    print(f"    {RESULTS_DIR}/backtest_sol_full.csv")
    print()


if __name__ == "__main__":
    main()
