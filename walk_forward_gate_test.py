#!/usr/bin/env python3
"""
walk_forward_gate_test.py

Hypothesis-driven walk-forward validation of 8 directional gate signals
across BTC / ETH / SOL 15m bars, Jan 2024 → May 2026.

Signals tested (all pure OHLCV — no Kalshi market data required):
  1. ema_bias_15m    EMA20 direction vs spot on 15m bars
  2. stoch_k_15m     14-period stochastic K (15m)
  3. composite_p_up  Simplified directional composite: EMA/RSI/VWAP/momentum
  4. vol_ratio       30m realized vol / rolling 2-week median (elevated = risky)
  5. stoch_k_1h      14-period stochastic K (1h)
  6. consec_dir_1h   Consecutive same-direction 1h closes
  7. ema_bias_1h     EMA20 direction vs spot on 1h bars
  8. body_dir_15m    Candle body direction on last completed 15m bar

Walk-forward folds (expanding train window, non-overlapping test):
  F1: Train 2024H1  → Test 2024H2
  F2: Train 2024    → Test 2025H1
  F3: Train 2025H1  → Test 2025H2   (note: shrinking window intentional — tests recent regime)
  F4: Train 2024+2025 → Test 2026 YTD

Gate thresholds are FIXED by economic logic (not selected from training data):
  - stoch midpoint: 50
  - composite_p_up neutral: 0.50
  - vol_ratio elevated: 1.50
  - stoch overbought/oversold: 70 / 30
  - consec_dir neutral: 0
  - ema_bias: categorical -1 / +1

For each gate, the test is: does blocking the predicted-loser condition
improve out-of-sample test-fold PnL consistently across folds?

PnL simulation:
  - Flat $50 stake per trade (non-compounding)
  - p_market computed from rolling 30m realized vol via normal distribution
  - Offset = 0.10% (typical Kalshi 15m near-ATM strike distance)
  - Win payout = stake * (1 - p_market) / p_market
  - Loss = -stake
  - KALSHI_RAKE = 0.07 applied to winnings

Output: per-gate summary table + per-fold detail + consistency score
"""

import warnings
import glob
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

warnings.filterwarnings("ignore")

BASE       = Path(__file__).parent
DATA       = BASE / "data"
RESULTS    = BASE / "results"

STAKE      = 50.0
KALSHI_RAKE= 0.07
OFFSET     = 0.001      # 0.10% strike offset from spot
MIN_BARS   = 200        # minimum bars for a fold to be reported

SEP  = "=" * 76
SEP2 = "-" * 76

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Walk-forward fold definitions (train_start, train_end, test_start, test_end)
FOLDS = [
    ("F1", "2024-01-01", "2024-07-01", "2024-07-01", "2025-01-01"),
    ("F2", "2024-01-01", "2025-01-01", "2025-01-01", "2025-07-01"),
    ("F3", "2025-01-01", "2025-07-01", "2025-07-01", "2026-01-01"),
    ("F4", "2024-01-01", "2026-01-01", "2026-01-01", "2026-06-01"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def latest_parquet(sym: str, tf: str) -> Path:
    full = sorted(DATA.glob(f"binanceus_{sym}_{tf}_2024-01-01_*.parquet"))
    if full:
        return full[-1]
    files = sorted(DATA.glob(f"binanceus_{sym}_{tf}_*.parquet"), key=lambda p: p.stat().st_size)
    if not files:
        raise FileNotFoundError(f"No {sym} {tf} parquet")
    return files[-1]


def load_1m(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(latest_parquet(sym, "1m"))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)


# ── Signal computation ────────────────────────────────────────────────────────

def stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    rng = (hi - lo).replace(0, np.nan)
    return ((close - lo) / rng * 100).clip(0, 100)


def build_15m_signals(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1m → 15m and compute all 8 gate signals.
    All signals are shifted by 1 bar so they represent what is KNOWN
    at bar open (no look-ahead within the bar).
    Returns one row per 15m bar.
    """
    df = df1m.resample("15min", origin="start_day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    df = df.astype(float)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── 1. ema_bias_15m — EMA20 direction vs close ──
    ema20_15m = c.ewm(span=20, adjust=False).mean()
    df["ema_bias_15m"] = np.where(ema20_15m > c, 1.0, -1.0)   # +1 = price above EMA (bullish)

    # ── 2. stoch_k_15m ──
    df["stoch_k_15m"] = stoch_k(c, h, l, period=14)

    # ── 3. composite_p_up — simplified OHLCV-only directional composite ──
    # Components (each contributes ±1):
    #   a. EMA20 vs EMA50 direction
    #   b. EMA50 vs close (is price above medium-term trend?)
    #   c. RSI-14 above 50
    #   d. Last bar momentum (close > close.shift(1))
    #   e. VWAP position (close vs session VWAP approximated as daily EMA of typical price)
    ema50_15m = c.ewm(span=50, adjust=False).mean()
    ema20_gt_ema50 = (ema20_15m > ema50_15m).astype(float) * 2 - 1   # -1 or +1
    price_gt_ema50 = (c > ema50_15m).astype(float) * 2 - 1

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
    rsi_gt_50 = (rsi > 50).astype(float) * 2 - 1

    mom = (c > c.shift(1)).astype(float) * 2 - 1

    typical = (h + l + c) / 3
    # Approximate session VWAP: cumulative over daily window; use 96-bar EMA as proxy
    day_ema_tp = typical.ewm(span=96, adjust=False).mean()
    price_gt_vwap = (c > day_ema_tp).astype(float) * 2 - 1

    composite_raw = ema20_gt_ema50 + price_gt_ema50 + rsi_gt_50 + mom + price_gt_vwap
    # Map from [-5, +5] to [0, 1]
    df["composite_p_up"] = (composite_raw + 5) / 10.0

    # ── 4. vol_ratio — realized 30m vol / rolling 2-week median ──
    log_ret = np.log(c / c.shift(1))
    vol_30m = log_ret.rolling(30).std() * math.sqrt(1440 * 365)   # annualized
    rolling_med = vol_30m.rolling(2 * 96, min_periods=48).median()  # 2 weeks of 15m bars
    df["vol_ratio"] = (vol_30m / rolling_med.replace(0, np.nan)).clip(0, 10)

    # Store vol_30m for p_market computation later
    df["vol_30m_abs"] = log_ret.rolling(30).std()   # per-bar (not annualized)

    # ── 5. body_dir_15m — candle body direction ──
    df["body_dir_15m"] = np.where(c > df["open"], 1.0, np.where(c < df["open"], -1.0, 0.0))

    # Shift all signals by 1 (signals known at bar OPEN, not close)
    sig_cols = ["ema_bias_15m", "stoch_k_15m", "composite_p_up",
                "vol_ratio", "vol_30m_abs", "body_dir_15m"]
    for col in sig_cols:
        df[col] = df[col].shift(1)

    return df


def build_1h_signals(df1m: pd.DataFrame, df15m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1h-bar signals and forward-fill to 15m index.
    Returns a DataFrame aligned to df15m.index.
    """
    df1h = df1m.resample("1h", origin="start_day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna(subset=["close"]).astype(float)

    c1h = df1h["close"]
    h1h = df1h["high"]
    l1h = df1h["low"]

    # ── 5. stoch_k_1h ──
    df1h["stoch_k_1h"] = stoch_k(c1h, h1h, l1h, period=14)

    # ── 6. consec_dir_1h — consecutive same-direction closes ──
    dir1h = np.sign(c1h.diff()).fillna(0)
    consec = pd.Series(0, index=df1h.index, dtype=float)
    for i in range(1, len(dir1h)):
        d = dir1h.iloc[i]
        if d == 0:
            consec.iloc[i] = 0
        elif d == consec.iloc[i - 1] / max(abs(consec.iloc[i - 1]), 1) if consec.iloc[i - 1] != 0 else True:
            consec.iloc[i] = consec.iloc[i - 1] + d
        else:
            consec.iloc[i] = d
    df1h["consec_dir_1h"] = consec

    # ── 7. ema_bias_1h ──
    ema20_1h = c1h.ewm(span=20, adjust=False).mean()
    df1h["ema_bias_1h"] = (np.where(c1h > ema20_1h, 1.0, -1.0))

    # Shift by 1 bar (no look-ahead)
    for col in ["stoch_k_1h", "consec_dir_1h", "ema_bias_1h"]:
        df1h[col] = df1h[col].shift(1)

    # Forward-fill to 15m index
    out = df1h[["stoch_k_1h", "consec_dir_1h", "ema_bias_1h"]].reindex(
        df15m.index, method="ffill"
    )
    return out


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trades(df15m: pd.DataFrame, sig_1h: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    For each 15m bar, simulate one YES and one NO trade opportunity.

    YES: bet price closes above strike = spot * (1 - OFFSET)  (slightly below spot)
    NO:  bet price closes below strike = spot * (1 + OFFSET)  (slightly above spot)

    p_market is computed from the vol model (normal distribution):
      sigma_tau = vol_30m_abs * sqrt(15/1)   [15 1m-bars in a 15m bar]
      p_YES = Phi( log(spot/strike) / sigma_tau )
      p_NO  = 1 - Phi( log(spot/strike_no) / sigma_tau )

    Win condition (determined from next bar's close):
      YES wins: next_close > strike_yes
      NO wins:  next_close < strike_no

    PnL:
      win:  STAKE * (1 - p_market) / p_market * (1 - KALSHI_RAKE)
      lose: -STAKE
    """
    df = df15m.copy()
    df = df.join(sig_1h, how="left")

    # Next close (the outcome)
    df["next_close"] = df["close"].shift(-1)
    df = df.dropna(subset=["next_close", "vol_30m_abs"])

    spot          = df["close"].values
    next_close    = df["next_close"].values
    sigma_abs     = df["vol_30m_abs"].values   # per 1m bar
    sigma_15m     = np.sqrt(15) * sigma_abs    # 15m realized sigma (log scale)
    sigma_15m     = np.where(sigma_15m < 1e-6, 1e-6, sigma_15m)

    # Strike levels
    strike_yes = spot * (1 - OFFSET)   # 0.1% below spot
    strike_no  = spot * (1 + OFFSET)   # 0.1% above spot

    # p_market from vol model
    z_yes = np.log(spot / strike_yes) / sigma_15m      # positive for YES (ITM)
    z_no  = np.log(strike_no / spot)  / sigma_15m      # same magnitude

    p_yes_mkt = sp_norm.cdf(z_yes).clip(0.05, 0.95)
    p_no_mkt  = sp_norm.cdf(z_no).clip(0.05, 0.95)     # = P(price < strike_no)

    # Outcomes
    yes_win = (next_close > strike_yes).astype(int)     # 1 = YES wins
    no_win  = (next_close < strike_no).astype(int)      # 1 = NO wins

    # PnL (flat $50 stake)
    yes_pnl = np.where(
        yes_win,
        STAKE * (1 - p_yes_mkt) / p_yes_mkt * (1 - KALSHI_RAKE),
        -STAKE,
    )
    no_pnl = np.where(
        no_win,
        STAKE * (1 - p_no_mkt) / p_no_mkt * (1 - KALSHI_RAKE),
        -STAKE,
    )

    # Build YES rows and NO rows
    common_cols = ["ema_bias_15m", "stoch_k_15m", "composite_p_up",
                   "vol_ratio", "body_dir_15m",
                   "stoch_k_1h", "consec_dir_1h", "ema_bias_1h"]

    yes_df = df[common_cols].copy()
    yes_df["ts"]        = df.index
    yes_df["asset"]     = asset
    yes_df["side"]      = "yes"
    yes_df["spot"]      = spot
    yes_df["strike"]    = strike_yes
    yes_df["p_market"]  = p_yes_mkt
    yes_df["win"]       = yes_win
    yes_df["pnl"]       = yes_pnl

    no_df = df[common_cols].copy()
    no_df["ts"]         = df.index
    no_df["asset"]      = asset
    no_df["side"]       = "no"
    no_df["spot"]       = spot
    no_df["strike"]     = strike_no
    no_df["p_market"]   = p_no_mkt
    no_df["win"]        = no_win
    no_df["pnl"]        = no_pnl

    out = pd.concat([yes_df, no_df], ignore_index=True)
    out = out.reset_index(drop=True)
    return out


# ── Gate definitions (fixed thresholds, no training-data selection) ───────────

GATES = {
    # name → (side_filter, condition_fn, description)
    # side_filter: "yes" | "no" | "both"
    # condition_fn: takes trade row, returns True = BLOCK this trade
    "ema_bias_15m | YES block when bearish": (
        "yes",
        lambda r: r["ema_bias_15m"] < 0,
        "Block YES when EMA20 is below spot (bearish 15m trend)",
    ),
    "ema_bias_15m | NO block when bullish": (
        "no",
        lambda r: r["ema_bias_15m"] > 0,
        "Block NO when EMA20 is above spot (bullish 15m trend)",
    ),
    "stoch_k_15m | YES block >=50 (overbought)": (
        "yes",
        lambda r: r["stoch_k_15m"] >= 50,
        "Block YES when 15m stoch >= 50 (not oversold)",
    ),
    "stoch_k_15m | NO block <50 (oversold)": (
        "no",
        lambda r: r["stoch_k_15m"] < 50,
        "Block NO when 15m stoch < 50 (not overbought)",
    ),
    "composite_p_up | YES block <0.5 (model bearish)": (
        "yes",
        lambda r: r["composite_p_up"] < 0.5,
        "Block YES when composite_p_up < 0.5 (model sees bearish bias)",
    ),
    "composite_p_up | NO block >=0.5 (model bullish)": (
        "no",
        lambda r: r["composite_p_up"] >= 0.5,
        "Block NO when composite_p_up >= 0.5 (model sees bullish bias)",
    ),
    "vol_ratio | BOTH block >1.5 (elevated vol)": (
        "both",
        lambda r: r["vol_ratio"] > 1.5,
        "Block both sides when vol_ratio > 1.5 (realized >> implied)",
    ),
    "stoch_k_1h | YES block >70 (1h overbought)": (
        "yes",
        lambda r: r["stoch_k_1h"] > 70,
        "Block YES when 1h stoch > 70 (overbought on higher timeframe)",
    ),
    "stoch_k_1h | NO block <30 (1h oversold)": (
        "no",
        lambda r: r["stoch_k_1h"] < 30,
        "Block NO when 1h stoch < 30 (oversold on higher timeframe)",
    ),
    "consec_dir_1h | YES block <=0 (no bullish momentum)": (
        "yes",
        lambda r: r["consec_dir_1h"] <= 0,
        "Block YES when consecutive 1h closes not bullish",
    ),
    "ema_bias_1h | YES block when bearish": (
        "yes",
        lambda r: r["ema_bias_1h"] < 0,
        "Block YES when 1h EMA20 is below spot",
    ),
    "ema_bias_1h | NO block when bullish": (
        "no",
        lambda r: r["ema_bias_1h"] > 0,
        "Block NO when 1h EMA20 is above spot",
    ),
    "body_dir_15m | YES block bearish candle": (
        "yes",
        lambda r: r["body_dir_15m"] < 0,
        "Block YES when last 15m candle was bearish (close < open)",
    ),
}


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def eval_gate(trades: pd.DataFrame, side: str, block_mask: pd.Series,
              fold_mask: pd.Series) -> dict:
    """
    Within fold_mask period, compare baseline vs gated PnL for the given side.
    Returns dict with stats.
    """
    if side == "both":
        subset = trades[fold_mask].copy()
    else:
        subset = trades[fold_mask & (trades["side"] == side)].copy()

    if len(subset) < MIN_BARS:
        return None

    # Align block_mask to subset's integer index
    bm = block_mask.loc[subset.index]
    blocked  = subset[bm.values]
    allowed  = subset[~bm.values]

    base_pnl  = subset["pnl"].sum()
    gate_pnl  = allowed["pnl"].sum()
    delta_pnl = gate_pnl - base_pnl   # positive = gate helped

    n_blocked      = len(blocked)
    wins_blocked   = blocked["win"].sum()
    losses_blocked = (1 - blocked["win"]).sum()
    blocked_wr     = blocked["win"].mean() if n_blocked > 0 else float("nan")
    base_wr        = subset["win"].mean()
    allowed_wr     = allowed["win"].mean() if len(allowed) > 0 else float("nan")
    avg_p_market   = subset["p_market"].mean()
    breakeven_wr   = avg_p_market  # for YES: breakeven WR = p_market (cost of bet)

    return {
        "n_base":          len(subset),
        "n_allowed":       len(allowed),
        "n_blocked":       n_blocked,
        "wins_blocked":    int(wins_blocked),
        "losses_blocked":  int(losses_blocked),
        "base_wr":         base_wr,
        "blocked_wr":      blocked_wr,
        "allowed_wr":      allowed_wr,
        "breakeven_wr":    breakeven_wr,
        "base_pnl":        base_pnl,
        "gate_pnl":        gate_pnl,
        "delta_pnl":       delta_pnl,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("Walk-Forward Gate Test — BTC / ETH / SOL 15m Signals")
    print(f"Offset: {OFFSET*100:.2f}%  Stake: ${STAKE:.0f}  Rake: {KALSHI_RAKE*100:.0f}%")
    print(SEP)

    # ── Build trade dataset ──
    all_trades = []
    for asset, sym in SYMBOLS.items():
        print(f"\nLoading {asset} ({sym})...")
        try:
            df1m = load_1m(sym)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue
        print(f"  1m bars: {len(df1m):,}  ({df1m.index[0].date()} → {df1m.index[-1].date()})")

        df15m  = build_15m_signals(df1m)
        sig_1h = build_1h_signals(df1m, df15m)
        trades = simulate_trades(df15m, sig_1h, asset)
        all_trades.append(trades)
        print(f"  Simulated {len(trades):,} trade rows ({len(trades)//2:,} bars)")

    if not all_trades:
        print("No data loaded. Exiting.")
        sys.exit(1)

    trades = pd.concat(all_trades, ignore_index=True)
    trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
    trades = trades.sort_values("ts").reset_index(drop=True)

    print(f"\nTotal trade rows: {len(trades):,}  ({trades['ts'].iloc[0].date()} → {trades['ts'].iloc[-1].date()})")
    print(SEP)

    # ── Per-asset baseline ──
    print("\nBaseline (no gate, all trades):")
    print(f"{'Asset':<6} {'Side':<5} {'N':>7} {'WR':>7} {'BkEven':>8} {'PnL':>10}")
    print(SEP2)
    for asset in list(SYMBOLS.keys()) + ["ALL"]:
        for side in ["yes", "no"]:
            sub = trades[trades["side"] == side] if asset == "ALL" \
                else trades[(trades["asset"] == asset) & (trades["side"] == side)]
            if len(sub) == 0:
                continue
            wr  = sub["win"].mean()
            pnl = sub["pnl"].sum()
            bk  = sub["p_market"].mean()
            n   = len(sub)
            label = f"{asset:<6} {side:<5}"
            edge_flag = "  ←EDGE" if (wr > bk + 0.01) else ("  ←BELOW" if wr < bk - 0.01 else "")
            print(f"{label} {n:>7,} {wr:>7.1%} {bk:>8.1%} {pnl:>+10,.0f}{edge_flag}")

    print()

    # ── Walk-forward gate tests ──
    fold_defs = []
    for name, tr_start, tr_end, te_start, te_end in FOLDS:
        fold_defs.append({
            "name":     name,
            "te_start": te_start,
            "te_end":   te_end,
        })

    # Collect results for summary
    gate_summary = {}   # gate_name → {fold_name → result_dict}

    print(SEP)
    print("Walk-Forward Results by Gate")
    print("Columns: Fold | N_base | N_blocked | WR_blocked | BkEven | PnL_delta | W_blk | L_blk")
    print(SEP)

    for gate_name, (side, block_fn, description) in GATES.items():
        print(f"\n{'─'*76}")
        print(f"GATE: {gate_name}")
        print(f"  {description}")
        print(f"{'─'*76}")
        print(f"  {'Fold':<5} {'Asset':<6} {'N_base':>8} {'N_blk':>7} {'WR_blk':>8} "
              f"{'BkEven':>8} {'PnL_base':>10} {'PnL_gate':>10} {'DELTA':>9} {'W_blk':>6} {'L_blk':>6}")

        gate_summary[gate_name] = {}
        asset_fold_deltas = []    # for consistency score

        for asset in list(SYMBOLS.keys()):
            asset_trades = trades[trades["asset"] == asset].copy()
            if len(asset_trades) == 0:
                continue

            block_mask = asset_trades.apply(block_fn, axis=1)

            fold_results = []
            for fd in fold_defs:
                te_s = pd.Timestamp(fd["te_start"], tz="UTC")
                te_e = pd.Timestamp(fd["te_end"],   tz="UTC")
                test_mask_aligned = (asset_trades["ts"] >= te_s) & (asset_trades["ts"] < te_e)

                res = eval_gate(asset_trades, side, block_mask, test_mask_aligned)
                if res is None:
                    fold_results.append(None)
                    continue

                fold_results.append(res)
                asset_fold_deltas.append(res["delta_pnl"])

                delta_flag = "★" if res["delta_pnl"] > 0 else " "
                print(f"  {fd['name']:<5} {asset:<6} "
                      f"{res['n_base']:>8,} {res['n_blocked']:>7,} "
                      f"{res['blocked_wr']:>8.1%} {res['breakeven_wr']:>8.1%} "
                      f"{res['base_pnl']:>+10,.0f} {res['gate_pnl']:>+10,.0f} "
                      f"{res['delta_pnl']:>+9,.0f}{delta_flag} "
                      f"{res['wins_blocked']:>6} {res['losses_blocked']:>6}")

            gate_summary[gate_name][asset] = fold_results

        # Consistency: % of non-None folds where delta > 0
        valid_deltas = [d for d in asset_fold_deltas if not math.isnan(d)]
        if valid_deltas:
            consistency = sum(1 for d in valid_deltas if d > 0) / len(valid_deltas)
            total_delta = sum(valid_deltas)
            print(f"\n  CONSISTENCY: {consistency:.0%} of folds improved  |  "
                  f"Total delta across all folds/assets: ${total_delta:+,.0f}")

    # ── Summary table ──
    print()
    print(SEP)
    print("SUMMARY — Gate Consistency Across All Folds & Assets")
    print(f"{'Gate':<55} {'Consistency':>12} {'Total Δ PnL':>12}")
    print(SEP2)

    rows = []
    for gate_name, (side, block_fn, description) in GATES.items():
        all_deltas = []
        for asset_results in gate_summary[gate_name].values():
            for r in asset_results:
                if r is not None:
                    all_deltas.append(r["delta_pnl"])
        if not all_deltas:
            continue
        consistency = sum(1 for d in all_deltas if d > 0) / len(all_deltas)
        total_delta = sum(all_deltas)
        rows.append((gate_name, consistency, total_delta))

    rows.sort(key=lambda x: (-x[1], -x[2]))
    for gate_name, consistency, total_delta in rows:
        stars = "★★★" if consistency >= 0.75 else ("★★" if consistency >= 0.50 else "★")
        print(f"{gate_name:<55} {consistency:>12.0%} {total_delta:>+12,.0f}  {stars}")

    print()
    print("★★★ = consistent in ≥75% of folds  ★★ = 50-74%  ★ = <50%")
    print(SEP)
    print("Done. Results reflect TEST folds only (train data never evaluated).")
    print("All thresholds fixed by economic logic — no in-sample threshold search.")


if __name__ == "__main__":
    main()
