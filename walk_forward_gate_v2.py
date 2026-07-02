#!/usr/bin/env python3
"""
walk_forward_gate_v2.py

Walk-forward validation of gate signals on MODEL-FILTERED trades.
Uses existing backtest CSVs (already filtered for edge) as the base dataset,
then enriches each row with gate signals computed from 1m parquets via merge_asof.

This avoids the unconditional-bar problem: every row represents a trade the
model already decided to take, so gate tests are conditional on model edge.

Assets / sources:
  BTC 1h  backtest_full.csv       gate_passed=True        52,978 rows  all NO
  ETH 1h  backtest_eth_full.csv   net_edge > 0.04         83,662 rows  YES+NO
  SOL 1h  backtest_sol_full.csv   net_edge > 0.04         91,827 rows  YES+NO
  BTC 15m backtest_15m_flat.csv   decision==trade          1,957 rows  mostly NO

Gate signals computed from 1m parquets (shift-1, no look-ahead):
  ema_bias_1h       EMA20 vs spot on 1h bars
  stoch_k_1h        14-bar stochastic K (1h)
  composite_p_up    Simplified directional composite (EMA/RSI/VWAP/momentum, 1h)
  vol_ratio_1h      Realized 1h vol / rolling 2-week median
  consec_dir_1h     Consecutive same-direction 1h closes
  ema_bias_15m      EMA20 vs spot on 15m bars  (15m only)
  stoch_k_15m       14-bar stochastic K (15m)  (15m only)
  body_dir_15m      Candle body direction (15m) (15m only)

Walk-forward folds (expanding train, non-overlapping test — 4 folds):
  F1: Train 2024H1  Test 2024H2
  F2: Train 2024    Test 2025H1
  F3: Train 2024+2025H1  Test 2025H2
  F4: Train 2024+2025    Test 2026 YTD

Gate thresholds fixed by economic logic (no training-data search):
  stoch midpoint 50 / overbought 70 / oversold 30
  composite_p_up neutral 0.50
  vol_ratio elevated 1.50
  consec_dir neutral 0
  ema_bias categorical -1 / +1

Metric: PnL delta = gated_PnL − baseline_PnL per test fold.
Gate is beneficial if delta > 0 in ≥ 3 of 4 folds (consistent across regimes).
"""

import warnings, math, glob
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

FLAT_STAKE    = 50.0   # dollars per trade (non-compounding)
KALSHI_RAKE   = 0.07
MIN_FOLD_ROWS = 100    # skip fold/gate combos below this

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

FOLDS = [
    ("F1", "2024-01-01", "2024-07-01", "2024-07-01", "2025-01-01"),
    ("F2", "2024-01-01", "2025-01-01", "2025-01-01", "2025-07-01"),
    ("F3", "2024-01-01", "2025-07-01", "2025-07-01", "2026-01-01"),
    ("F4", "2024-01-01", "2026-01-01", "2026-01-01", "2026-06-01"),
]

SEP  = "=" * 76
SEP2 = "-" * 76


# ── Signal computation from 1m parquets ──────────────────────────────────────

def load_parquet(sym: str, tf: str) -> pd.DataFrame:
    files = sorted(DATA.glob(f"binanceus_{sym}_{tf}_2024-01-01_*.parquet"))
    if not files:
        files = sorted(DATA.glob(f"binanceus_{sym}_{tf}_*.parquet"),
                       key=lambda p: p.stat().st_size)
    if not files:
        raise FileNotFoundError(f"No {sym} {tf} parquet")
    df = pd.read_parquet(files[-1])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index().astype({"open": float, "high": float,
                                    "low": float, "close": float,
                                    "volume": float})


def stoch_k_series(close, high, low, period=14):
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    return ((close - lo) / (hi - lo).replace(0, np.nan) * 100).clip(0, 100)


def build_1h_signals(sym: str) -> pd.DataFrame:
    """
    Compute all 1h gate signals from 1m parquet.
    Returns a DataFrame indexed by 1h bar-open timestamp.
    All signals are shift(1) — known at bar open, not bar close.
    """
    df1m = load_parquet(sym, "1m")
    df = df1m.resample("1h", origin="start_day").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])

    c, h, l = df["close"], df["high"], df["low"]

    # 1. ema_bias_1h
    ema20 = c.ewm(span=20, adjust=False).mean()
    df["ema_bias_1h"] = (c > ema20).astype(float) * 2 - 1   # +1 bullish, -1 bearish

    # 2. stoch_k_1h
    df["stoch_k_1h"] = stoch_k_series(c, h, l, 14)

    # 3. composite_p_up (5 binary components → [0,1])
    ema50 = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
    tp    = (h + l + c) / 3
    vwap_proxy = tp.ewm(span=24, adjust=False).mean()   # ~24h EMA of typical price
    comp = (
        ((c > ema20).astype(float) * 2 - 1) +
        ((ema20 > ema50).astype(float) * 2 - 1) +
        ((rsi > 50).astype(float) * 2 - 1) +
        ((c > c.shift(1)).astype(float) * 2 - 1) +
        ((c > vwap_proxy).astype(float) * 2 - 1)
    )
    df["composite_p_up"] = (comp + 5) / 10.0

    # 4. vol_ratio_1h
    log_ret = np.log(c / c.shift(1))
    vol_1h  = log_ret.rolling(24).std() * math.sqrt(365 * 24)
    med_vol = vol_1h.rolling(14 * 24, min_periods=48).median()
    df["vol_ratio_1h"] = (vol_1h / med_vol.replace(0, np.nan)).clip(0, 10)

    # 5. consec_dir_1h
    dir1h = np.sign(c.diff()).fillna(0).values
    consec = np.zeros(len(dir1h))
    for i in range(1, len(dir1h)):
        d = dir1h[i]
        prev = consec[i - 1]
        if d == 0:
            consec[i] = 0
        elif (prev > 0 and d > 0) or (prev < 0 and d < 0):
            consec[i] = prev + d
        else:
            consec[i] = d
    df["consec_dir_1h"] = consec

    # Shift all signals by 1 bar
    sig_cols = ["ema_bias_1h", "stoch_k_1h", "composite_p_up",
                "vol_ratio_1h", "consec_dir_1h"]
    for col in sig_cols:
        df[col] = df[col].shift(1)

    return df[sig_cols]


def build_15m_signals(sym: str) -> pd.DataFrame:
    """Compute 15m-specific gate signals. Also returns 1h signals resampled to 15m."""
    df1m = load_parquet(sym, "1m")

    # 15m bars
    df15 = df1m.resample("15min", origin="start_day").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
    ).dropna(subset=["close"])
    c15, h15, l15 = df15["close"], df15["high"], df15["low"]

    df15["ema_bias_15m"]  = ((c15 > c15.ewm(span=20, adjust=False).mean()).astype(float)*2-1).shift(1)
    df15["stoch_k_15m"]   = stoch_k_series(c15, h15, l15, 14).shift(1)
    df15["body_dir_15m"]  = (np.sign(c15 - df15["open"]).shift(1))

    # Join 1h signals
    sig_1h = build_1h_signals(sym)
    df15 = df15.join(sig_1h, how="left")
    for col in sig_1h.columns:
        df15[col] = df15[col].ffill()

    keep = ["ema_bias_15m", "stoch_k_15m", "body_dir_15m"] + list(sig_1h.columns)
    return df15[keep]


# ── Load and enrich backtest datasets ────────────────────────────────────────

def enrich_with_signals(df_trades: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """
    Join precomputed signal series to trade rows via merge_asof (backward lookup).
    df_trades must have a 'ts' column (UTC). sig is indexed by bar-open timestamp.
    """
    sig_reset = sig.reset_index().rename(columns={"index": "bar_ts"})
    if sig_reset.columns[0] != "bar_ts":
        sig_reset = sig_reset.rename(columns={sig_reset.columns[0]: "bar_ts"})
    sig_reset["bar_ts"] = pd.to_datetime(sig_reset["bar_ts"], utc=True)

    trades_sorted = df_trades.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(
        trades_sorted,
        sig_reset,
        left_on="ts",
        right_on="bar_ts",
        direction="backward",
    )
    return merged


def load_btc_1h() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "backtest_full.csv", low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df[df["gate_passed"] == True].copy()
    df["win"]     = pd.to_numeric(df["win"], errors="coerce")
    df["pnl"]     = pd.to_numeric(df.get("pnl", pd.Series(dtype=float)), errors="coerce")
    df["p_market"]= pd.to_numeric(df["p_market"], errors="coerce")
    df["net_edge"]= pd.to_numeric(df["net_edge"], errors="coerce")
    # Compute flat-stake PnL from win + p_market
    df["flat_pnl"] = np.where(
        df["win"] == 1,
        FLAT_STAKE * (1 - df["p_market"]) / df["p_market"] * (1 - KALSHI_RAKE),
        -FLAT_STAKE,
    )
    df["asset"] = "BTC"
    return df[["ts", "asset", "side", "win", "flat_pnl", "p_market", "net_edge"]]


def load_eth_1h() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "backtest_eth_full.csv", low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ["win", "flat_pnl", "p_market", "net_edge"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    df = df[df["net_edge"] > 0.04].copy()
    if "flat_pnl" not in df.columns or df["flat_pnl"].isna().all():
        df["flat_pnl"] = np.where(
            df["win"] == 1,
            FLAT_STAKE * (1 - df["p_market"]) / df["p_market"] * (1 - KALSHI_RAKE),
            -FLAT_STAKE,
        )
    df["asset"] = "ETH"
    return df[["ts", "asset", "side", "win", "flat_pnl", "p_market", "net_edge"]]


def load_sol_1h() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "backtest_sol_full.csv", low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ["win", "flat_pnl", "p_market", "net_edge"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    df = df[df["net_edge"] > 0.04].copy()
    if "flat_pnl" not in df.columns or df["flat_pnl"].isna().all():
        df["flat_pnl"] = np.where(
            df["win"] == 1,
            FLAT_STAKE * (1 - df["p_market"]) / df["p_market"] * (1 - KALSHI_RAKE),
            -FLAT_STAKE,
        )
    df["asset"] = "SOL"
    return df[["ts", "asset", "side", "win", "flat_pnl", "p_market", "net_edge"]]


def load_btc_15m() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "backtest_15m_flat.csv", low_memory=False)
    df["ts"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
    df = df[df["decision"] == "trade"].copy()
    for c in ["resolved_yes", "model_correct", "pnl", "p_market", "net_edge"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    df["win"]      = df["model_correct"]
    df["flat_pnl"] = np.where(
        df["win"] == 1,
        FLAT_STAKE * (1 - df["p_market"]) / df["p_market"] * (1 - KALSHI_RAKE),
        -FLAT_STAKE,
    )
    df["asset"] = "BTC"
    return df[["ts", "asset", "side", "win", "flat_pnl", "p_market", "net_edge"]]


# ── Gate definitions ──────────────────────────────────────────────────────────

GATES_1H = {
    "ema_bias_1h | YES block bearish": (
        "yes", lambda r: r["ema_bias_1h"] < 0,
        "Block YES when 1h EMA20 below spot"),
    "ema_bias_1h | NO block bullish": (
        "no", lambda r: r["ema_bias_1h"] > 0,
        "Block NO when 1h EMA20 above spot"),
    "stoch_k_1h | YES block >70 (overbought)": (
        "yes", lambda r: r["stoch_k_1h"] > 70,
        "Block YES when 1h stoch > 70"),
    "stoch_k_1h | NO block <30 (oversold)": (
        "no", lambda r: r["stoch_k_1h"] < 30,
        "Block NO when 1h stoch < 30"),
    "composite_p_up | YES block <0.5 (bearish)": (
        "yes", lambda r: r["composite_p_up"] < 0.5,
        "Block YES when composite < 0.5"),
    "composite_p_up | NO block >=0.5 (bullish)": (
        "no", lambda r: r["composite_p_up"] >= 0.5,
        "Block NO when composite >= 0.5"),
    "vol_ratio_1h | BOTH block >1.5 (high vol)": (
        "both", lambda r: r["vol_ratio_1h"] > 1.5,
        "Block both sides when vol_ratio > 1.5"),
    "consec_dir_1h | YES block <=0 (no momentum)": (
        "yes", lambda r: r["consec_dir_1h"] <= 0,
        "Block YES when no bullish 1h momentum"),
    "consec_dir_1h | NO block >=0 (no bear momentum)": (
        "no", lambda r: r["consec_dir_1h"] >= 0,
        "Block NO when no bearish 1h momentum"),
}

GATES_15M = {
    **GATES_1H,
    "ema_bias_15m | YES block bearish": (
        "yes", lambda r: r["ema_bias_15m"] < 0,
        "Block YES when 15m EMA20 below spot"),
    "stoch_k_15m | YES block >=50 (overbought)": (
        "yes", lambda r: r["stoch_k_15m"] >= 50,
        "Block YES when 15m stoch >= 50"),
    "body_dir_15m | YES block bearish candle": (
        "yes", lambda r: r["body_dir_15m"] < 0,
        "Block YES when last 15m candle bearish"),
}


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def eval_fold(trades: pd.DataFrame, side: str, block_mask: pd.Series,
              te_start: str, te_end: str):
    ts = pd.Timestamp(te_start, tz="UTC")
    te = pd.Timestamp(te_end,   tz="UTC")
    fold_idx = (trades["ts"] >= ts) & (trades["ts"] < te)

    if side == "both":
        sub = trades[fold_idx].copy()
    else:
        sub = trades[fold_idx & (trades["side"] == side)].copy()

    if len(sub) < MIN_FOLD_ROWS:
        return None

    bm = block_mask.loc[sub.index]
    blocked = sub[bm.values]
    allowed = sub[~bm.values]

    if len(blocked) == 0:
        return None

    base_pnl   = sub["flat_pnl"].sum()
    gate_pnl   = allowed["flat_pnl"].sum()
    delta      = gate_pnl - base_pnl
    bk_wr      = sub["p_market"].mean() if side == "yes" else (1 - sub["p_market"].mean())

    return {
        "n_base":        len(sub),
        "n_blocked":     len(blocked),
        "base_wr":       sub["win"].mean(),
        "blocked_wr":    blocked["win"].mean(),
        "allowed_wr":    allowed["win"].mean() if len(allowed) > 0 else float("nan"),
        "bkeven_wr":     bk_wr,
        "base_pnl":      base_pnl,
        "gate_pnl":      gate_pnl,
        "delta_pnl":     delta,
        "wins_blocked":  int(blocked["win"].sum()),
        "losses_blocked":int((1 - blocked["win"]).sum()),
    }


def run_gates(label: str, trades: pd.DataFrame, gates: dict) -> dict:
    """Run all gates on trades. Returns {gate_name: [fold_result_or_None]}."""
    print(f"\n{'─'*76}")
    print(f"DATASET: {label}  ({len(trades):,} model-filtered trades)")
    wr_all = trades["win"].mean()
    bk_all = trades[trades["side"]=="yes"]["p_market"].mean() if (trades["side"]=="yes").any() else float("nan")
    print(f"  Overall WR: {wr_all:.1%}  |  YES bkeven: {bk_all:.1%}  |  "
          f"Flat PnL: ${trades['flat_pnl'].sum():+,.0f}")
    print(f"  Date range: {trades['ts'].min().date()} → {trades['ts'].max().date()}")
    print(f"  Side mix: {trades['side'].value_counts().to_dict()}")
    print(f"{'─'*76}")

    results = {}
    for gate_name, (side, block_fn, desc) in gates.items():
        # Skip 15m-specific gates if column not present
        needed_col = None
        if "15m" in gate_name:
            needed_col = gate_name.split("|")[0].strip()
            if needed_col not in trades.columns:
                continue

        block_mask = trades.apply(block_fn, axis=1)
        fold_res   = []
        deltas     = []

        print(f"\n  GATE: {gate_name}")
        print(f"  {desc}")
        print(f"  {'Fold':<5} {'N_base':>8} {'N_blk':>7} {'WR_blk':>8} "
              f"{'BkEven':>8} {'PnL_base':>10} {'PnL_gate':>10} {'DELTA':>9} "
              f"{'W_blk':>6} {'L_blk':>6}")

        for fname, tr_s, tr_e, te_s, te_e in FOLDS:
            res = eval_fold(trades, side, block_mask, te_s, te_e)
            fold_res.append(res)
            if res:
                deltas.append(res["delta_pnl"])
                flag = "★" if res["delta_pnl"] > 0 else " "
                print(f"  {fname:<5} {res['n_base']:>8,} {res['n_blocked']:>7,} "
                      f"{res['blocked_wr']:>8.1%} {res['bkeven_wr']:>8.1%} "
                      f"{res['base_pnl']:>+10,.0f} {res['gate_pnl']:>+10,.0f} "
                      f"{res['delta_pnl']:>+9,.0f}{flag} "
                      f"{res['wins_blocked']:>6} {res['losses_blocked']:>6}")
            else:
                print(f"  {fname:<5} {'(insufficient data)':>60}")

        if deltas:
            consistency = sum(1 for d in deltas if d > 0) / len(deltas)
            print(f"  Consistency: {consistency:.0%}  |  Total Δ: ${sum(deltas):+,.0f}")

        results[gate_name] = fold_res

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("Walk-Forward Gate Test v2 — Model-Filtered Trades")
    print(f"Stake: ${FLAT_STAKE:.0f} flat  Rake: {KALSHI_RAKE*100:.0f}%")
    print(SEP)

    all_results = {}

    # ── BTC 1h ──
    print("\nLoading BTC 1h backtest + signals...")
    btc_trades = load_btc_1h()
    btc_sig    = build_1h_signals("BTCUSDT")
    btc_trades = enrich_with_signals(btc_trades, btc_sig)
    res = run_gates("BTC 1h (gate_passed=True, all NO)", btc_trades, GATES_1H)
    all_results["BTC_1h"] = res

    # ── ETH 1h ──
    print("\n\nLoading ETH 1h backtest + signals...")
    eth_trades = load_eth_1h()
    eth_sig    = build_1h_signals("ETHUSDT")
    eth_trades = enrich_with_signals(eth_trades, eth_sig)
    res = run_gates("ETH 1h (net_edge>0.04, YES+NO)", eth_trades, GATES_1H)
    all_results["ETH_1h"] = res

    # ── SOL 1h ──
    print("\n\nLoading SOL 1h backtest + signals...")
    sol_trades = load_sol_1h()
    sol_sig    = build_1h_signals("SOLUSDT")
    sol_trades = enrich_with_signals(sol_trades, sol_sig)
    res = run_gates("SOL 1h (net_edge>0.04, YES+NO)", sol_trades, GATES_1H)
    all_results["SOL_1h"] = res

    # ── BTC 15m ──
    print("\n\nLoading BTC 15m backtest + signals...")
    btc15_trades = load_btc_15m()
    btc15_sig    = build_15m_signals("BTCUSDT")
    btc15_trades = enrich_with_signals(btc15_trades, btc15_sig)
    res = run_gates("BTC 15m (decision=trade)", btc15_trades, GATES_15M)
    all_results["BTC_15m"] = res

    # ── Summary table ──
    print()
    print(SEP)
    print("SUMMARY — Cross-Dataset Consistency")
    print(f"{'Gate':<52} {'BTC_1h':>8} {'ETH_1h':>8} {'SOL_1h':>8} {'BTC_15m':>8} {'Overall':>8}")
    print(SEP2)

    # Collect all gate names across datasets
    all_gate_names = set()
    for ds_res in all_results.values():
        all_gate_names.update(ds_res.keys())

    summary_rows = []
    for gate_name in sorted(all_gate_names):
        per_ds = []
        total_deltas = []
        for ds_key in ["BTC_1h", "ETH_1h", "SOL_1h", "BTC_15m"]:
            ds_res = all_results.get(ds_key, {})
            fold_list = ds_res.get(gate_name, [])
            deltas = [r["delta_pnl"] for r in fold_list if r is not None]
            if deltas:
                c = sum(1 for d in deltas if d > 0) / len(deltas)
                per_ds.append(f"{c:.0%}")
                total_deltas.extend(deltas)
            else:
                per_ds.append("  —  ")
        if total_deltas:
            overall = sum(1 for d in total_deltas if d > 0) / len(total_deltas)
            stars = "★★★" if overall >= 0.75 else ("★★" if overall >= 0.50 else "★")
            summary_rows.append((overall, sum(total_deltas), gate_name, per_ds, stars))

    summary_rows.sort(key=lambda x: (-x[0], -x[1]))
    for overall, tot_delta, gate_name, per_ds, stars in summary_rows:
        ds_str = "  ".join(f"{v:>8}" for v in per_ds)
        print(f"{gate_name:<52} {ds_str}  {overall:>8.0%}  {stars}  ${tot_delta:+,.0f}")

    print()
    print("★★★ ≥75% consistent  ★★ 50-74%  ★ <50%")
    print(SEP)
    print("All signals shift(1) — no look-ahead within the bar.")
    print("All thresholds fixed before running — no in-sample threshold search.")


if __name__ == "__main__":
    main()
