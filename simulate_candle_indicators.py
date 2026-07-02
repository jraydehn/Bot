"""
simulate_candle_indicators.py

Backtest candlestick indicators as gates/rescues against live paper trade data.
Tests all 6 asset×model combos: BTC/ETH/SOL × 15m/1hr.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "data"
RESULTS_DIR = "results"


# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_1m_parquet(sym: str) -> str:
    """Return the latest 2024-01-01_* parquet for the given symbol."""
    pattern = os.path.join(DATA_DIR, f"binanceus_{sym}_1m_2024-01-01_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No 1m parquet found for {sym}")
    return files[-1]


def load_1m(sym: str) -> pd.DataFrame:
    path = find_latest_1m_parquet(sym)
    df = pd.read_parquet(path)
    # Normalise index to UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m OHLCV to the requested period."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df_1m.resample(rule, closed="left", label="left").agg(agg).dropna(subset=["open"])
    return out


def wick_indicators(ohlc: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Compute upper/lower wick + imbalance for an OHLCV frame."""
    rng = ohlc["high"] - ohlc["low"]
    safe_rng = rng.replace(0, np.nan)
    upper = ((ohlc["high"] - ohlc[["open", "close"]].max(axis=1)) / safe_rng).clip(0, 1)
    lower = ((ohlc[["open", "close"]].min(axis=1) - ohlc["low"]) / safe_rng).clip(0, 1)
    imbal = lower - upper
    body = (ohlc["close"] - ohlc["open"]).abs() / safe_rng
    direction = np.where(ohlc["close"] >= ohlc["open"], 1, -1)
    out = pd.DataFrame({
        f"upper_wick_{suffix}": upper,
        f"lower_wick_{suffix}": lower,
        f"wick_imbalance_{suffix}": imbal,
        f"body_{suffix}": body.clip(0, 1),
        f"dir_{suffix}": direction,
    }, index=ohlc.index)
    return out


def compute_all_indicators(df_1m: pd.DataFrame) -> dict:
    """
    Compute all candlestick indicators at 5m, 15m, and 1h timeframes.
    Returns dict: {timeframe_str: pd.DataFrame indexed by bar open_time}
    """
    frames = {}

    # ── 5m ──────────────────────────────────────────────────────────────────
    ohlc_5m = resample_ohlcv(df_1m, "5min")
    ind_5m = wick_indicators(ohlc_5m, "5m")
    frames["5m"] = ind_5m

    # ── 15m ─────────────────────────────────────────────────────────────────
    ohlc_15m = resample_ohlcv(df_1m, "15min")
    ind_15m = wick_indicators(ohlc_15m, "15m")

    # Pattern indicators
    uw = ind_15m["upper_wick_15m"]
    lw = ind_15m["lower_wick_15m"]
    bd = ind_15m["body_15m"]
    ind_15m["is_hammer_15m"] = ((lw >= 0.50) & (bd <= 0.30)).astype(int)
    ind_15m["is_shooting_star_15m"] = ((uw >= 0.50) & (bd <= 0.30)).astype(int)
    ind_15m["is_doji_15m"] = (bd < 0.10).astype(int)
    ind_15m["is_pin_bar_15m"] = ((lw.combine(uw, max) >= 0.60) & (bd <= 0.20)).astype(int)

    # Engulfing (current vs previous bar)
    prev_dir = ind_15m["dir_15m"].shift(1)
    prev_body = ind_15m["body_15m"].shift(1)
    cur_dir = ind_15m["dir_15m"]
    cur_body = ind_15m["body_15m"]
    bull_eng = (cur_dir == 1) & (prev_dir == -1) & (cur_body > prev_body)
    bear_eng = (cur_dir == -1) & (prev_dir == 1) & (cur_body > prev_body)
    ind_15m["engulfing_15m"] = np.where(bull_eng, 1, np.where(bear_eng, -1, 0))

    # Consecutive direction (signed streak)
    dirs = ind_15m["dir_15m"].values
    streak = np.zeros(len(dirs), dtype=int)
    for i in range(len(dirs)):
        if i == 0:
            streak[i] = int(dirs[i])
        else:
            if dirs[i] == dirs[i - 1]:
                streak[i] = streak[i - 1] + int(dirs[i])
            else:
                streak[i] = int(dirs[i])
    streak = np.clip(streak, -5, 5)
    ind_15m["consecutive_dir_15m"] = streak

    # Range ratio (current range / 20-bar rolling mean range)
    rng_15m = ohlc_15m["high"] - ohlc_15m["low"]
    ind_15m["range_ratio_15m"] = (rng_15m / rng_15m.rolling(20, min_periods=5).mean()).clip(0, 5)

    frames["15m"] = ind_15m

    # ── 1h ──────────────────────────────────────────────────────────────────
    ohlc_1h = resample_ohlcv(df_1m, "1h")
    ind_1h = wick_indicators(ohlc_1h, "1h")
    rng_1h = ohlc_1h["high"] - ohlc_1h["low"]
    ind_1h["range_ratio_1h"] = (rng_1h / rng_1h.rolling(10, min_periods=3).mean()).clip(0, 5)
    frames["1h"] = ind_1h

    return frames


def join_indicators(df_trades: pd.DataFrame, frames: dict) -> pd.DataFrame:
    """
    For each trade, find the last completed bar for each timeframe and join indicators.
    Returns df_trades with indicator columns appended.
    """
    periods = {"5m": pd.Timedelta("5min"), "15m": pd.Timedelta("15min"), "1h": pd.Timedelta("1h")}

    result = df_trades.copy()

    for tf, ind_df in frames.items():
        period = periods[tf]
        idx = ind_df.index  # sorted UTC timestamps (bar open times)

        # For each trade decision_time, compute last_completed_bar = floor(dt, period) - period
        dt = result["decision_time_utc"]  # UTC Timestamps

        # Vectorised floor + offset
        def get_bar_ts(decision_ts):
            """Floor to period boundary then go back one period."""
            if pd.isna(decision_ts):
                return pd.NaT
            floored = decision_ts.floor(period)
            return floored - period

        bar_ts_series = dt.apply(get_bar_ts)

        # Build lookup: Series(value, index=bar_open_time)
        lookup = ind_df  # DataFrame with bar open_time as index

        # Map each column
        for col in ind_df.columns:
            col_lookup = lookup[col]
            mapped = bar_ts_series.map(col_lookup)
            # Fallback: if NaN, try one more period back
            mask_nan = mapped.isna()
            if mask_nan.any():
                bar_ts_fallback = bar_ts_series[mask_nan].apply(
                    lambda ts: ts - period if pd.notna(ts) else pd.NaT
                )
                mapped_fb = bar_ts_fallback.map(col_lookup)
                mapped[mask_nan] = mapped_fb
            result[col] = mapped

    return result


# ── trade loading ─────────────────────────────────────────────────────────────

def load_trades_15m(csv_path: str) -> pd.DataFrame:
    """Load 15m model CSV, keep only actual resolved trades."""
    df = pd.read_csv(csv_path)
    # Parse decision_time
    if "decision_time" in df.columns:
        dt = pd.to_datetime(df["decision_time"], errors="coerce", format="mixed", utc=True)
    else:
        dt = pd.to_datetime(df["logged_at"], errors="coerce", format="mixed", utc=True)
    df["decision_time_utc"] = dt
    # Keep only trades (not passes) with resolved outcomes
    df = df[df["decision"] == "trade"].copy()
    df = df[df["would_pnl"].notna() & df["would_win"].notna()].copy()
    df["would_win"] = pd.to_numeric(df["would_win"], errors="coerce")
    df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")
    df = df.dropna(subset=["would_win", "would_pnl"])
    return df.reset_index(drop=True)


def load_trades_1hr(csv_path: str) -> pd.DataFrame:
    """Load hourly model CSV, keep only actual resolved trades."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "decision_time" in df.columns:
        dt = pd.to_datetime(df["decision_time"], errors="coerce", format="mixed", utc=True)
    else:
        dt = pd.to_datetime(df["logged_at"], errors="coerce", format="mixed", utc=True)
    df["decision_time_utc"] = dt
    df = df[df["decision"] == "trade"].copy()
    df = df[df["would_pnl"].notna() & df["would_win"].notna()].copy()
    df["would_win"] = pd.to_numeric(df["would_win"], errors="coerce")
    df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")
    df = df.dropna(subset=["would_win", "would_pnl"])
    return df.reset_index(drop=True)


# ── gate sweep ────────────────────────────────────────────────────────────────

def gate_result(df: pd.DataFrame, mask_keep: pd.Series, label: str, baseline_n: int):
    """
    Compute stats for the kept subset.
    mask_keep = True for rows to KEEP (not blocked).
    Returns: (label, n, wr, pnl, blk, wins_blocked, losses_blocked)
    """
    kept = df[mask_keep]
    blocked = df[~mask_keep]
    n = len(kept)
    wr = kept["would_win"].mean() if n > 0 else 0.0
    pnl = kept["would_pnl"].sum()
    blk = baseline_n - n
    wins_blocked = int(blocked["would_win"].sum()) if len(blocked) > 0 else 0
    losses_blocked = int((1 - blocked["would_win"]).sum()) if len(blocked) > 0 else 0
    return label, n, wr, pnl, blk, wins_blocked, losses_blocked


STAR_PNL_DELTA = 200.0  # flag if PnL improves by this much
MIN_N_RATIO = 0.60      # must keep >= 60% of baseline trades


def format_gate_row(label, n, wr, pnl, blk, baseline_n, baseline_pnl, star=False):
    flag = " ★" if star else "  "
    pnl_str = f"${pnl:+.0f}"
    return f"{flag} {label:<65} N={n:4d}  WR={wr*100:.1f}%  PnL={pnl_str:>8}  blk={blk}"


def print_rescue_check(df: pd.DataFrame, blocked_mask: pd.Series, rescue_col: str,
                       rescue_threshold: float, rescue_op: str, label: str):
    """Show rescue: within blocked trades, how many are rescued by rescue_col op threshold."""
    blocked = df[blocked_mask]
    if len(blocked) == 0:
        return
    if rescue_op == ">":
        rescue_mask = blocked[rescue_col].notna() & (blocked[rescue_col] > rescue_threshold)
    elif rescue_op == "<":
        rescue_mask = blocked[rescue_col].notna() & (blocked[rescue_col] < rescue_threshold)
    else:
        return
    rescued = blocked[rescue_mask]
    if len(rescued) == 0:
        return
    wr_r = rescued["would_win"].mean()
    pnl_r = rescued["would_pnl"].sum()
    blk_r = len(blocked) - len(rescued)
    print(f"      Rescue {rescue_col}{rescue_op}{rescue_threshold}: {label} [{len(rescued)} rescued, WR={wr_r*100:.1f}%, PnL=${pnl_r:+.0f}, still-blk={blk_r}]")


def sweep_continuous(df: pd.DataFrame, col: str, thresholds: list,
                     baseline_n: int, baseline_pnl: float,
                     yes_col: str = "side", rescue_col: str = None) -> list:
    """
    For a continuous signal, sweep thresholds for:
      - Block YES when col >= thresh
      - Block NO when col <= thresh  (for wick/pressure signals this is inverted)
      - Block YES when col <= thresh
      - Block NO when col >= thresh
    Returns list of (label, n, wr, pnl, blk, wins_blocked, losses_blocked, is_star)
    """
    results = []
    valid = df[col].notna()

    for thresh in thresholds:
        # Gate: block YES when col >= thresh  (e.g. strong upper wick = bearish rejection, block YES bets)
        mask_yes = df["side"] == "yes"
        block_yes_high = valid & mask_yes & (df[col] >= thresh)
        keep = ~block_yes_high
        # include rows where signal is NA (keep them)
        keep = keep | (~valid)
        lbl = f"Block YES when {col}>={thresh:.2f}"
        row = gate_result(df, keep, lbl, baseline_n)
        pnl_delta = row[3] - baseline_pnl
        is_star = (pnl_delta > STAR_PNL_DELTA) and (row[1] >= baseline_n * MIN_N_RATIO)
        results.append((*row, is_star))

        # Gate: block NO when col <= thresh (e.g. low lower wick = weak demand, block NO bets)
        mask_no = df["side"] == "no"
        block_no_low = valid & mask_no & (df[col] <= thresh)
        keep = ~block_no_low | (~valid)
        lbl = f"Block NO when {col}<={thresh:.2f}"
        row = gate_result(df, keep, lbl, baseline_n)
        pnl_delta = row[3] - baseline_pnl
        is_star = (pnl_delta > STAR_PNL_DELTA) and (row[1] >= baseline_n * MIN_N_RATIO)
        results.append((*row, is_star))

        # Gate: block YES when col <= thresh
        block_yes_low = valid & mask_yes & (df[col] <= thresh)
        keep = ~block_yes_low | (~valid)
        lbl = f"Block YES when {col}<={thresh:.2f}"
        row = gate_result(df, keep, lbl, baseline_n)
        pnl_delta = row[3] - baseline_pnl
        is_star = (pnl_delta > STAR_PNL_DELTA) and (row[1] >= baseline_n * MIN_N_RATIO)
        results.append((*row, is_star))

        # Gate: block NO when col >= thresh
        block_no_high = valid & mask_no & (df[col] >= thresh)
        keep = ~block_no_high | (~valid)
        lbl = f"Block NO when {col}>={thresh:.2f}"
        row = gate_result(df, keep, lbl, baseline_n)
        pnl_delta = row[3] - baseline_pnl
        is_star = (pnl_delta > STAR_PNL_DELTA) and (row[1] >= baseline_n * MIN_N_RATIO)
        results.append((*row, is_star))

    return results


def sweep_binary(df: pd.DataFrame, col: str, baseline_n: int, baseline_pnl: float) -> list:
    """For binary indicators (is_hammer, engulfing, etc.), test gates directly."""
    results = []
    valid = df[col].notna()

    # values to test depend on column
    unique_vals = df.loc[valid, col].unique()

    for val in sorted(unique_vals):
        if val == 0:
            continue
        for side_filter in ["yes", "no", "both"]:
            if side_filter == "both":
                block_mask = valid & (df[col] == val)
            else:
                block_mask = valid & (df[col] == val) & (df["side"] == side_filter)

            if block_mask.sum() < 3:
                continue

            keep = ~block_mask | (~valid)
            lbl = f"Block {side_filter.upper()} when {col}=={val}"
            row = gate_result(df, keep, lbl, baseline_n)
            pnl_delta = row[3] - baseline_pnl
            is_star = (pnl_delta > STAR_PNL_DELTA) and (row[1] >= baseline_n * MIN_N_RATIO)
            results.append((*row, is_star))

    return results


def print_section(title: str, rows: list, df: pd.DataFrame, baseline_n: int, baseline_pnl: float,
                  rescue_col: str = None):
    """Print a labeled group of gate results, sorted by PnL delta desc."""
    if not rows:
        return
    print(f"\n  ── {title} ──")
    # Sort by PnL descending
    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    for row in rows_sorted:
        label, n, wr, pnl, blk, wins_blk, losses_blk, is_star = row
        flag = " ★" if is_star else "  "
        pnl_str = f"${pnl:+.0f}"
        print(f"{flag} {label:<65} N={n:4d}  WR={wr*100:.1f}%  PnL={pnl_str:>8}  blk={blk} (W-blk={wins_blk}/L-blk={losses_blk})")

        # Rescue check for large blocks
        if blk > baseline_n * 0.20 and rescue_col and rescue_col in df.columns:
            # reconstruct block mask from label (approximate)
            pass  # rescue shown separately in sweep


def run_asset_model(asset_label: str, model_label: str, df: pd.DataFrame) -> list:
    """
    Run full gate sweep for one asset×model combination.
    Returns list of top findings for summary.
    """
    baseline_n = len(df)
    baseline_pnl = df["would_pnl"].sum()
    baseline_wr = df["would_win"].mean()

    print()
    print("=" * 80)
    print(f"{asset_label} {model_label} — N={baseline_n}  WR={baseline_wr*100:.1f}%  PnL=${baseline_pnl:+.0f}")
    print("=" * 80)
    pnl_str = f"${baseline_pnl:+.0f}"
    print(f"   Baseline{'':<57} N={baseline_n:4d}  WR={baseline_wr*100:.1f}%  PnL={pnl_str:>8}  blk=0")

    if baseline_n < 10:
        print("  [SKIP — too few trades]")
        return []

    all_top = []

    # ── 5m indicators ───────────────────────────────────────────────────────
    print("\n  ════ 5m Indicators ════")
    for col, thresholds, note in [
        ("upper_wick_5m", [0.25, 0.40, 0.60], "rejection at highs"),
        ("lower_wick_5m", [0.15, 0.30, 0.50], "demand at lows"),
        ("wick_imbalance_5m", [-0.20, 0.00, 0.20], "buying/selling pressure"),
    ]:
        if col not in df.columns:
            continue
        rows = sweep_continuous(df, col, thresholds, baseline_n, baseline_pnl)
        print_section(f"{col} ({note})", rows, df, baseline_n, baseline_pnl)
        # Collect best rows
        for r in rows:
            if r[7]:  # is_star
                all_top.append(r[:7])

        # Rescue search for large-block results
        for row in rows:
            label, n, wr, pnl, blk, wins_blk, losses_blk, is_star = row
            if blk > baseline_n * 0.20:
                # Try wick_imbalance rescue within block
                rescue_col = "wick_imbalance_5m"
                if rescue_col in df.columns and rescue_col != col:
                    # Reconstruct block from label
                    _parse_and_print_rescue(df, label, col, rescue_col, baseline_n)

    # ── 15m indicators ──────────────────────────────────────────────────────
    print("\n  ════ 15m Indicators ════")

    for col, thresholds, note in [
        ("upper_wick_15m", [0.25, 0.40, 0.60], "rejection at highs"),
        ("lower_wick_15m", [0.15, 0.30, 0.50], "demand at lows"),
        ("wick_imbalance_15m", [-0.20, 0.00, 0.20], "buying/selling pressure"),
        ("range_ratio_15m", [0.70, 1.00, 1.50], "volatility expansion"),
        ("consecutive_dir_15m", [1, 2, 3], "directional streak"),
    ]:
        if col not in df.columns:
            continue
        rows = sweep_continuous(df, col, thresholds, baseline_n, baseline_pnl)
        print_section(f"{col} ({note})", rows, df, baseline_n, baseline_pnl)
        for r in rows:
            if r[7]:
                all_top.append(r[:7])

    for col, note in [
        ("is_hammer_15m", "bullish reversal"),
        ("is_shooting_star_15m", "bearish reversal"),
        ("is_doji_15m", "indecision"),
        ("is_pin_bar_15m", "strong rejection"),
        ("engulfing_15m", "engulfing pattern"),
    ]:
        if col not in df.columns:
            continue
        rows = sweep_binary(df, col, baseline_n, baseline_pnl)
        print_section(f"{col} ({note})", rows, df, baseline_n, baseline_pnl)
        for r in rows:
            if r[7]:
                all_top.append(r[:7])

    # ── 1h indicators ───────────────────────────────────────────────────────
    print("\n  ════ 1h Indicators ════")

    for col, thresholds, note in [
        ("upper_wick_1h", [0.25, 0.40, 0.60], "rejection at highs"),
        ("lower_wick_1h", [0.15, 0.30, 0.50], "demand at lows"),
        ("wick_imbalance_1h", [-0.20, 0.00, 0.20], "buying/selling pressure"),
        ("body_1h", [0.10, 0.30, 0.50], "strong directional bar"),
        ("range_ratio_1h", [0.70, 1.00, 1.50], "volatility expansion"),
    ]:
        if col not in df.columns:
            continue
        rows = sweep_continuous(df, col, thresholds, baseline_n, baseline_pnl)
        print_section(f"{col} ({note})", rows, df, baseline_n, baseline_pnl)
        for r in rows:
            if r[7]:
                all_top.append(r[:7])

    for col, note in [
        ("dir_1h", "1h direction"),
    ]:
        if col not in df.columns:
            continue
        rows = sweep_binary(df, col, baseline_n, baseline_pnl)
        print_section(f"{col} ({note})", rows, df, baseline_n, baseline_pnl)
        for r in rows:
            if r[7]:
                all_top.append(r[:7])

    # Return top 5 by PnL delta
    all_top_sorted = sorted(all_top, key=lambda r: r[3] - baseline_pnl, reverse=True)
    return all_top_sorted[:5], baseline_n, baseline_pnl


def _parse_and_print_rescue(df: pd.DataFrame, label: str, col: str, rescue_col: str, baseline_n: int):
    """Re-parse the gate condition from label string and show rescue."""
    try:
        # Parse label: "Block YES when col>=0.40" or "Block NO when col<=0.15"
        parts = label.split()
        side = parts[1].lower()
        op_thresh = parts[-1]
        if ">=" in op_thresh:
            op, thresh_str = ">=", op_thresh.split(">=")[1]
        elif "<=" in op_thresh:
            op, thresh_str = "<=", op_thresh.split("<=")[1]
        else:
            return
        thresh = float(thresh_str)
        valid = df[col].notna()

        if side == "yes":
            side_mask = df["side"] == "yes"
        elif side == "no":
            side_mask = df["side"] == "no"
        else:
            side_mask = pd.Series(True, index=df.index)

        if op == ">=":
            block_mask = valid & side_mask & (df[col] >= thresh)
        else:
            block_mask = valid & side_mask & (df[col] <= thresh)

        # Rescue: wick_imbalance > 0 within block
        for rescue_thresh, rescue_op in [(0.10, ">"), (-0.10, "<")]:
            print_rescue_check(df, block_mask, rescue_col, rescue_thresh, rescue_op,
                                f"within-block edge rescue")
    except Exception:
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading parquet data...")
    indicators_by_sym = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"  Loading {sym} 1m data and computing indicators...")
        df_1m = load_1m(sym)
        indicators_by_sym[sym] = compute_all_indicators(df_1m)
    print("Done loading.\n")

    # (asset_label, model_label, csv_path, sym, loader_fn)
    configs = [
        ("BTC", "15m",  "results/paper_trades_btc15m.csv", "BTCUSDT", "15m"),
        ("ETH", "15m",  "results/paper_trades_eth15m.csv", "ETHUSDT", "15m"),
        ("SOL", "15m",  "results/paper_trades_sol15m.csv", "SOLUSDT", "15m"),
        ("BTC", "1hr",  "results/paper_trades.csv",        "BTCUSDT", "1hr"),
        ("ETH", "1hr",  "results/paper_trades_eth.csv",    "ETHUSDT", "1hr"),
        ("SOL", "1hr",  "results/paper_trades_sol.csv",    "SOLUSDT", "1hr"),
    ]

    summary = {}  # key -> (top_rows, baseline_n, baseline_pnl)

    for asset, model, csv_path, sym, model_type in configs:
        key = f"{asset} {model}"
        print(f"\nProcessing {key}...")

        if model_type == "15m":
            df_trades = load_trades_15m(csv_path)
        else:
            df_trades = load_trades_1hr(csv_path)

        print(f"  Loaded {len(df_trades)} resolved trades")

        # Join indicators
        frames = indicators_by_sym[sym]
        df_with_ind = join_indicators(df_trades, frames)

        # Show join coverage
        for col in ["upper_wick_5m", "upper_wick_15m", "upper_wick_1h"]:
            if col in df_with_ind.columns:
                cov = df_with_ind[col].notna().sum()
                total = len(df_with_ind)
                print(f"  {col} coverage: {cov}/{total} ({100*cov/total:.0f}%)")

        result = run_asset_model(asset, f"{model}", df_with_ind)
        if result:
            top_rows, baseline_n, baseline_pnl = result
            summary[key] = (top_rows, baseline_n, baseline_pnl)

    # ── Summary of Top Findings ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY OF TOP FINDINGS")
    print("=" * 80)

    for key, (top_rows, baseline_n, baseline_pnl) in summary.items():
        print(f"\n{key} (baseline N={baseline_n}, PnL=${baseline_pnl:+.0f})")
        if not top_rows:
            print("  [No starred gates found — check near-miss gates in output above]")
            continue
        for i, row in enumerate(top_rows, 1):
            label, n, wr, pnl, blk, wins_blk, losses_blk = row
            delta = pnl - baseline_pnl
            flag = " ★" if (delta > STAR_PNL_DELTA and n >= baseline_n * MIN_N_RATIO) else "  "
            print(f"  {i}.{flag}{label}")
            print(f"      N={n} WR={wr*100:.1f}% PnL=${pnl:+.0f} (delta={delta:+.0f}) blk={blk} W-blk={wins_blk}/L-blk={losses_blk}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
