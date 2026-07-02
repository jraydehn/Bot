"""
mine_live_losses.py
-------------------
Load resolved live trades for BTC/ETH/SOL, join 1h indicator signals at
each trade timestamp, then compare signal distributions between wins and
losses to surface candidate gate conditions.

Signals computed (all shift(1) — no look-ahead):
  ema_bias_1h    : +1 if close > EMA20, -1 otherwise
  stoch_k_1h     : 14-period stochastic %K
  composite_p_up : 5-signal composite mapped to [0,1]
  vol_ratio_1h   : realized 1h vol / rolling 14-day median
  ema_stack_1h   : count of 4 EMAs (20/50/100/200) below price  (0-4)
  rsi_1h         : RSI-14
"""

import pandas as pd
import numpy as np
import glob, os

DATA_DIR = "data"
STAKE    = 50.0

# ── Helpers ──────────────────────────────────────────────────────────────────

def latest_parquet(sym, tf):
    pattern = os.path.join(DATA_DIR, f"binanceus_{sym}USDT_{tf}_2024-01-01_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet: {pattern}")
    return files[-1]

def load_1m(sym):
    df = pd.read_parquet(latest_parquet(sym, "1m"))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC")
    return df

def stoch_k(c, h, l, period=14):
    lo = l.rolling(period).min()
    hi = h.rolling(period).max()
    denom = (hi - lo).replace(0, np.nan)
    return (c - lo) / denom * 100

def build_signals(sym):
    df1m = load_1m(sym)
    df = df1m.resample("1h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    c = df["close"]; h = df["high"]; l = df["low"]

    ema20  = c.ewm(span=20,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema100 = c.ewm(span=100, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    df["ema_bias_1h"]  = ((c > ema20).astype(float) * 2 - 1)
    df["stoch_k_1h"]   = stoch_k(c, h, l, 14)
    df["rsi_1h"]       = _rsi(c, 14)
    df["ema_stack_1h"] = (
        (c > ema20).astype(int) +
        (c > ema50).astype(int) +
        (c > ema100).astype(int) +
        (c > ema200).astype(int)
    )

    # composite_p_up: 5 binary signals → [0,1]
    mom   = (c - c.shift(1)) > 0
    rsi_g = df["rsi_1h"] > 50
    vwap  = c  # proxy; can't compute true VWAP on 1h without tick data
    comp  = (
        (ema20 > ema50).astype(int) +
        (c > ema50).astype(int) +
        rsi_g.astype(int) +
        mom.astype(int) +
        (c > c.rolling(24).mean()).astype(int)  # price vs 24h MA as VWAP proxy
    )
    df["composite_p_up"] = (comp + 0) / 5.0  # 0–1

    # vol_ratio_1h: 1h realized-vol / 14-day rolling median
    log_ret    = np.log(c / c.shift(1))
    rv_1h      = log_ret.rolling(60).std() * np.sqrt(60)   # 60-bar window
    median_14d = rv_1h.rolling(14 * 24).median()
    df["vol_ratio_1h"] = rv_1h / median_14d.replace(0, np.nan)

    # shift all signals by 1 bar — no look-ahead
    sig_cols = ["ema_bias_1h","stoch_k_1h","rsi_1h","ema_stack_1h",
                "composite_p_up","vol_ratio_1h"]
    df[sig_cols] = df[sig_cols].shift(1)

    return df[sig_cols].reset_index().rename(columns={"open_time":"ts","index":"ts"})

def _rsi(c, period=14):
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def load_trades(f):
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["logged_at"], utc=True)
    resolved = df[df["resolved_yes"].notna()].copy()
    resolved["win"] = resolved["live_pnl"] > 0
    return resolved

def enrich(trades, sig_df):
    sig_df = sig_df.sort_values("ts")
    trades = trades.sort_values("ts")
    merged = pd.merge_asof(trades, sig_df, on="ts", direction="backward")
    return merged

def summarize(df, label, side_filter=None):
    sub = df if side_filter is None else df[df["side"] == side_filter]
    if len(sub) < 5:
        return
    wins   = sub[sub["win"]]
    losses = sub[~sub["win"]]
    wr     = sub["win"].mean()
    pnl    = sub["live_pnl"].sum()
    tag    = f"{label}" + (f" {side_filter}" if side_filter else "")
    print(f"\n{'─'*70}")
    print(f"  {tag}: n={len(sub)}  WR={wr:.1%}  PnL=${pnl:+.2f}")
    print(f"  Wins={len(wins)}  Losses={len(losses)}")
    print(f"  {'Signal':<20}  {'WINS mean':>10}  {'LOSSES mean':>11}  {'Δ':>8}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*11}  {'─'*8}")
    sig_cols = ["ema_bias_1h","stoch_k_1h","rsi_1h","ema_stack_1h",
                "composite_p_up","vol_ratio_1h"]
    diffs = []
    for col in sig_cols:
        if col not in sub.columns or sub[col].isna().all():
            continue
        wm = wins[col].mean()
        lm = losses[col].mean()
        diff = lm - wm
        diffs.append((col, wm, lm, diff))
        print(f"  {col:<20}  {wm:>10.3f}  {lm:>11.3f}  {diff:>+8.3f}")

    print()
    # Threshold analysis: find signal thresholds where losses concentrate
    print(f"  Threshold clustering (losses vs wins):")
    for col, wm, lm, diff in sorted(diffs, key=lambda x: abs(x[3]), reverse=True)[:4]:
        sig = sub[[col,"win","live_pnl"]].dropna()
        if len(sig) < 10: continue
        # Bucket into quartiles
        try:
            sig["bucket"] = pd.qcut(sig[col], 4, duplicates="drop")
        except Exception:
            continue
        bkt = sig.groupby("bucket", observed=True).agg(
            n=("win","count"), wr=("win","mean"), pnl=("live_pnl","sum")
        ).reset_index()
        bkt["loss_n"] = (bkt["n"] * (1 - bkt["wr"])).round().astype(int)
        print(f"\n  {col} quartiles:")
        print(f"  {'Bucket':<25}  {'N':>4}  {'WR':>6}  {'PnL':>8}  {'Losses':>6}")
        for _, row in bkt.iterrows():
            flag = " ◄" if row["wr"] < 0.45 else ""
            print(f"  {str(row['bucket']):<25}  {row['n']:>4}  {row['wr']:>5.1%}  {row['pnl']:>8.2f}  {row['loss_n']:>6}{flag}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    assets = [
        ("BTC", "results/live_trades.csv"),
        ("ETH", "results/live_trades_eth.csv"),
        ("SOL", "results/live_trades_sol.csv"),
    ]

    for sym, trade_file in assets:
        print(f"\n{'='*70}")
        print(f"  ASSET: {sym}")
        print(f"{'='*70}")

        print(f"  Loading {sym} signals...", end="", flush=True)
        try:
            sig = build_signals(sym)
        except Exception as e:
            print(f" ERROR: {e}")
            continue
        print(f" done ({len(sig)} 1h bars)")

        trades = load_trades(trade_file)
        enriched = enrich(trades, sig)

        print(f"  Enriched {len(enriched)} trades "
              f"({enriched[sig.columns[1:]].notna().all(axis=1).sum()} with full signals)")

        # Overall
        summarize(enriched, sym)

        # By side
        for side in ["YES", "NO"]:
            summarize(enriched, sym, side)

if __name__ == "__main__":
    run()
