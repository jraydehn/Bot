#!/usr/bin/env python3
"""
simulate_multitf_signals.py

Gate sweep: body_1h / dir_1h (last completed hourly bar) and bp_15m
(buyer pressure on last completed 15m bar) against the hourly paper trade log.

For each asset (BTC, ETH, SOL) show baseline + gate scenarios with
N / WR / PnL / blk — same format as the bp_5m/body_15m sweep.

Run: python3 simulate_multitf_signals.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA   = Path("data")
TRADES = {
    "BTC": ("results/paper_trades.csv",     "KXBTC"),
    "ETH": ("results/paper_trades_eth.csv", "KXETH"),
    "SOL": ("results/paper_trades_sol.csv", "KXSOL"),
}
SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
SEP  = "=" * 80


# ── candle loading ─────────────────────────────────────────────────────────────

def latest_parquet(sym: str, tf: str) -> Path:
    # Prefer full-history files (2024-01-01 start) sorted by end date; fall back to largest
    full = sorted(DATA.glob(f"binanceus_{sym}_{tf}_2024-01-01_*.parquet"))
    if full:
        return full[-1]
    files = sorted(DATA.glob(f"binanceus_{sym}_{tf}_*.parquet"), key=lambda p: p.stat().st_size)
    if not files:
        raise FileNotFoundError(f"No {sym} {tf} parquet found")
    return files[-1]


def load_signals(sym: str) -> pd.DataFrame:
    """Return a DataFrame indexed by bar-open timestamp with body_1h, dir_1h, bp_15m."""
    print(f"  Loading {sym} 1m candles...", end=" ", flush=True)
    df1m = pd.read_parquet(latest_parquet(sym, "1m"))
    if not isinstance(df1m.index, pd.DatetimeIndex):
        df1m.index = pd.to_datetime(df1m.index, utc=True)
    elif df1m.index.tz is None:
        df1m.index = df1m.index.tz_localize("UTC")
    print(f"{len(df1m):,} bars")

    # ── bp_15m: (close - low) / (high - low) on last completed 15m bar ─────────
    df15 = df1m.resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
    ).dropna()
    rng15 = df15["high"] - df15["low"]
    df15["bp_15m"] = np.where(rng15 > 0, (df15["close"] - df15["low"]) / rng15, 0.5)

    # ── body_1h / dir_1h: body ratio on last completed 1h bar ──────────────────
    df1h = df1m.resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
    ).dropna()
    rng1h = df1h["high"] - df1h["low"]
    df1h["body_1h"] = np.where(rng1h > 0,
                               (df1h["close"] - df1h["open"]).abs() / rng1h, 0.0)
    df1h["dir_1h"]  = np.where(df1h["close"] >= df1h["open"], 1, -1)

    return df15[["bp_15m"]], df1h[["body_1h", "dir_1h"]]


# ── trade loading + signal join ────────────────────────────────────────────────

def load_trades(asset: str) -> pd.DataFrame:
    csv_path, prefix = TRADES[asset]
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["contract_ticker"].str.startswith(prefix, na=False)].copy()
    df = df[df["would_pnl"].notna()].copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"], format="mixed", utc=True, errors="coerce")
    df["would_win"] = pd.to_numeric(df["would_win"], errors="coerce")
    df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")
    df["p_market"]  = pd.to_numeric(df["p_market"],  errors="coerce")
    df = df[df["would_win"].notna() & df["would_pnl"].notna()].copy()
    return df.reset_index(drop=True)


def join_signals(trades: pd.DataFrame,
                 df15: pd.DataFrame,
                 df1h: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()

    # last completed bar: floor to bar open, then back one period
    floor15 = t["decision_time"].dt.floor("15min") - pd.Timedelta(minutes=15)
    floor1h = t["decision_time"].dt.floor("1h")    - pd.Timedelta(hours=1)

    t["bp_15m"]  = floor15.map(df15["bp_15m"])
    t["body_1h"] = floor1h.map(df1h["body_1h"])
    t["dir_1h"]  = floor1h.map(df1h["dir_1h"])

    # one-bar fallback
    t["bp_15m"]  = t["bp_15m"].fillna((floor15 - pd.Timedelta(minutes=15)).map(df15["bp_15m"]))
    t["body_1h"] = t["body_1h"].fillna((floor1h - pd.Timedelta(hours=1)).map(df1h["body_1h"]))
    t["dir_1h"]  = t["dir_1h"].fillna((floor1h - pd.Timedelta(hours=1)).map(df1h["dir_1h"]))

    n_bp15  = t["bp_15m"].notna().sum()
    n_b1h   = t["body_1h"].notna().sum()
    print(f"  bp_15m matched: {n_bp15}/{len(t)}  body_1h matched: {n_b1h}/{len(t)}")
    return t


# ── gate sweep helpers ─────────────────────────────────────────────────────────

def _row(label: str, sub: pd.DataFrame, baseline_n: int) -> str:
    n   = len(sub)
    wr  = sub["would_win"].mean()
    pnl = sub["would_pnl"].sum()
    blk = baseline_n - n
    return f"  {label:<58}  N={n:>4}  WR={wr:.1%}  PnL=${pnl:>+7,.0f}  blk={blk}"


def gate_sweep(asset: str, trades: pd.DataFrame):
    t = trades[trades["bp_15m"].notna() & trades["body_1h"].notna()].copy()
    print(f"\n{SEP}")
    print(f"{asset} 1hr — N={len(t)}  WR={t['would_win'].mean():.1%}  "
          f"PnL=${t['would_pnl'].sum():+,.0f}")
    print(SEP)
    print(_row("Baseline", t, len(t)))

    # ── bp_15m gates ─────────────────────────────────────────────────────────
    print("\n  ── bp_15m gates ──")
    for thresh, label in [(0.3, "bearish pressure"), (0.4, ""), (0.5, "")]:
        lbl = f"Block YES when bp_15m<{thresh}" + (f" ({label})" if label else "")
        sub = t[~((t["side"] == "yes") & (t["bp_15m"] < thresh))]
        print(_row(lbl, sub, len(t)))

    for thresh, label in [(0.7, "bullish pressure"), (0.6, "")]:
        lbl = f"Block NO when bp_15m>{thresh}" + (f" ({label})" if label else "")
        sub = t[~((t["side"] == "no") & (t["bp_15m"] > thresh))]
        print(_row(lbl, sub, len(t)))

    sub = t[~(
        ((t["side"] == "yes") & (t["bp_15m"] < 0.4)) |
        ((t["side"] == "no")  & (t["bp_15m"] > 0.6))
    )]
    print(_row("YES<0.4 AND NO>0.6 combined", sub, len(t)))

    # ── body_1h gates ─────────────────────────────────────────────────────────
    print("\n  ── body_1h gates ──")
    for thresh, label in [(0.7, ""), (0.5, "thresh=0.5")]:
        lbl = f"Block YES when large bearish 1h bar (body>={thresh}, dir=-1)"
        sub = t[~((t["side"] == "yes") & (t["body_1h"] >= thresh) & (t["dir_1h"] == -1))]
        print(_row(lbl, sub, len(t)))

        lbl = f"Block NO when large bullish 1h bar (body>={thresh}, dir=+1)"
        sub = t[~((t["side"] == "no") & (t["body_1h"] >= thresh) & (t["dir_1h"] == 1))]
        print(_row(lbl, sub, len(t)))

        lbl = f"Both: contra-bar gate on both sides thresh={thresh}"
        sub = t[~(
            ((t["side"] == "yes") & (t["body_1h"] >= thresh) & (t["dir_1h"] == -1)) |
            ((t["side"] == "no")  & (t["body_1h"] >= thresh) & (t["dir_1h"] ==  1))
        )]
        print(_row(lbl, sub, len(t)))

    # ── bp_15m + body_1h combined ─────────────────────────────────────────────
    print("\n  ── bp_15m + body_1h combined ──")
    for bp_t, body_t in [(0.4, 0.7), (0.4, 0.5), (0.5, 0.5)]:
        lbl = f"bp_15m<{bp_t} blocks YES  +  body_1h>={body_t} contra-bar both"
        sub = t[~(
            ((t["side"] == "yes") & (t["bp_15m"] < bp_t)) |
            ((t["side"] == "yes") & (t["body_1h"] >= body_t) & (t["dir_1h"] == -1)) |
            ((t["side"] == "no")  & (t["body_1h"] >= body_t) & (t["dir_1h"] ==  1))
        )]
        print(_row(lbl, sub, len(t)))

    # ── compare vs existing body_15m (if in CSV) ─────────────────────────────
    if "body_15m" in t.columns and t["body_15m"].notna().sum() > 10:
        print("\n  ── body_15m (already in model, for reference) ──")
        sub = t[~((t["side"] == "yes") & (t["body_15m"] >= 0.7) & (t["dir_15m"] == -1))]
        print(_row("Block YES when body_15m>=0.7 bearish (current gate)", sub, len(t)))
        sub = t[~(
            ((t["side"] == "yes") & (t["body_15m"] >= 0.5) & (t["dir_15m"] == -1)) |
            ((t["side"] == "no")  & (t["body_15m"] >= 0.5) & (t["dir_15m"] ==  1))
        )]
        print(_row("body_15m>=0.5 contra both (current gate, thresh=0.5)", sub, len(t)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Multi-timeframe signal sweep: bp_15m + body_1h vs hourly model")
    print(SEP)

    for asset in ("BTC", "ETH", "SOL"):
        sym = SYMS[asset]
        df15, df1h = load_signals(sym)
        trades = load_trades(asset)
        print(f"  {asset}: {len(trades)} resolved trades")
        trades = join_signals(trades, df15, df1h)
        gate_sweep(asset, trades)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
