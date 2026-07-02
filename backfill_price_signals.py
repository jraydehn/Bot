"""
backfill_price_signals.py — Backfill all price-derivable signals into combined BTC 1h paper trades.

Computes for each trade at time T using last-completed bar (no lookahead):
  4h: stoch_k_4h, stoch_d_4h, rsi_4h, macd_hist_4h, ema_stack_4h, chg_4h, bb_pct_4h, adx_4h
  1h: stoch_k_1h, stoch_d_1h, rsi_1h, macd_hist_1h, adx_1h_bf, rvol_1h_bf,
      chg_1h_bf, ema_stack_1h, bb_pct_1h, ema20_dist_1h, ema50_dist_1h
  1m: chg_5m_bf, chg_10m_bf, chg_30m_bf
  Regime: markov_1h (Bull/Bear/Sideways from 4-bar return),
           markov_daily (from 24-bar return)

Output: results/paper_trades_enriched.csv
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS = Path("results")


def load_all_trades():
    dfs = [pd.read_csv("results/paper_trades.csv", low_memory=False)]
    for p in sorted(glob.glob("results/paper_trades_archive_*.csv")):
        if not any(x in p.lower() for x in ["eth", "sol", "15m"]):
            try:
                dfs.append(pd.read_csv(p, low_memory=False))
            except Exception:
                pass
    df = pd.concat(dfs, ignore_index=True)
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
    df["resolved_yes"] = pd.to_numeric(df.get("resolved_yes", pd.Series(dtype=float)), errors="coerce")
    df["would_win"] = df["would_win"].map(
        {"True": True, "False": False, True: True, False: False,
         1: True, 0: False, "true": True, "false": False}
    )
    df["would_pnl"] = pd.to_numeric(df.get("would_pnl", pd.Series(dtype=float)), errors="coerce")
    df["p_market"]  = pd.to_numeric(df["p_market"], errors="coerce")
    return df


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _stoch(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    sk = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return sk, sk.rolling(d).mean()


def _macd_hist(close, fast=12, slow=26, sig=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    return macd - macd.ewm(span=sig, adjust=False).mean()


def _adx(high, low, close, n=14):
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    up = high.diff()
    dn = -low.diff()
    pdm = up.where((up > 0) & (up > dn), 0.0)
    ndm = dn.where((dn > 0) & (dn > up), 0.0)
    pdi = 100 * pdm.rolling(n).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.rolling(n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(n).mean()


def _ema_stack(close, periods=(20, 50, 100, 200)):
    emas = [close.ewm(span=p, adjust=False).mean() for p in periods]
    bull = (emas[0] > emas[1]) & (emas[1] > emas[2]) & (emas[2] > emas[3])
    bear = (emas[0] < emas[1]) & (emas[1] < emas[2]) & (emas[2] < emas[3])
    return (bull.astype(int) - bear.astype(int)).astype(float)


def _rvol(close, window=20):
    lr = np.log(close / close.shift())
    return lr.rolling(window).std() / lr.rolling(window * 3).std()


def _bb_pct(close, window=20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - (ma - 2 * std)) / (4 * std).replace(0, np.nan)


def _markov(ret, bull=0.005, bear=-0.005):
    s = pd.Series("Sideways", index=ret.index, dtype=object)
    s[ret > bull] = "Bull"
    s[ret < bear] = "Bear"
    return s


def build_4h():
    path = max(Path("data").glob("binanceus_BTCUSDT_4h_1970*"), key=lambda p: p.name)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    c, h, l = df["close"], df["high"], df["low"]
    ind = pd.DataFrame(index=df.index)
    ind["stoch_k_4h"], ind["stoch_d_4h"] = _stoch(h, l, c)
    ind["rsi_4h"]        = _rsi(c)
    ind["macd_hist_4h"]  = _macd_hist(c)
    ind["ema_stack_4h"]  = _ema_stack(c)
    ind["chg_4h"]        = c.pct_change() * 100
    ind["bb_pct_4h"]     = _bb_pct(c)
    ind["adx_4h"]        = _adx(h, l, c)
    return ind.shift(1)  # use last COMPLETED bar


def build_1h():
    path = max(Path("data").glob("binanceus_BTCUSDT_1h_1970*"), key=lambda p: p.name)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    c, h, l = df["close"], df["high"], df["low"]
    ind = pd.DataFrame(index=df.index)
    ind["stoch_k_1h"], ind["stoch_d_1h"] = _stoch(h, l, c)
    ind["rsi_1h"]        = _rsi(c)
    ind["macd_hist_1h"]  = _macd_hist(c)
    ind["adx_1h_bf"]     = _adx(h, l, c)
    ind["rvol_1h_bf"]    = _rvol(c)
    ind["chg_1h_bf"]     = c.pct_change() * 100
    ind["ema_stack_1h"]  = _ema_stack(c)
    ind["bb_pct_1h"]     = _bb_pct(c)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ind["ema20_dist_1h"] = (c - ema20) / ema20 * 100
    ind["ema50_dist_1h"] = (c - ema50) / ema50 * 100
    ind["markov_1h"]     = _markov(c.pct_change(4),  bull=0.004, bear=-0.004)
    ind["markov_daily"]  = _markov(c.pct_change(24), bull=0.010, bear=-0.010)
    return ind.shift(1)  # last COMPLETED bar


def build_1m_close():
    path = max(Path("data").glob("binanceus_BTCUSDT_1m_1970*"), key=lambda p: p.name)
    df = pd.read_parquet(path, columns=["close"]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df["close"]


if __name__ == "__main__":
    print("Loading all paper trades...")
    trades = load_all_trades()
    res = trades[trades["decision"].isin(["trade", "bet"])].dropna(subset=["resolved_yes"]).copy()
    print(f"  {len(res)} resolved trades | {res['logged_at'].min().date()} → {res['logged_at'].max().date()}")

    print("\nBuilding 4h indicators...")
    ind4h = build_4h()
    ind4h.index.name = "open_time"
    print(f"  {len(ind4h)} 4h bars | {ind4h.index.min()} → {ind4h.index.max()}")

    print("Building 1h indicators...")
    ind1h = build_1h()
    ind1h.index.name = "open_time"
    print(f"  {len(ind1h)} 1h bars")

    print("Loading 1m close...")
    close1m = build_1m_close()
    print(f"  {len(close1m)} 1m bars")

    # ── Merge 4h ────────────────────────────────────────────────────────────
    print("\nMerging 4h...")
    res["_4h_floor"] = res["logged_at"].dt.floor("4h")
    res = res.merge(
        ind4h.reset_index().rename(columns={"open_time": "_4h_floor"}),
        on="_4h_floor", how="left"
    )

    # ── Merge 1h ────────────────────────────────────────────────────────────
    print("Merging 1h...")
    res["_1h_floor"] = res["logged_at"].dt.floor("1h")
    res = res.merge(
        ind1h.reset_index().rename(columns={"open_time": "_1h_floor"}),
        on="_1h_floor", how="left"
    )

    # ── Merge 1m changes ────────────────────────────────────────────────────
    print("Merging 1m price changes...")
    c1m_df = close1m.reset_index()
    c1m_df.columns = ["_ts_1m", "_close_now"]
    res["_ts_1m"] = res["logged_at"].dt.floor("1min")
    res = res.merge(c1m_df, on="_ts_1m", how="left")
    for mins in [5, 10, 30]:
        bk = f"_ts_{mins}m_back"
        res[bk] = res["_ts_1m"] - pd.Timedelta(minutes=mins)
        bk_df = c1m_df.rename(columns={"_ts_1m": bk, "_close_now": f"_close_{mins}m_back"})
        res = res.merge(bk_df, on=bk, how="left")
        res[f"chg_{mins}m_bf"] = (
            (res["_close_now"] - res[f"_close_{mins}m_back"])
            / res[f"_close_{mins}m_back"].replace(0, np.nan) * 100
        )

    # Drop helper columns
    res = res.drop(columns=[c for c in res.columns if c.startswith("_")], errors="ignore")

    # ── Save ────────────────────────────────────────────────────────────────
    out = RESULTS / "paper_trades_enriched.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(res)} rows)")

    new_sigs = [
        "stoch_k_4h", "stoch_d_4h", "rsi_4h", "macd_hist_4h", "ema_stack_4h",
        "adx_4h", "chg_4h", "bb_pct_4h",
        "stoch_k_1h", "rsi_1h", "adx_1h_bf", "rvol_1h_bf", "chg_1h_bf",
        "ema_stack_1h", "bb_pct_1h", "ema20_dist_1h", "ema50_dist_1h",
        "markov_1h", "markov_daily",
        "chg_5m_bf", "chg_10m_bf", "chg_30m_bf",
    ]
    print("\nNew signal fill rates:")
    for s in new_sigs:
        if s in res.columns:
            n = res[s].notna().sum()
            print(f"  {s}: {n}/{len(res)} ({n/len(res)*100:.0f}%)")
