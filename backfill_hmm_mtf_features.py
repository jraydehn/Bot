"""
backfill_hmm_mtf_features.py

Computes the 10 HMM MTF momentum model features AND hmm_vol_state (R0/R1)
from Binance OHLCV parquets and writes them into the BTC scan archives.

Features: stoch_k_5m, stoch_k_15m, stoch_k_1h, rsi_1h, bp_1h, chg_1h,
          macd_hist_1h, adx_1h, macd_hist_4h, adx_4h, hmm_vol_state

  hmm_vol_state: 0=R0 (low-vol), 1=R1 (high-vol) — from hmm_ergodic_2state_btc_15m.pkl
                 Viterbi on 20-bar 15m log-return window at each scan row timestamp.

Usage:
  python3 backfill_hmm_mtf_features.py [--archive 1h|15m|all] [--dry-run]
"""
import argparse, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
RESULTS = BASE / "results"
DATA    = BASE / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--archive",  choices=["1h", "15m", "all"], default="all")
parser.add_argument("--dry-run",  action="store_true")
args = parser.parse_args()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _stoch(high, low, close, period=14):
    ll = low.rolling(period, min_periods=period).min()
    hh = high.rolling(period, min_periods=period).max()
    rng = (hh - ll).replace(0, np.nan)
    return (close - ll) / rng * 100


def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _bp(high, low, close):
    rng = (high - low).replace(0, np.nan)
    return (close - low) / rng


def _macd_hist(close, fast=12, slow=26, sig=9):
    e_fast = close.ewm(span=fast, adjust=False).mean()
    e_slow = close.ewm(span=slow, adjust=False).mean()
    macd   = e_fast - e_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal


def _adx(high, low, close, period=14):
    h_diff = high.diff()
    l_diff = -low.diff()
    dm_p = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
    dm_m = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, adjust=False).mean()
    pdi  = dm_p.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan) * 100
    mdi  = dm_m.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan) * 100
    dx   = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    return dx.ewm(span=period, adjust=False).mean()


# ── OHLCV loading ─────────────────────────────────────────────────────────────

def _latest_parquet(pattern):
    import os
    matches = list(DATA.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No parquet matching {pattern}")
    return max(matches, key=os.path.getmtime)


def load_ohlcv(freq_code, date_from, date_to, warmup_days=60):
    """Load Binance BTC OHLCV parquet, returning [date_from-warmup, date_to].

    freq_code: '1m', '1h', '4h'
    Returns DataFrame with UTC DatetimeIndex, columns open/high/low/close.
    """
    path = _latest_parquet(f"binanceus_BTCUSDT_{freq_code}_*.parquet")
    df   = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    def _to_utc(ts):
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    t0 = _to_utc(date_from) - pd.Timedelta(days=warmup_days)
    t1 = _to_utc(date_to)
    return df[(df.index >= t0) & (df.index <= t1)].copy()


# ── Indicator frame builder ───────────────────────────────────────────────────

def build_indicator_frame(ohlcv, freq_label):
    """Compute all required indicators for a given OHLCV DataFrame.

    Returns a DataFrame indexed by bar close time with columns named
    {feature}_{freq_label}, e.g. stoch_k_1h.
    """
    h = ohlcv["high"].astype(float)
    l = ohlcv["low"].astype(float)
    c = ohlcv["close"].astype(float)

    out = pd.DataFrame(index=ohlcv.index)
    out[f"stoch_k_{freq_label}"] = _stoch(h, l, c)
    out[f"rsi_{freq_label}"]     = _rsi(c)
    out[f"bp_{freq_label}"]      = _bp(h, l, c)
    out[f"chg_{freq_label}"]     = c.pct_change() * 100
    out[f"macd_hist_{freq_label}"] = _macd_hist(c)
    out[f"adx_{freq_label}"]     = _adx(h, l, c)
    return out


# ── Merge into archive ────────────────────────────────────────────────────────

def merge_indicators(archive, indicators, on="logged_at"):
    """Left-merge indicators into archive using merge_asof (backward lookup)."""
    ind = indicators.copy()
    ind.index.name = "_bar_ts"
    ind = ind.reset_index()
    arch_sorted = archive.sort_values(on).copy()
    merged = pd.merge_asof(
        arch_sorted,
        ind.sort_values("_bar_ts"),
        left_on=on,
        right_on="_bar_ts",
        direction="backward",
    ).drop(columns=["_bar_ts"])
    # Restore original index order
    return merged.set_index(archive.index.name or "index").reindex(archive.index)


# ── Process one archive ───────────────────────────────────────────────────────

def process_archive(csv_path, needed_feats, out_path=None):
    print(f"\n{'─'*60}")
    print(f"Archive: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df = df.reset_index(drop=True)

    # Existing features — skip if already full
    existing = {f: (f in df.columns and df[f].notna().mean() > 0.80)
                for f in needed_feats}
    to_add = [f for f, ok in existing.items() if not ok]
    print(f"  Needed: {needed_feats}")
    print(f"  Already populated (>80%): {[f for f, ok in existing.items() if ok]}")
    print(f"  To compute: {to_add}")

    if not to_add:
        print("  Nothing to do.")
        return

    t_min = df["logged_at"].min()
    t_max = df["logged_at"].max()
    print(f"  Date range: {t_min.date()} → {t_max.date()}")

    # ── Load and resample OHLCV ───────────────────────────────────────────────
    print("  Loading 1m parquet …", end=" ", flush=True)
    df_1m = load_ohlcv("1m", t_min, t_max)
    print(f"{len(df_1m)} bars.")

    # Resample to needed timeframes
    _RULE_MAP = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}

    def resample_ohlcv(df_1m, rule):
        r = _RULE_MAP.get(rule, rule)
        return df_1m.resample(r, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["close"])

    frames_needed = set()
    for f in to_add:
        if f.endswith("_5m"):   frames_needed.add("5m")
        elif f.endswith("_15m"): frames_needed.add("15m")
        elif f.endswith("_1h"):  frames_needed.add("1h")
        elif f.endswith("_4h"):  frames_needed.add("4h")

    ohlcv_by_tf = {}
    for tf in sorted(frames_needed):
        print(f"  Resampling to {tf} …", end=" ", flush=True)
        if tf == "1h":
            # Prefer dedicated 1h parquet (lower noise from resampling)
            try:
                ohlcv_by_tf[tf] = load_ohlcv("1h", t_min, t_max)
                print(f"{len(ohlcv_by_tf[tf])} bars (from 1h parquet).")
                continue
            except FileNotFoundError:
                pass
        if tf == "4h":
            try:
                ohlcv_by_tf[tf] = load_ohlcv("4h", t_min, t_max)
                print(f"{len(ohlcv_by_tf[tf])} bars (from 4h parquet).")
                continue
            except FileNotFoundError:
                pass
        ohlcv_by_tf[tf] = resample_ohlcv(df_1m, tf)
        print(f"{len(ohlcv_by_tf[tf])} bars.")

    # ── Compute indicators ────────────────────────────────────────────────────
    all_indicators = pd.DataFrame(index=df["logged_at"].sort_values())

    for tf, ohlcv in ohlcv_by_tf.items():
        print(f"  Computing indicators ({tf}) …", end=" ", flush=True)
        ind = build_indicator_frame(ohlcv, tf)
        # Keep only features we actually need
        keep = [c for c in ind.columns if c in to_add]
        if not keep:
            print("skipped (no needed features).")
            continue
        ind = ind[keep].dropna(how="all")
        print(f"done.  ({', '.join(keep)})")

        # merge_asof: for each logged_at find the most recent completed bar
        arch_ts = df[["logged_at"]].copy()
        arch_ts["_orig_idx"] = np.arange(len(arch_ts))
        arch_ts = arch_ts.dropna(subset=["logged_at"]).sort_values("logged_at")
        ind.index.name = "_bar_ts"
        ind_reset = ind.reset_index()
        merged_sub = pd.merge_asof(
            arch_ts,
            ind_reset.sort_values("_bar_ts"),
            left_on="logged_at",
            right_on="_bar_ts",
            direction="backward",
        ).drop(columns=["_bar_ts"])
        # Restore original row order
        merged_sub = merged_sub.set_index("_orig_idx").sort_index()
        for col in keep:
            df[col] = np.nan
            df.loc[merged_sub.index, col] = merged_sub[col].values

    # ── stoch_k_1h alias ─────────────────────────────────────────────────────
    if "stoch_k_1h" in to_add and "stoch_k" in df.columns:
        populated = df["stoch_k_1h"].notna().sum() if "stoch_k_1h" in df.columns else 0
        stoch_k_pop = df["stoch_k"].notna().sum()
        if stoch_k_pop > populated:
            print(f"  Aliasing stoch_k → stoch_k_1h "
                  f"({stoch_k_pop} values vs {populated} from OHLCV).")
            df["stoch_k_1h"] = df["stoch_k_1h"].fillna(df["stoch_k"].astype(float))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Coverage after backfill:")
    for f in needed_feats:
        if f in df.columns:
            pct = df[f].notna().mean()
            print(f"    {f:20s}: {pct:.1%}")
        else:
            print(f"    {f:20s}: MISSING")

    # ── hmm_vol_state (R0/R1) ─────────────────────────────────────────────────
    if "hmm_vol_state" not in df.columns or df["hmm_vol_state"].notna().mean() < 0.80:
        df["hmm_vol_state"] = backfill_vol_state(df, t_min, t_max)
        pct = df["hmm_vol_state"].notna().mean()
        print(f"    {'hmm_vol_state':20s}: {pct:.1%}")
    else:
        print(f"  hmm_vol_state already populated ({df['hmm_vol_state'].notna().mean():.1%}) — skipping.")

    if args.dry_run:
        print("  [dry-run] Not saving.")
        return

    dest = out_path if out_path is not None else csv_path
    if str(dest).endswith(".parquet"):
        df.to_parquet(dest, index=False)
    else:
        df.to_csv(dest, index=False)
    print(f"  Saved → {dest.name}")


# ── HMM vol state (R0/R1) backfill ───────────────────────────────────────────

_VOL_HMM_PKL = BASE / "models" / "hmm_ergodic_2state_btc_15m.pkl"
_LOOKBACK    = 20   # 20 × 15m = 5h context window


def _load_vol_hmm():
    with open(_VOL_HMM_PKL, "rb") as f:
        pkg = pickle.load(f)
    model   = pkg["model"]
    order   = sorted(range(model.n_components),
                     key=lambda s: float(np.sqrt(model.covars_[s, 0, 0])))
    rank_of = {s: i for i, s in enumerate(order)}
    return model, rank_of


def backfill_vol_state(df, t_min, t_max):
    """Compute hmm_vol_state for each row in df using a 20-bar 15m Viterbi window."""
    print("  Computing hmm_vol_state (R0/R1) …", end=" ", flush=True)
    model, rank_of = _load_vol_hmm()

    # Load 1m data and resample to 15m
    pq_1m = max(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"), key=lambda p: p.stat().st_mtime)
    df_1m = pd.read_parquet(pq_1m, columns=["close"])
    df_1m.index = pd.to_datetime(df_1m.index, utc=True)
    warmup = pd.Timedelta(days=10)   # enough for 20 × 15m bars + buffer
    def _utc(ts):
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    df_1m = df_1m[(_utc(t_min) - warmup <= df_1m.index) & (df_1m.index <= _utc(t_max))]
    c15   = df_1m["close"].resample("15min").last().dropna()
    lr    = np.log(c15 / c15.shift(1)).dropna()
    lr_idx  = lr.index
    lr_vals = lr.values

    states = []
    for ts in df["logged_at"]:
        if pd.isna(ts):
            states.append(np.nan)
            continue
        pos = lr_idx.searchsorted(ts, side="right") - 1
        if pos < _LOOKBACK:
            states.append(np.nan)
            continue
        window = lr_vals[pos - _LOOKBACK + 1: pos + 1].reshape(-1, 1)
        raw    = model.predict(window)
        states.append(float(rank_of[int(raw[-1])]))

    filled = sum(1 for s in states if not math.isnan(s) if s == s)
    print(f"done.  {filled}/{len(states)} rows filled.")
    return states


# ── Main ──────────────────────────────────────────────────────────────────────

HMM_FEATS = ["stoch_k_5m", "stoch_k_15m", "stoch_k_1h", "rsi_1h",
             "bp_1h", "chg_1h", "macd_hist_1h", "adx_1h",
             "macd_hist_4h", "adx_4h", "rsi_4h"]

# Output goes to separate files so the runner's _ensure_csv never clobbers them.
ARCHIVES = {
    "1h":  (RESULTS / "btc_scan_archive.csv",
            RESULTS / "btc_scan_archive_hmm.parquet",
            HMM_FEATS),
    "15m": (RESULTS / "btc_scan_archive_15m.csv",
            RESULTS / "btc_scan_archive_15m_hmm.parquet",
            HMM_FEATS),
}

targets = ["1h", "15m"] if args.archive == "all" else [args.archive]
for key in targets:
    src_path, out_path, feats = ARCHIVES[key]
    process_archive(src_path, feats, out_path=out_path)

print("\nDone.")
