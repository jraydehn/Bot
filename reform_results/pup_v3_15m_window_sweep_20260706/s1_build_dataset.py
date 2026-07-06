"""
S1 — Build M-style intra-window microstructure features at multiple window
sizes (15/30/45/60 min), sampled at every 15-min bar boundary, to test
whether a shorter/longer lookback than the hourly model's 60-min window
suits the 15-min BTC contract's decision cadence better.

Same feature *shapes* as the hourly M group (rv, rv_z_10d, upmin_frac,
ret_first(0.75W), ret_last(0.25W), maxdd, volskew_last(1/3 W)), just
computed over a trailing W-minute window ending at each 15-min mark instead
of a fixed non-overlapping 60-min bucket.

Label: direction of the NEXT 15-min bar (price(T+15) vs price(T)) --
directly analogous to the hourly model's "bar T+1 close vs bar T close".

CAUSAL ALIGNMENT (the part that's easy to get subtly wrong): a 1-minute
candle indexed at timestamp X has open=price(X), close=price(X+1min). So
"price at clock time T" = close of the candle indexed (T - 1min), NOT the
candle indexed T itself (that candle covers [T, T+1min), i.e. the future).
Every "price at time X" lookup below goes through price_at(X), which
applies this -1min offset consistently. Likewise, pandas' time-based
.rolling(window) is right-closed by default -- rolling(...).loc[T] would
include the [T, T+1min) candle. Sampling the rolling series at
(bar_index - 1min) instead of bar_index avoids that leak.

Data: data/binanceus_BTCUSDT_1m_1970-01-01_2026-07-06.parquet. The 1970
row is a fetch-sentinel artifact (single placeholder row) -- real
continuous coverage starts 2024-01-01.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
WINDOWS = [15, 30, 45, 60]
ONE_MIN = pd.Timedelta(minutes=1)

m = pd.read_parquet("data/binanceus_BTCUSDT_1m_1970-01-01_2026-07-06.parquet")
m = m[m.index >= "2024-01-01"].sort_index()
m = m[~m.index.duplicated(keep="last")]
print(f"1m rows: {len(m)}  ({m.index.min()} -> {m.index.max()})")

r1m = np.log(m["close"] / m["close"].shift(1))
dt_min = m.index.to_series().diff().dt.total_seconds() / 60.0
r1m[dt_min > 2.0] = np.nan  # guard the one known ~34h data gap (2025-10-10)
m["_r1"] = r1m
m["_r2"] = r1m ** 2
m["_up"] = (m["close"] > m["open"]).astype(float)


def price_at(ts_index: pd.DatetimeIndex) -> pd.Series:
    """Price achieved BY clock time ts (causally safe): close of the
    1m candle indexed ts-1min, since that candle covers [ts-1min, ts)."""
    return m["close"].reindex(ts_index - ONE_MIN).set_axis(ts_index)


def rolling_at(series: pd.Series, ts_index: pd.DatetimeIndex) -> pd.Series:
    """Sample a right-closed time-rolling series at ts-1min (last fully
    completed minute strictly before ts) instead of ts itself."""
    return series.reindex(ts_index - ONE_MIN).set_axis(ts_index)


# 15-min bar boundary timeline (the decision cadence) -- every 15-min mark
# for which we have a full trailing 60min (largest window) of history.
bar_index = pd.date_range(m.index.min() + pd.Timedelta(minutes=60),
                          m.index.max(), freq="15min")
# next bar's boundary, 15 min later, for the label
next_bar_index = bar_index + pd.Timedelta(minutes=15)

price_T = price_at(bar_index)
price_T15 = price_at(next_bar_index)
label = (price_T15.values > price_T.values).astype(float)
label[np.isnan(price_T15.values) | np.isnan(price_T.values)] = np.nan

feat_frames = {}
for W in WINDOWS:
    minp = max(int(W * 0.7), 5)
    rv2_roll = m["_r2"].rolling(f"{W}min", min_periods=minp).sum()
    up_roll  = m["_up"].rolling(f"{W}min", min_periods=minp).mean()
    vol_roll = m["volume"].rolling(f"{W}min", min_periods=minp).sum()

    n_last3 = max(1, round(W / 3))
    vol_last3_roll = m["volume"].rolling(f"{n_last3}min", min_periods=1).sum()

    roll_max = m["close"].rolling(f"{W}min", min_periods=minp).max()
    dd = m["close"] / roll_max - 1.0
    maxdd_roll = dd.rolling(f"{W}min", min_periods=minp).min()

    df = pd.DataFrame(index=bar_index)
    df[f"rv_{W}"] = np.sqrt(rolling_at(rv2_roll, bar_index).values)
    df[f"upmin_frac_{W}"] = rolling_at(up_roll, bar_index).values
    df[f"maxdd_{W}"] = rolling_at(maxdd_roll, bar_index).values
    df[f"volskew_last_{W}"] = (rolling_at(vol_last3_roll, bar_index).values /
                               rolling_at(vol_roll, bar_index).values)

    # split point: first 75% / last 25% of the window (matches the 45/15
    # of 60 = 3:1 ratio used by the hourly M group). Rounded to whole
    # minutes -- 1m data has no fractional-minute timestamps to align to.
    first_min = round(W * 0.75)
    last_min = W - first_min
    close_T       = price_at(bar_index)
    close_Wback   = price_at(bar_index - pd.Timedelta(minutes=W))
    close_splitback = price_at(bar_index - pd.Timedelta(minutes=last_min)) if last_min >= 1 else None

    if close_splitback is not None:
        df[f"ret_first_{W}"] = np.log(close_splitback.values / close_Wback.values)
        df[f"ret_last_{W}"] = np.log(close_T.values / close_splitback.values)
    else:
        df[f"ret_first_{W}"] = np.nan
        df[f"ret_last_{W}"] = np.log(close_T.values / close_Wback.values)

    lrv = np.log(df[f"rv_{W}"].replace(0, np.nan))
    df[f"rv_{W}_z10d"] = (lrv - lrv.rolling(960, min_periods=200).mean()) / \
                          lrv.rolling(960, min_periods=200).std().replace(0, np.nan)

    feat_frames[W] = df
    print(f"W={W:2d}min: built {list(df.columns)}")

full = pd.concat(feat_frames.values(), axis=1)
full["label"] = label
full["close"] = price_T.values

full = full.dropna(subset=["label"])
full.to_parquet(f"{OUT}/window_sweep_dataset.parquet")
print(f"\nsaved {OUT}/window_sweep_dataset.parquet: {full.shape}")
print(f"range: {full.index.min()} -> {full.index.max()}")
print(f"label balance: up={full['label'].mean():.3f}")
