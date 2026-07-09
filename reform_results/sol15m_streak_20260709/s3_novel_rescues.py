"""
S3 -- Five novel rescue angles WITHIN the z_drift_6h<0.55 block bucket (n=559),
none of which a single-feature quantile sweep can see:

  1. z_drift dynamics: delta over the past hour + time-continuously-below-0.55
     (from the scan archive's per-cycle z_drift series, asof-joined causally).
  2. Multi-TF momentum-absence conjunction: chg_5m<=0 AND chg_15m<=0 (and the
     danger complement: both positive = live momentum against the NO).
  3. Cross-asset: BTC's own 15m/1h momentum + stoch at the SOL trade timestamp
     (zero-lookahead from BTC 1m parquet).
  4. ATR-normalized strike distance: offset_pct / atr_ratio_15m.
  5. Episode position: 1st vs 2nd+ trade within the same <=45min episode.

Bar for a rescue: episode-clustered P(edge<=0) <= 0.05, n>=80, ZERO streak
leakage, >=60% positive weeks. Bar for a gate-tightening (danger subset):
concentrates the bucket's losses with the remainder ~flat.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1021)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

no = pd.read_csv(f"{OUT}/no_book_reconstructed.csv", low_memory=False)
no["logged_at_p"] = pd.to_datetime(no["logged_at_p"], utc=True)
no["week"] = no["logged_at_p"].dt.to_period("W-FRI").astype(str)
bucket = no[(no["z_drift_6h"] < 0.55).fillna(False)].copy().reset_index(drop=True)
print(f"bucket: n={len(bucket)}  edge={bucket['tedge'].mean():+.4f}  $={bucket['would_pnl'].sum():+.2f}")
streak_mask_all = bucket["contract_ticker"].isin(STREAK)


def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean(), (means >= 0).mean()


def report(name, mask, n_tests_ctx=""):
    d = bucket[mask.fillna(False)]
    if len(d) < 25:
        print(f"  {name}: n={len(d)} (thin)")
        return
    ne, ee, p_neg, p_pos = ep_stats(d)
    wk = d.groupby("week")["tedge"].mean()
    sh = int(d["contract_ticker"].isin(STREAK).sum())
    print(f"  {name}: n={len(d)} eps={ne} edge={d['tedge'].mean():+.4f} ep_edge={ee:+.4f} "
          f"P(<=0)={p_neg:.4f} P(>=0)={p_pos:.4f} wk+={int((wk>0).sum())}/{len(wk)} "
          f"streak={sh}/4 $={d['would_pnl'].sum():+.2f}")


# ── 1. z_drift dynamics from the scan archive series ──────────────────────
print("\n=== 1. z_drift dynamics ===")
sa = pd.read_csv("results/sol_scan_archive_15m.csv", low_memory=False, usecols=["logged_at", "z_drift_6h"])
sa["z_drift_6h"] = pd.to_numeric(sa["z_drift_6h"], errors="coerce")
def parse_mixed(s):
    def _u(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_u(v) for v in s], utc=True)
sa["ts"] = parse_mixed(sa["logged_at"])
zs = sa.dropna(subset=["ts", "z_drift_6h"]).drop_duplicates(subset="ts").sort_values("ts")[["ts", "z_drift_6h"]]
print(f"scan-archive z_drift series: {len(zs)} points {zs['ts'].min()} -> {zs['ts'].max()}")

zs_idx = zs.set_index("ts")["z_drift_6h"]
def z_delta_1h(ts):
    # value ~1h before trade (causal: strictly earlier scans)
    past = zs_idx[zs_idx.index <= ts - pd.Timedelta("55min")]
    if not len(past):
        return np.nan
    return None if pd.isna(past.iloc[-1]) else past.iloc[-1]

bucket["z_1h_ago"] = bucket["logged_at_p"].apply(z_delta_1h).astype(float)
bucket["z_delta_1h"] = bucket["z_drift_6h"] - bucket["z_1h_ago"]

def time_below(ts):
    # minutes z_drift has been continuously < 0.55 before this trade
    hist = zs[(zs["ts"] <= ts) & (zs["ts"] >= ts - pd.Timedelta("24h"))]
    if len(hist) < 3:
        return np.nan
    above = hist[hist["z_drift_6h"] >= 0.55]
    if not len(above):
        return 24 * 60.0
    return (ts - above["ts"].iloc[-1]).total_seconds() / 60.0
bucket["mins_below"] = bucket["logged_at_p"].apply(time_below)

print(f"coverage: z_delta_1h {bucket['z_delta_1h'].notna().sum()}/{len(bucket)}, "
      f"mins_below {bucket['mins_below'].notna().sum()}/{len(bucket)}")
report("z rising (delta>0)", bucket["z_delta_1h"] > 0)
report("z falling (delta<=0)", bucket["z_delta_1h"] <= 0)
report("fresh low-drift (<60min below)", bucket["mins_below"] < 60)
report("stale low-drift (>=240min below)", bucket["mins_below"] >= 240)
report("mid low-drift (60-240min)", (bucket["mins_below"] >= 60) & (bucket["mins_below"] < 240))

# ── 2. momentum-absence conjunction ───────────────────────────────────────
print("\n=== 2. multi-TF momentum conjunctions ===")
report("chg_5m<=0 AND chg_15m<=0 (no live momentum)", (bucket["chg_5m"] <= 0) & (bucket["chg_15m"] <= 0))
report("chg_5m>0 AND chg_15m>0 (danger: live up-momentum)", (bucket["chg_5m"] > 0) & (bucket["chg_15m"] > 0))
report("chg_15m<=0 alone", bucket["chg_15m"] <= 0)
report("chg_5m<=0 AND chg_15m<=0 AND chg_1h<=0 (3-TF absent)",
       (bucket["chg_5m"] <= 0) & (bucket["chg_15m"] <= 0) & (bucket["chg_1h"] <= 0))

# ── 3. cross-asset BTC state ─────────────────────────────────────────────
print("\n=== 3. cross-asset BTC momentum at SOL trade time ===")
pbtc = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
b1m = pd.read_parquet(pbtc).sort_index()
b1m = b1m[b1m.index >= "2026-04-15"]
print(f"BTC 1m: ends {b1m.index.max()}")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
b15 = b1m.resample("15min").agg(AGG).dropna()
b60 = b1m.resample("1h").agg(AGG).dropna()

def btc_feats(ts):
    out = {}
    for nm, df, fm in [("b15", b15, 15), ("b60", b60, 60)]:
        cutoff = ts - pd.Timedelta(minutes=fm)
        i = df.index.searchsorted(cutoff, side="right") - 1
        if i < 20:
            out[f"chg_{nm}"] = np.nan
            out[f"sto_{nm}"] = np.nan
            continue
        b = df.iloc[max(0, i - 30):i + 1]
        out[f"chg_{nm}"] = float((b["close"].iloc[-1] / b["close"].iloc[-2] - 1) * 100)
        lo14, hi14 = b["low"].rolling(14).min(), b["high"].rolling(14).max()
        out[f"sto_{nm}"] = float((((b["close"] - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100).iloc[-1])
    return out

bf = bucket["logged_at_p"].apply(btc_feats)
for k in ["chg_b15", "sto_b15", "chg_b60", "sto_b60"]:
    bucket[k] = bf.apply(lambda d: d[k])
print(f"coverage: {bucket['chg_b15'].notna().sum()}/{len(bucket)}")
report("BTC 15m chg<=0 (BTC not rising)", bucket["chg_b15"] <= 0)
report("BTC 15m chg>0 (BTC rising too)", bucket["chg_b15"] > 0)
report("BTC stoch15<50", bucket["sto_b15"] < 50)
report("BTC 1h chg<=0", bucket["chg_b60"] <= 0)
report("BTC 15m AND SOL 15m both non-rising", (bucket["chg_b15"] <= 0) & (bucket["chg_15m"] <= 0))

# ── 4. ATR-normalized strike distance ────────────────────────────────────
print("\n=== 4. ATR-normalized strike distance ===")
bucket["atr_ratio_15m"] = pd.to_numeric(bucket["atr_ratio_15m"], errors="coerce")
bucket["offset_atr"] = bucket["offset_pct"].abs() / (bucket["atr_ratio_15m"] * 100).replace(0, np.nan)
cov = bucket["offset_atr"].notna().sum()
print(f"coverage: {cov}/{len(bucket)}")
if cov > 200:
    for q in [0.3, 0.5, 0.7]:
        th = bucket["offset_atr"].quantile(q)
        report(f"offset_atr >= {th:.2f} (far strike, q{q})", bucket["offset_atr"] >= th)
        report(f"offset_atr < {th:.2f} (near strike, q{q})", bucket["offset_atr"] < th)

# ── 5. episode position ──────────────────────────────────────────────────
print("\n=== 5. episode position ===")
bucket["ep_pos"] = bucket.groupby("episode").cumcount() + 1
report("1st trade of episode", bucket["ep_pos"] == 1)
report("2nd+ trade of episode", bucket["ep_pos"] >= 2)
report("3rd+ trade of episode", bucket["ep_pos"] >= 3)

bucket.to_csv(f"{OUT}/bucket_novel_features.csv", index=False)
print("\nDONE_S3")
