"""
analyze_price_movement_corr.py

Finds indicators correlated with price movement MAGNITUDE at expiry.
Uses spot_at_expiry / price_move_pct / miss_pct backfilled by backfill_expiry_prices.py.

Key targets:
  price_move_pct : (spot_exp - spot_scan) / spot_scan * 100   — raw price drift
  miss_pct       : (spot_exp - strike) / strike * 100          — how far above/below strike
  abs_miss_pct   : |miss_pct|                                  — conviction proxy

Separate analysis for:
  - Each asset (BTC, ETH, SOL)
  - 1h vs 15m timeframe
  - YES trades vs NO trades vs all trades
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS = Path("results")

SEP  = "=" * 72
SEP2 = "-" * 56

# ── 1h paper trade files ────────────────────────────────────────────────────
H1_FILES = {
    "BTC": RESULTS / "paper_trades.csv",
    "ETH": RESULTS / "paper_trades_eth.csv",
    "SOL": RESULTS / "paper_trades_sol.csv",
}

# ── 15m paper trade files ───────────────────────────────────────────────────
M15_FILES = {
    "BTC": RESULTS / "paper_trades_btc15m.csv",
    "ETH": RESULTS / "paper_trades_eth15m.csv",
    "SOL": RESULTS / "paper_trades_sol15m.csv",
}

# ── Feature columns for each timeframe ─────────────────────────────────────
H1_FEATURES = [
    "offset_pct", "composite_p_up", "composite_trend", "composite_rev",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score", "vwap_distance_pct",
    "stoch_k", "chg_30m", "chg_10m", "chg_5m", "bp_5m", "body_15m", "dir_15m",
    "vol_score", "vpin_score", "obi_score", "confirmation_score", "no_score",
    "funding_bias", "vol_eff", "pm_drift_5m", "adx_1h", "rvol_1h", "squeeze_1h",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
    "p_market", "p_yes_model", "z_score", "net_edge", "tau_minutes",
]

M15_FEATURES = [
    "offset_pct", "composite_p_up", "bp_5m", "vol_ratio", "vol_ratio_5m",
    "body_15m", "bp_15m", "dir_15m", "upper_wick_15m", "lower_wick_15m",
    "atr_ratio_15m", "range_ratio_15m", "consec_dir_15m",
    "stoch_k_5m", "stoch_k_15m", "chg_1m", "chg_5m", "chg_15m",
    "vwap_dist", "ema_bias", "ema_bias_1h", "nearest_res_dist_pct",
    "realized_vol_annual", "vol_ratio_1h",
    "bp_1h", "chg_1h", "dir_1h", "consec_dir_1h",
    "stoch_k_1h", "stoch_cross_1h", "rsi_1h", "macd_hist_1h",
    "donchian_breakout_1h", "engulfing_1h",
    "liq_score", "liq_bias", "oi_chg_pct", "ls_long_pct",
    "fear_greed", "cg_composite", "p_market", "tau_minutes", "spread",
]

MIN_N = 30  # minimum observations for a valid correlation


def load_df(path: Path, features: list) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)

    # Need resolved rows with expiry price
    df = df[df["price_move_pct"].notna() & df["price_move_pct"].astype(str).str.strip().ne("")].copy()
    if df.empty:
        return df

    df["price_move_pct"] = pd.to_numeric(df["price_move_pct"], errors="coerce")
    df["miss_pct"]       = pd.to_numeric(df.get("miss_pct", pd.Series(dtype=float)), errors="coerce")
    df["resolved_yes"]   = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df["abs_miss_pct"]   = df["miss_pct"].abs()

    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # side / decision
    if "side" in df.columns:
        df["side"] = df["side"].str.strip().str.lower()
    if "decision" in df.columns:
        df["decision"] = df["decision"].str.strip().str.lower()

    return df


def corr_table(df: pd.DataFrame, features: list, target: str, min_n: int = MIN_N):
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        sub = df[[feat, target]].dropna()
        n = len(sub)
        if n < min_n:
            continue
        r, p = stats.pearsonr(sub[feat], sub[target])
        rs, ps = stats.spearmanr(sub[feat], sub[target])
        rows.append({
            "feature": feat,
            "n":       n,
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 4),
            "spearman_r": round(rs, 4),
            "spearman_p": round(ps, 4),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False)
    return out


def print_corr(ct: pd.DataFrame, label: str, n_show: int = 20):
    if ct.empty:
        print(f"  {label}: no data")
        return
    sig = ct[ct["spearman_p"] < 0.05]
    print(f"\n  {label}  (n rows in analysis = {ct['n'].max()}, "
          f"{len(sig)}/{len(ct)} significant at p<0.05)")
    print(f"  {'Feature':<28} {'Spear-r':>8} {'p':>7}  {'Pears-r':>8} {'p':>7}")
    print("  " + "-" * 60)
    for _, row in ct.head(n_show).iterrows():
        sig_flag = "*" if row["spearman_p"] < 0.05 else " "
        print(f"  {row['feature']:<28} {row['spearman_r']:>+8.4f} {row['spearman_p']:>7.4f}"
              f"  {row['pearson_r']:>+8.4f} {row['pearson_p']:>7.4f}  {sig_flag}")


def analyze(label: str, df: pd.DataFrame, features: list):
    if df.empty or len(df) < MIN_N:
        print(f"\n{label}: insufficient data ({len(df)} rows)")
        return

    print(f"\n{SEP}")
    print(f"  {label}  ({len(df)} rows total)")
    print(SEP)

    # ── price_move_pct (raw directional drift) ──────────────────────────────
    print(f"\n  TARGET: price_move_pct  "
          f"(mean={df['price_move_pct'].mean():+.4f}%  "
          f"std={df['price_move_pct'].std():.4f}%)")
    ct = corr_table(df, features, "price_move_pct")
    print_corr(ct, "ALL trades vs price_move_pct")

    if "side" in df.columns:
        for side in ("yes", "no"):
            sub = df[df["side"] == side]
            if len(sub) >= MIN_N:
                ct = corr_table(sub, features, "price_move_pct")
                print_corr(ct, f"{side.upper()} trades vs price_move_pct")

    # ── miss_pct (distance from strike at expiry) ───────────────────────────
    if df["miss_pct"].notna().sum() >= MIN_N:
        print(f"\n  TARGET: miss_pct  "
              f"(mean={df['miss_pct'].mean():+.4f}%  "
              f"std={df['miss_pct'].std():.4f}%)")
        ct = corr_table(df, features, "miss_pct")
        print_corr(ct, "ALL trades vs miss_pct")

    # ── abs_miss_pct (strength / conviction proxy) ──────────────────────────
    if df["abs_miss_pct"].notna().sum() >= MIN_N:
        print(f"\n  TARGET: abs_miss_pct  (conviction proxy — larger = resolved more decisively)")
        ct = corr_table(df, features, "abs_miss_pct")
        print_corr(ct, "ALL trades vs abs_miss_pct")

    # ── Conditional means: top vs bottom quartile of miss_pct ───────────────
    print(f"\n{SEP2}")
    print("  Top/bottom quartile split (miss_pct):")
    if df["miss_pct"].notna().sum() >= 40:
        q25 = df["miss_pct"].quantile(0.25)
        q75 = df["miss_pct"].quantile(0.75)
        hi  = df[df["miss_pct"] >= q75]
        lo  = df[df["miss_pct"] <= q25]
        print(f"  Bottom 25% miss_pct ≤ {q25:+.4f}%  (n={len(lo)}, "
              f"resolved_yes={lo['resolved_yes'].mean():.2f}  "
              f"price_move={lo['price_move_pct'].mean():+.4f}%)")
        print(f"  Top    25% miss_pct ≥ {q75:+.4f}%  (n={len(hi)}, "
              f"resolved_yes={hi['resolved_yes'].mean():.2f}  "
              f"price_move={hi['price_move_pct'].mean():+.4f}%)")
        # Feature means in top vs bottom quartile
        feat_diff = []
        for feat in features:
            if feat not in df.columns:
                continue
            hi_m = hi[feat].dropna().mean()
            lo_m = lo[feat].dropna().mean()
            if pd.notna(hi_m) and pd.notna(lo_m) and hi_m != lo_m:
                feat_diff.append((feat, hi_m - lo_m, hi_m, lo_m))
        feat_diff.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Features with largest difference (top vs bottom miss_pct quartile):")
        print(f"  {'Feature':<28} {'Hi_mean':>9} {'Lo_mean':>9} {'Diff':>9}")
        print("  " + "-" * 58)
        for feat, diff, hi_m, lo_m in feat_diff[:15]:
            print(f"  {feat:<28} {hi_m:>+9.4f} {lo_m:>+9.4f} {diff:>+9.4f}")


def main():
    print("\n" + SEP)
    print("  PRICE MOVEMENT CORRELATION ANALYSIS")
    print("  Targets: price_move_pct, miss_pct, abs_miss_pct")
    print(SEP)

    # ── 1h timeframe ────────────────────────────────────────────────────────
    for asset, path in H1_FILES.items():
        df = load_df(path, H1_FEATURES)
        if df.empty:
            print(f"\n1h {asset}: no data with expiry prices yet")
            continue
        analyze(f"1h {asset} — {path.name}", df, H1_FEATURES)

    # ── 15m timeframe ───────────────────────────────────────────────────────
    for asset, path in M15_FILES.items():
        df = load_df(path, M15_FEATURES)
        if df.empty:
            print(f"\n15m {asset}: no data with expiry prices yet")
            continue
        analyze(f"15m {asset} — {path.name}", df, M15_FEATURES)

    # ── Cross-asset combined (1h only) ───────────────────────────────────────
    frames = []
    for asset, path in H1_FILES.items():
        df = load_df(path, H1_FEATURES)
        if not df.empty:
            df["asset"] = asset
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        shared = [f for f in H1_FEATURES if f in combined.columns]
        analyze("1h COMBINED (BTC+ETH+SOL)", combined, shared)


if __name__ == "__main__":
    main()
