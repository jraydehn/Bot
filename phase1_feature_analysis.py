"""
phase1_feature_analysis.py — Feature correlation analysis against real Kalshi outcomes.

Target: market_miss = resolved_yes - p_market
  Positive → market underpriced YES (YES was a better bet than market said)
  Negative → market overpriced YES (NO was a better bet)

For each feature we compute:
  - Spearman correlation with market_miss (full dataset + YES-side + NO-side)
  - Win-rate by quintile to spot non-linearities

Includes both logged signals and backfilled price-based features.
"""
import glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"


# ── load trades ───────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES_DIR / "paper_trades_archive_*.csv")))
    files += [str(RES_DIR / "paper_trades.csv")]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["logged_at"] = pd.to_datetime(combined["logged_at"], format="mixed", utc=True)
    combined = (combined
                .sort_values("logged_at")
                .drop_duplicates(subset=["contract_ticker", "logged_at", "side"], keep="last"))

    trades = combined[combined["decision"] == "trade"].copy()
    trades = trades[trades["resolved_yes"].notna()].copy()

    for col in ["p_market", "resolved_yes", "spot", "strike", "tau_minutes", "vol_eff"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(subset=["p_market", "resolved_yes"])

    # Merge backfilled signals
    pup_path = RES_DIR / "p_up_v2_backfilled.csv"
    if pup_path.exists():
        pup = pd.read_csv(pup_path)
        pup["logged_at"] = pd.to_datetime(pup["logged_at"], utc=True)
        trades = trades.merge(pup, on=["contract_ticker", "logged_at", "side"], how="left")

    trades = trades.reset_index(drop=True)

    # Core target: how much did the market miss?
    trades["market_miss"] = trades["resolved_yes"] - trades["p_market"]
    # Binary win per side
    trades["won"] = np.where(
        trades["side"] == "yes",
        trades["resolved_yes"] == 1,
        trades["resolved_yes"] == 0,
    )
    if "vol_implied_kalshi" in trades.columns:
        vi = pd.to_numeric(trades["vol_implied_kalshi"], errors="coerce")
        trades["vol_ratio_realized"] = trades["vol_eff"] / vi.replace(0, float("nan"))
    else:
        trades["vol_ratio_realized"] = float("nan")

    return trades


# ── feature catalogue ─────────────────────────────────────────────────────────

FEATURES = {
    # Directional / momentum
    "p_up_v2_backfilled":  "p_up_v2 (LGBM directional)",
    "composite_p_up":      "composite_p_up (lookup table)",
    "stoch_k":             "Stoch K (1h)",
    "stoch_k_4h":          "Stoch K (4h) [backfilled]",       # from backfill
    "stoch_bias":          "Stoch bias score",
    "ema_stack_bias":      "EMA stack bias",
    "ema_stretch_score":   "EMA stretch score",
    "rsi_14":              "RSI 14 (1h) [backfilled]",
    "rsi_4h":              "RSI 14 (4h) [backfilled]",
    "macd_hist_1h":        "MACD hist (1h) [backfilled]",
    "chg_5m":              "Price chg 5m",
    "chg_10m":             "Price chg 10m",
    "chg_30m":             "Price chg 30m",
    "pm_drift_5m":         "p_market drift 5m",
    "chg_4h_atr":          "4h chg / ATR [backfilled]",
    # Trend / structure
    "composite_trend":     "Composite trend score",
    "composite_rev":       "Composite reversion score",
    "ema50_dist":          "EMA50 distance % [backfilled]",
    "bb_pct":              "Bollinger Band %B [backfilled]",
    "vwap_distance_pct":   "VWAP distance % [backfilled]",
    "vwap_score":          "VWAP score (logged)",
    "adx_1h":              "ADX (1h)",
    "smc_1h":              "SMC bias (1h)",
    "smc_4h":              "SMC bias (4h)",
    "markov_regime_daily": "Markov regime (daily)",
    # Volume / liquidity
    "vpin_score":          "VPIN score",
    "vol_score":           "Vol score",
    "vol_ratio":           "Vol ratio (model/implied)",
    "rvol_inv_backfilled": "rvol_inv (168h/24h) [backfilled]",
    "vol_ratio_realized":  "vol_eff / vol_implied_kalshi",
    "ob_imbalance":        "Order book imbalance",
    "liq_score":           "Liquidity score",
    "liq_bias":            "Liquidity bias",
    "oi_chg_pct":          "OI change %",
    "ls_long_pct":         "Long/short long %",
    "funding_bias":        "Funding rate bias",
    "avg_funding_rate":    "Avg funding rate",
    "obi_score":           "OBI score",
    # Confirmation
    "confirmation_bias":   "Confirmation bias",
    "confirmation_score":  "Confirmation score",
    # Market price context
    "p_market":            "p_market (baseline check)",
    "tau_minutes":         "Tau (minutes)",
    # Donchian channels — 15m
    "dc_15m_n20_pos":      "Donchian pos  15m N=20",
    "dc_15m_n20_break":    "Donchian break 15m N=20",
    "dc_15m_n20_width":    "Donchian width 15m N=20",
    "dc_15m_n55_pos":      "Donchian pos  15m N=55",
    "dc_15m_n55_break":    "Donchian break 15m N=55",
    "dc_15m_n55_width":    "Donchian width 15m N=55",
    # Donchian channels — 1h
    "dc_1h_n20_pos":       "Donchian pos  1h N=20",
    "dc_1h_n20_break":     "Donchian break 1h N=20",
    "dc_1h_n20_width":     "Donchian width 1h N=20",
    "dc_1h_n55_pos":       "Donchian pos  1h N=55",
    "dc_1h_n55_break":     "Donchian break 1h N=55",
    "dc_1h_n55_width":     "Donchian width 1h N=55",
    # Donchian channels — 4h
    "dc_4h_n20_pos":       "Donchian pos  4h N=20",
    "dc_4h_n20_break":     "Donchian break 4h N=20",
    "dc_4h_n20_width":     "Donchian width 4h N=20",
    "dc_4h_n55_pos":       "Donchian pos  4h N=55",
    "dc_4h_n55_break":     "Donchian break 4h N=55",
    "dc_4h_n55_width":     "Donchian width 4h N=55",
    # Donchian channels — 1d
    "dc_1d_n20_pos":       "Donchian pos  1d N=20",
    "dc_1d_n20_break":     "Donchian break 1d N=20",
    "dc_1d_n20_width":     "Donchian width 1d N=20",
    "dc_1d_n55_pos":       "Donchian pos  1d N=55",
    "dc_1d_n55_break":     "Donchian break 1d N=55",
    "dc_1d_n55_width":     "Donchian width 1d N=55",
}


# ── analysis ──────────────────────────────────────────────────────────────────

def spearman(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 20:
        return float("nan"), float("nan"), 0
    r, p = stats.spearmanr(a[mask], b[mask])
    return r, p, mask.sum()


def analyze(trades: pd.DataFrame) -> pd.DataFrame:
    yes_trades = trades[trades["side"] == "yes"]
    no_trades  = trades[trades["side"] == "no"]

    rows = []
    for feat, label in FEATURES.items():
        # Backfilled price features need special handling
        if feat in ("stoch_k_4h", "rsi_14", "rsi_4h", "macd_hist_1h",
                    "chg_4h_atr", "ema50_dist", "bb_pct", "vwap_distance_pct"):
            # These are in the backfill CSV joined via p_up_v2 merge
            if feat not in trades.columns:
                continue

        if feat not in trades.columns:
            continue

        vals = pd.to_numeric(trades[feat], errors="coerce").values
        miss = trades["market_miss"].values

        r_all, p_all, n_all = spearman(vals, miss)
        if n_all < 20:
            continue

        # YES side: positive market_miss = good (YES resolved when underpriced)
        vals_y = pd.to_numeric(yes_trades[feat], errors="coerce").values
        r_yes, _, n_yes = spearman(vals_y, yes_trades["market_miss"].values)

        # NO side: negative market_miss = good (YES didn't resolve when overpriced)
        vals_n = pd.to_numeric(no_trades[feat], errors="coerce").values
        r_no, _, n_no = spearman(vals_n, no_trades["market_miss"].values)

        rows.append({
            "feature":    feat,
            "label":      label,
            "r_all":      r_all,
            "p_all":      p_all,
            "n_all":      n_all,
            "r_yes":      r_yes,
            "n_yes":      n_yes,
            "r_no":       r_no,
            "n_no":       n_no,
            "abs_r":      abs(r_all),
        })

    df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    return df


def quintile_breakdown(trades: pd.DataFrame, feat: str, n_bins: int = 5):
    vals = pd.to_numeric(trades[feat], errors="coerce")
    mask = vals.notna()
    sub  = trades[mask].copy()
    sub[feat] = vals[mask].values
    sub["_q"] = pd.qcut(vals[mask], n_bins, labels=False, duplicates="drop")

    rows = []
    for q, g in sub.groupby("_q"):
        rows.append({
            "quintile":    int(q) + 1,
            "feat_range":  f"{g[feat].min():.3g}–{g[feat].max():.3g}",
            "n":           len(g),
            "wr":          g["won"].mean(),
            "mean_miss":   g["market_miss"].mean(),
            "pnl_proxy":   g["market_miss"].sum(),
        })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading trades...")
    trades = load_trades()
    print(f"  {len(trades):,} resolved trades  "
          f"({trades['logged_at'].min().date()} → {trades['logged_at'].max().date()})")
    print(f"  YES: {(trades['side']=='yes').sum():,}   NO: {(trades['side']=='no').sum():,}")
    print(f"  Overall WR: {trades['won'].mean():.1%}   "
          f"Mean market_miss: {trades['market_miss'].mean():+.4f}\n")

    print("Running feature correlations with market_miss (resolved_yes - p_market)...")
    results = analyze(trades)

    print(f"\n{'='*80}")
    print(f"  FEATURE → MARKET_MISS SPEARMAN CORRELATION  (sorted by |r| overall)")
    print(f"  Positive r_all = feature predicts YES over-resolution (market underpriced YES)")
    print(f"  Negative r_all = feature predicts YES under-resolution (market overpriced YES)")
    print(f"{'='*80}")
    print(f"  {'Feature':<28} {'r_all':>7} {'p':>7} {'n':>5}  {'r_yes':>7} {'r_no':>7}")
    print(f"  {'-'*70}")

    sig_threshold = 0.05
    for _, row in results.iterrows():
        sig = "*" if row["p_all"] < sig_threshold else " "
        print(f"  {row['feature']:<28} {row['r_all']:>+7.3f}{sig} {row['p_all']:>7.4f} "
              f"{row['n_all']:>5,}  {row['r_yes']:>+7.3f} {row['r_no']:>+7.3f}  "
              f"  {row['label']}")

    # Quintile breakdown for top features by |r|
    top_feats = results[results["abs_r"] > 0.04].head(10)["feature"].tolist()

    print(f"\n{'='*80}")
    print(f"  QUINTILE BREAKDOWN — top features")
    print(f"  (WR = actual win rate of trades taken; mean_miss = how much market was off)")
    print(f"{'='*80}")

    for feat in top_feats:
        if feat not in trades.columns:
            continue
        vals = pd.to_numeric(trades[feat], errors="coerce")
        if vals.notna().sum() < 50:
            continue
        qdf = quintile_breakdown(trades, feat)
        label = FEATURES.get(feat, feat)
        print(f"\n  {feat}  —  {label}")
        print(f"  {'Q':<4} {'Range':<18} {'N':>5} {'WR':>7} {'mean_miss':>10} {'sum_miss':>10}")
        for _, r in qdf.iterrows():
            print(f"  Q{int(r['quintile']):<3} {r['feat_range']:<18} {r['n']:>5,} "
                  f"{r['wr']:>7.1%} {r['mean_miss']:>+10.4f} {r['pnl_proxy']:>+10.2f}")

    # Save full results
    out = RES_DIR / "phase1_feature_correlations.csv"
    results.to_csv(out, index=False)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    run()
