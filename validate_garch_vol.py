"""
validate_garch_vol.py — Validate GARCH(1,1) conditional vol as a vol_layer signal.

Two-part analysis:
  Part 1: Signal validity — does GARCH ratio predict next-hour |return| better than
          existing signals? Uses the same '>0.5% move' metric as the original vol_layer.
  Part 2: Trade outcomes — does GARCH regime correlate with win rate in btc_scan_archive,
          and does it add information beyond the existing vol_score column?

Usage: python3 validate_garch_vol.py [--asset BTC|ETH|SOL]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

ASSET_CONFIG = {
    "BTC": {
        "parquet_1h": "data/binanceus_BTCUSDT_1h_1970-01-01_2026-05-24.parquet",
        "scan_archive": "results/btc_scan_archive.csv",
        "big_move_thr": 0.005,   # 0.5% — same threshold used in original vol_layer validation
    },
    "ETH": {
        "parquet_1h": "data/binanceus_ETHUSDT_1h_2024-01-01_2026-05-24.parquet",
        "scan_archive": "results/eth_scan_archive.csv",
        "big_move_thr": 0.007,
    },
    "SOL": {
        "parquet_1h": "data/binanceus_SOLUSDT_1h_2024-01-01_2026-05-24.parquet",
        "scan_archive": "results/sol_scan_archive.csv",
        "big_move_thr": 0.010,
    },
}

GARCH_FIT_WINDOW  = 500    # bars used to fit each rolling GARCH model
GARCH_RATIO_HI    = 1.50   # proposed high-vol threshold
GARCH_RATIO_LO    = 0.67   # proposed low-vol threshold


# ── Part 1: Rolling GARCH on 1h data ─────────────────────────────────────────

def compute_rolling_garch(close: pd.Series, fit_window: int = 500) -> pd.DataFrame:
    """Fit GARCH(1,1) expanding window on 1h log returns.

    For each bar t (starting at fit_window), fits on bars [t-fit_window:t],
    then records the 1-step-ahead conditional vol and the long-run vol.

    Returns DataFrame with columns:
        cond_vol, lr_vol, ratio, persistence, next_abs_ret
    """
    log_ret = np.log(close / close.shift(1)).dropna() * 100  # in %
    results = []

    print(f"  Fitting rolling GARCH(1,1) on {len(log_ret)} bars "
          f"(window={fit_window}, {len(log_ret)-fit_window} iterations)...")

    for i in range(fit_window, len(log_ret)):
        window = log_ret.iloc[i - fit_window:i]
        try:
            am  = arch_model(window, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            cond_vol    = float(res.conditional_volatility.iloc[-1])
            omega       = float(res.params["omega"])
            alpha       = float(res.params["alpha[1]"])
            beta        = float(res.params["beta[1]"])
            persistence = alpha + beta
            lr_vol      = (float(np.sqrt(omega / (1.0 - persistence)))
                           if persistence < 1.0 else float(window.std()))
            ratio = cond_vol / lr_vol if lr_vol > 0 else 1.0
        except Exception:
            cond_vol = lr_vol = ratio = persistence = np.nan

        # Next-bar absolute log return (what we're trying to predict)
        next_ret = (float(log_ret.iloc[i]) / 100) if i < len(log_ret) else np.nan

        results.append({
            "timestamp":   log_ret.index[i],
            "cond_vol":    cond_vol,
            "lr_vol":      lr_vol,
            "ratio":       ratio,
            "persistence": persistence,
            "next_abs_ret": abs(next_ret) if not np.isnan(next_ret) else np.nan,
            "next_big_move": (abs(next_ret) > 0) if not np.isnan(next_ret) else np.nan,
        })

        if (i - fit_window) % 200 == 0:
            pct = (i - fit_window) / (len(log_ret) - fit_window) * 100
            print(f"    {pct:.0f}% complete...", end="\r")

    print()
    return pd.DataFrame(results).set_index("timestamp")


def section1_signal_validity(df_garch: pd.DataFrame, big_move_thr: float, asset: str):
    print("\n" + "="*70)
    print(f"PART 1 — SIGNAL VALIDITY ({asset})")
    print("="*70)
    print(f"Metric: frequency of |next 1h return| > {big_move_thr*100:.1f}%")
    print(f"Baseline (all bars): {df_garch['next_big_move'].mean()*100:.1f}%")
    print()

    hi  = df_garch[df_garch["ratio"] > GARCH_RATIO_HI]
    mid = df_garch[(df_garch["ratio"] >= GARCH_RATIO_LO) & (df_garch["ratio"] <= GARCH_RATIO_HI)]
    lo  = df_garch[df_garch["ratio"] < GARCH_RATIO_LO]

    # Update big_move using the threshold (recalculate since it wasn't asset-specific above)
    def big_move_freq(subset):
        bm = (subset["next_abs_ret"] > big_move_thr).mean()
        return bm

    print(f"{'Regime':<12} {'n':>6} {'Pct':>6} {'BigMove%':>10} {'AvgAbsRet%':>12} {'AvgRatio':>10}")
    print("-" * 62)
    for label, subset in [("HIGH >1.5", hi), ("MID 0.67–1.5", mid), ("LOW <0.67", lo)]:
        if len(subset) == 0:
            print(f"  {label:<12} n=0")
            continue
        bm  = big_move_freq(subset)
        avg = subset["next_abs_ret"].mean() * 100
        ratio_avg = subset["ratio"].mean()
        pct_of_total = len(subset) / len(df_garch) * 100
        print(f"  {label:<14} {len(subset):>5}  {pct_of_total:>5.1f}%  {bm*100:>8.1f}%  {avg:>10.3f}%  {ratio_avg:>8.3f}")

    print()
    # Spearman rank correlation between ratio and |next return|
    from scipy import stats
    valid = df_garch.dropna(subset=["ratio", "next_abs_ret"])
    corr, pval = stats.spearmanr(valid["ratio"], valid["next_abs_ret"])
    print(f"  Spearman ρ(ratio, |next_ret|) = {corr:.4f}  (p={pval:.4f})")

    # Persistence distribution
    print(f"\n  Persistence (α+β) stats:")
    p = df_garch["persistence"].dropna()
    print(f"    mean={p.mean():.3f}  median={p.median():.3f}  "
          f"min={p.min():.3f}  max={p.max():.3f}")
    print(f"    Bars with persistence >= 0.95 (near-unit-root): "
          f"{(p >= 0.95).mean()*100:.1f}%")


# ── Part 2: Trade outcomes vs GARCH regime ────────────────────────────────────

def section2_trade_outcomes(df_garch: pd.DataFrame, scan_csv: str, asset: str):
    print("\n" + "="*70)
    print(f"PART 2 — TRADE OUTCOMES BY GARCH REGIME ({asset})")
    print("="*70)

    if not Path(scan_csv).exists():
        print(f"  Scan archive not found: {scan_csv}")
        return

    arc = pd.read_csv(scan_csv)

    # Only resolved trades with a side
    # Determine outcome column
    if "resolved_yes" in arc.columns:
        arc = arc.dropna(subset=["resolved_yes"])
        arc["win"] = arc.apply(
            lambda r: (r["resolved_yes"] == 1) if r.get("side", "yes") == "yes"
                      else (r["resolved_yes"] == 0), axis=1
        ) if "side" in arc.columns else (arc["resolved_yes"] == 1)
    else:
        print("  No outcome column found in scan archive.")
        return

    # Parse timestamps and merge GARCH regime
    arc["ts"] = pd.to_datetime(arc["logged_at"], utc=True).dt.floor("h")
    df_garch_reset = df_garch.reset_index()
    df_garch_reset["timestamp"] = pd.to_datetime(df_garch_reset["timestamp"], utc=True).dt.floor("h")

    merged = arc.merge(
        df_garch_reset[["timestamp", "ratio", "cond_vol", "lr_vol", "persistence"]],
        left_on="ts", right_on="timestamp", how="left"
    )
    matched = merged.dropna(subset=["ratio"])
    print(f"  Trades in archive: {len(arc)}, matched to GARCH ratio: {len(matched)} ({len(matched)/len(arc)*100:.0f}%)")

    if len(matched) == 0:
        print("  No matches — date ranges may not overlap.")
        return

    # Bucket by GARCH regime
    matched["garch_regime"] = pd.cut(
        matched["ratio"],
        bins=[-np.inf, GARCH_RATIO_LO, GARCH_RATIO_HI, np.inf],
        labels=["LOW <0.67", "MID 0.67–1.5", "HIGH >1.5"]
    )

    def pnl_est(subset):
        """Rough P&L estimate using p_market as payout proxy."""
        wins = subset[subset["win"] == True]
        losses = subset[subset["win"] == False]
        avg_pm = subset["p_market"].mean()
        win_payout  = (1 - avg_pm)  # approx per-dollar
        loss_payout = avg_pm
        return wins["win"].count() * win_payout - losses["win"].count() * loss_payout

    print(f"\n  {'Regime':<16} {'n':>5} {'WR%':>7} {'AvgPM':>7}  VolScore breakdown")
    print("  " + "-"*65)
    for regime in ["HIGH >1.5", "MID 0.67–1.5", "LOW <0.67"]:
        sub = matched[matched["garch_regime"] == regime]
        if len(sub) == 0:
            continue
        wr  = sub["win"].mean() * 100
        apm = sub["p_market"].mean()
        print(f"  {regime:<16} {len(sub):>5} {wr:>6.1f}% {apm:>7.3f}", end="")
        if "vol_score" in sub.columns:
            vs_dist = sub["vol_score"].value_counts().sort_index().to_dict()
            print(f"  vol_score dist: {vs_dist}", end="")
        print()

    # Key test: does GARCH add info beyond vol_score?
    if "vol_score" in matched.columns:
        print(f"\n  INCREMENTAL VALUE vs vol_score:")
        print(f"  (Does GARCH regime shift WR within the same vol_score bucket?)")
        for vs in sorted(matched["vol_score"].dropna().unique()):
            sub_vs = matched[matched["vol_score"] == vs]
            if len(sub_vs) < 20:
                continue
            wr_all = sub_vs["win"].mean() * 100
            sub_hi = sub_vs[sub_vs["ratio"] > GARCH_RATIO_HI]
            sub_lo = sub_vs[sub_vs["ratio"] < GARCH_RATIO_LO]
            sub_md = sub_vs[(sub_vs["ratio"] >= GARCH_RATIO_LO) & (sub_vs["ratio"] <= GARCH_RATIO_HI)]
            print(f"  vol_score={vs:+.0f}  all n={len(sub_vs)} WR={wr_all:.1f}%", end="")
            for label, sub in [("  HIGH", sub_hi), ("  MID", sub_md), ("  LOW", sub_lo)]:
                if len(sub) >= 8:
                    print(f"  {label} n={len(sub)} WR={sub['win'].mean()*100:.1f}%", end="")
            print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC", choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip the rolling GARCH fit (uses cached garch_cache_<asset>.pkl)")
    args = parser.parse_args()

    cfg = ASSET_CONFIG[args.asset]
    cache_path = Path(f"garch_cache_{args.asset}.pkl")

    # Load 1h data
    print(f"\nLoading {args.asset} 1h data...")
    df_1h = pd.read_parquet(cfg["parquet_1h"])
    df_1h = df_1h[df_1h.index > "2020-01-01"]
    print(f"  {len(df_1h)} bars from {df_1h.index[0].date()} to {df_1h.index[-1].date()}")

    # GARCH rolling fit (slow — cache result)
    if args.skip_rolling and cache_path.exists():
        print(f"\nLoading cached GARCH results from {cache_path}...")
        df_garch = pd.read_pickle(cache_path)
    else:
        df_garch = compute_rolling_garch(df_1h["close"], fit_window=GARCH_FIT_WINDOW)
        df_garch.to_pickle(cache_path)
        print(f"  Saved to {cache_path}")

    section1_signal_validity(df_garch, cfg["big_move_thr"], args.asset)
    section2_trade_outcomes(df_garch, cfg["scan_archive"], args.asset)

    print("\nDone.")


if __name__ == "__main__":
    main()
