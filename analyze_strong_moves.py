"""
analyze_strong_moves.py

Target: |price_move_pct| — how large was the actual price move from trade time to expiry.
Goal: find signals that predict LARGE moves, then layer with stoch<17 to find the best
      combination for the stoch_bounce YES model.

Also runs the existing stoch_bounce bucket analysis:
  stoch_k < 17, pm < 0.60, side=YES — what additional signals improve WR?
"""
import glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"

FEE_RATE = 0.07

# ── load ─────────────────────────────────────────────────────────────────────

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

    for col in ["p_market", "resolved_yes", "spot", "strike", "tau_minutes",
                "vol_eff", "price_move_pct", "stoch_k", "stoch_d", "avg_funding_rate"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(subset=["p_market", "resolved_yes"])

    # Merge backfilled features
    pup_path = RES_DIR / "p_up_v2_backfilled.csv"
    if pup_path.exists():
        pup = pd.read_csv(pup_path)
        pup["logged_at"] = pd.to_datetime(pup["logged_at"], utc=True)
        trades = trades.merge(pup, on=["contract_ticker", "logged_at", "side"], how="left")

    trades["won"] = np.where(
        trades["side"] == "yes",
        trades["resolved_yes"] == 1,
        trades["resolved_yes"] == 0,
    )
    trades["market_miss"] = trades["resolved_yes"] - trades["p_market"]
    trades["abs_move"]    = trades["price_move_pct"].abs()
    return trades.reset_index(drop=True)


def spearman(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 20:
        return float("nan"), float("nan"), 0
    r, p = stats.spearmanr(a[mask], b[mask])
    return r, p, mask.sum()


# ── 1. Signal → |price_move_pct| correlations (what predicts BIG moves?) ─────

MOVE_FEATURES = {
    "stoch_k":             "Stoch K",
    "stoch_d":             "Stoch D",
    "avg_funding_rate":    "Avg funding rate",
    "funding_bias":        "Funding bias",
    "adx_1h":              "ADX 1h",
    "rvol_1h":             "RVOL 1h",
    "squeeze_1h":          "Squeeze 1h",
    "ob_imbalance":        "OB imbalance",
    "obi_score":           "OBI score",
    "vpin_score":          "VPIN score",
    "vpin_raw":            "VPIN raw",
    "liq_score":           "Liq score",
    "liq_bias":            "Liq bias",
    "ema_stretch_score":   "EMA stretch",
    "stoch_bias":          "Stoch bias",
    "stoch_crossover_active": "Stoch crossover active",
    "stoch_flipped":       "Stoch flipped",
    "vwap_score":          "VWAP score",
    "vwap_distance_pct":   "VWAP distance %",
    "composite_rev":       "Composite rev",
    "composite_trend":     "Composite trend",
    "chg_5m":              "chg 5m",
    "chg_10m":             "chg 10m",
    "chg_30m":             "chg 30m",
    "vol_eff":             "vol_eff",
    "vol_score":           "Vol score",
    "dc_4h_n20_pos":       "DC 4h N20 pos",
    "dc_4h_n20_break":     "DC 4h N20 break",
    "dc_1h_n20_pos":       "DC 1h N20 pos",
    "ema20_slope_4h":      "EMA20 slope 4h",
    "ema20_slope_1h":      "EMA20 slope 1h",
    "pm_drift_5m":         "p_market drift 5m",
    "p_market":            "p_market",
    "tau_minutes":         "Tau (min)",
}


def analyze_move_predictors(trades: pd.DataFrame):
    btc_yes = trades[(trades["side"] == "yes")].copy()
    target = btc_yes["abs_move"].values

    rows = []
    for feat, label in MOVE_FEATURES.items():
        if feat not in btc_yes.columns:
            continue
        vals = pd.to_numeric(btc_yes[feat], errors="coerce").values
        r, p, n = spearman(vals, target)
        if n < 20:
            continue
        rows.append({"feature": feat, "label": label, "r_abs_move": r,
                     "p": p, "n": n, "abs_r": abs(r)})

    df = pd.DataFrame(rows)
    if df.empty:
        print("  (no features with n>=20 and abs_move data)")
        return df
    df = df.sort_values("abs_r", ascending=False)
    print(f"\n{'='*72}")
    print(f"  SIGNAL → |price_move_pct| SPEARMAN  (YES side, n≥20)")
    print(f"  Positive r = feature high → bigger move; Negative = smaller")
    print(f"{'='*72}")
    print(f"  {'Feature':<26} {'r':>7} {'p':>7} {'n':>5}")
    print(f"  {'-'*52}")
    for _, row in df.iterrows():
        sig = "*" if row["p"] < 0.05 else " "
        print(f"  {row['feature']:<26} {row['r_abs_move']:>+7.3f}{sig} {row['p']:>7.4f} {row['n']:>5,}  {row['label']}")
    return df


# ── 2. Stoch bounce bucket: stoch<17 + pm<0.60 + YES ─────────────────────────

def stoch_bounce_breakdown(trades: pd.DataFrame):
    yes = trades[trades["side"] == "yes"].copy()
    bucket = yes[(yes["stoch_k"] < 17) & (yes["p_market"] < 0.60)].copy()

    print(f"\n{'='*72}")
    print(f"  STOCH BOUNCE BUCKET  (YES, stoch_k<17, pm<0.60)")
    print(f"  n={len(bucket):,}  WR={bucket['won'].mean():.1%}  "
          f"mean_miss={bucket['market_miss'].mean():+.4f}")
    print(f"{'='*72}")

    # Check which secondary signals improve WR within the bucket
    SECONDARY = {
        "stoch_crossover_active": (1, "== 1"),
        "stoch_flipped":          (1, "== 1"),
        "stoch_d":                (17, "< 17 (both oversold)"),
        "squeeze_1h":             (1, "== 1 (compression)"),
        "adx_1h":                 (20, "< 20 (low trend, ranging)"),
        "funding_bias":           (-1, "== -1 (shorts crowded)"),
        "avg_funding_rate":       (0,  "< 0 (negative funding)"),
        "vpin_score":             (0,  "< 0 (low informed trading)"),
        "rvol_1h":                (1,  "< 1 (low relative vol)"),
        "liq_bias":               (1,  "== 1 (buy-side liq)"),
        "obi_score":              (1,  "> 0 (bid-heavy OB)"),
        "composite_rev":          (0,  "> 0 (bullish rev)"),
        "dc_4h_n20_pos":          (0.3,"< 0.3 (near channel bottom)"),
        "ema20_slope_1h":         (0,  "< 0 (falling EMA, oversold more likely)"),
        "ema20_slope_4h":         (0,  "< 0 (4h falling)"),
        "vwap_score":             (-1, "== -1 (below VWAP)"),
        "ob_imbalance":           (0,  "> 0 (buy-side OB)"),
    }

    rows = []
    for feat, (thresh, desc) in SECONDARY.items():
        if feat not in bucket.columns:
            continue
        vals = pd.to_numeric(bucket[feat], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 5:
            continue

        # Compute sub-group
        if desc.startswith("== 1"):
            sub = bucket[mask & (vals == 1)]
            anti = bucket[mask & (vals != 1)]
        elif desc.startswith("== -1"):
            sub = bucket[mask & (vals == -1)]
            anti = bucket[mask & (vals != -1)]
        elif desc.startswith("< "):
            sub = bucket[mask & (vals < thresh)]
            anti = bucket[mask & (vals >= thresh)]
        elif desc.startswith("> "):
            sub = bucket[mask & (vals > thresh)]
            anti = bucket[mask & (vals <= thresh)]
        else:
            continue

        if len(sub) < 3:
            continue

        sub_wr   = sub["won"].mean()
        anti_wr  = anti["won"].mean() if len(anti) >= 3 else float("nan")
        delta_wr = sub_wr - anti_wr
        avg_pm = sub["p_market"].mean()
        fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
        be_wr = avg_pm + fee_avg

        rows.append({
            "feat": feat, "condition": f"{feat} {desc}",
            "n_sub": len(sub), "wr_sub": sub_wr, "be_wr": be_wr,
            "n_anti": len(anti), "wr_anti": anti_wr, "delta_wr": delta_wr,
            "miss_sub": sub["market_miss"].mean(),
        })

    df = pd.DataFrame(rows).sort_values("wr_sub", ascending=False)
    print(f"\n  Secondary signal breakdown (within stoch<17 + pm<0.60 + YES bucket):")
    print(f"  {'Condition':<42} {'n':>4} {'WR':>6} {'BE':>6} {'Δvs-rest':>9} {'miss':>7}")
    print(f"  {'-'*72}")
    for _, row in df.iterrows():
        delta = f"{row['delta_wr']:>+.1%}" if not np.isnan(row["delta_wr"]) else "  NaN "
        print(f"  {row['condition']:<42} {row['n_sub']:>4} {row['wr_sub']:>6.1%} "
              f"{row['be_wr']:>6.1%} {delta:>9} {row['miss_sub']:>+7.4f}")

    return df


# ── 3. Multi-signal combination analysis ─────────────────────────────────────

def combo_analysis(trades: pd.DataFrame):
    yes = trades[trades["side"] == "yes"].copy()

    print(f"\n{'='*72}")
    print(f"  MULTI-SIGNAL COMBO ANALYSIS  (YES side, pm<0.60)")
    print(f"  Base: stoch_k<17 + pm<0.60")
    print(f"{'='*72}")

    base = yes[(yes["stoch_k"] < 17) & (yes["p_market"] < 0.60)]

    combos = [
        ("stoch_k<17 [base]",                  base),
        ("+ stoch_d<17",                        base[pd.to_numeric(base.get("stoch_d", pd.Series(dtype=float)), errors="coerce") < 17]),
        ("+ crossover==1",                      base[pd.to_numeric(base.get("stoch_crossover_active", pd.Series(dtype=float)), errors="coerce") == 1]),
        ("+ flipped==1",                        base[pd.to_numeric(base.get("stoch_flipped", pd.Series(dtype=float)), errors="coerce") == 1]),
        ("+ squeeze_1h==1",                     base[pd.to_numeric(base.get("squeeze_1h", pd.Series(dtype=float)), errors="coerce") == 1]),
        ("+ funding_bias==-1",                  base[pd.to_numeric(base.get("funding_bias", pd.Series(dtype=float)), errors="coerce") == -1]),
        ("+ avg_funding<0",                     base[pd.to_numeric(base.get("avg_funding_rate", pd.Series(dtype=float)), errors="coerce") < 0]),
        ("+ dc_4h_pos<0.30",                    base[pd.to_numeric(base.get("dc_4h_n20_pos", pd.Series(dtype=float)), errors="coerce") < 0.30]),
        ("+ obi>0",                             base[pd.to_numeric(base.get("obi_score", pd.Series(dtype=float)), errors="coerce") > 0]),
        ("+ composite_rev>0",                   base[pd.to_numeric(base.get("composite_rev", pd.Series(dtype=float)), errors="coerce") > 0]),
        ("+ adx<20",                            base[pd.to_numeric(base.get("adx_1h", pd.Series(dtype=float)), errors="coerce") < 20]),
        # AND combos
        ("+ crossover + funding<0",             base[
            (pd.to_numeric(base.get("stoch_crossover_active", pd.Series(dtype=float)), errors="coerce") == 1) &
            (pd.to_numeric(base.get("avg_funding_rate", pd.Series(dtype=float)), errors="coerce") < 0)
        ]),
        ("+ dc_pos<0.30 + funding<0",           base[
            (pd.to_numeric(base.get("dc_4h_n20_pos", pd.Series(dtype=float)), errors="coerce") < 0.30) &
            (pd.to_numeric(base.get("avg_funding_rate", pd.Series(dtype=float)), errors="coerce") < 0)
        ]),
    ]

    print(f"  {'Condition':<38} {'n':>4} {'WR':>6} {'BE_WR':>6} {'edge_above_BE':>14}")
    print(f"  {'-'*68}")
    for label, sub in combos:
        if len(sub) < 3:
            print(f"  {label:<38} {len(sub):>4}  (too few)")
            continue
        avg_pm = sub["p_market"].mean()
        fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
        be_wr = avg_pm + fee_avg
        wr = sub["won"].mean()
        print(f"  {label:<38} {len(sub):>4} {wr:>6.1%} {be_wr:>6.1%} {wr - be_wr:>+14.1%}")


# ── 4. Stoch crossover timing analysis ───────────────────────────────────────

def crossover_analysis(trades: pd.DataFrame):
    yes = trades[trades["side"] == "yes"].copy()
    yes["stoch_k"] = pd.to_numeric(yes["stoch_k"], errors="coerce")
    yes["stoch_crossover_active"] = pd.to_numeric(yes.get("stoch_crossover_active", pd.Series(dtype=float)), errors="coerce")
    yes["stoch_flipped"] = pd.to_numeric(yes.get("stoch_flipped", pd.Series(dtype=float)), errors="coerce")

    print(f"\n{'='*72}")
    print(f"  STOCH CROSSOVER / FLIP ANALYSIS  (YES side)")
    print(f"{'='*72}")

    buckets = [
        ("All YES",                          yes),
        ("crossover_active==1",              yes[yes["stoch_crossover_active"] == 1]),
        ("crossover_active==1 + pm<0.60",    yes[(yes["stoch_crossover_active"] == 1) & (yes["p_market"] < 0.60)]),
        ("crossover_active==1 + sk<30",      yes[(yes["stoch_crossover_active"] == 1) & (yes["stoch_k"] < 30)]),
        ("crossover_active==1 + sk<20",      yes[(yes["stoch_crossover_active"] == 1) & (yes["stoch_k"] < 20)]),
        ("crossover_active==1 + sk<17",      yes[(yes["stoch_crossover_active"] == 1) & (yes["stoch_k"] < 17)]),
        ("stoch_flipped==1",                 yes[yes["stoch_flipped"] == 1]),
        ("stoch_flipped==1 + sk<30",         yes[(yes["stoch_flipped"] == 1) & (yes["stoch_k"] < 30)]),
        ("stoch_flipped==1 + sk<20",         yes[(yes["stoch_flipped"] == 1) & (yes["stoch_k"] < 20)]),
    ]

    print(f"  {'Bucket':<38} {'n':>4} {'WR':>6} {'BE_WR':>6}")
    print(f"  {'-'*58}")
    for label, sub in buckets:
        if len(sub) < 3:
            print(f"  {label:<38} {len(sub):>4}  (too few)")
            continue
        avg_pm = sub["p_market"].mean()
        fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
        be_wr = avg_pm + fee_avg
        wr = sub["won"].mean()
        print(f"  {label:<38} {len(sub):>4} {wr:>6.1%} {be_wr:>6.1%}")


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading trades...")
    trades = load_trades()
    print(f"  {len(trades):,} resolved trades")

    analyze_move_predictors(trades)
    stoch_bounce_breakdown(trades)
    combo_analysis(trades)
    crossover_analysis(trades)


if __name__ == "__main__":
    run()
