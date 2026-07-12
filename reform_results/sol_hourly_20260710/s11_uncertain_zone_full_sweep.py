"""
S11 -- real rescue search. s1-s10 exhausted composite_trend/rev, composite_p_up,
p_gbdt, and DRIFT_MULTIPLIER variants -- all dead in the p_market 0.35-0.65
"uncertain zone" (the only population where beating the market matters).
That's one signal family tested three ways, not a real sweep.

Sweep every OTHER signal already logged in sol_scan_archive.csv against
that same zone: order-flow (vpin/obi/liq/ls_long_pct/oi_chg_pct), a second
ML output (p_up_v2, never tested), a cross-timeframe signal (pup15m, the
BTC-15m-style p_up model apparently also logged for SOL), pattern/structure
(ichi_bear, cloud_thick_pct, flag_*, sigma_swing_high/dist), macro regime
HMM probs, momentum/vol (adx_1h, rvol_1h, squeeze_1h, vol_eff, chg_*).

Ticker-clustered, bootstrap-significance-tested extreme-quantile edge vs
resolved_yes, isolated to the uncertain zone only.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ALL_COLS = [
    "logged_at", "contract_ticker", "p_market", "resolved_yes", "offset_pct",
    "p_up_v2", "pup15m",
    "vpin_score", "obi_score", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
    "funding_bias", "confirmation_score", "no_score", "vol_score",
    "adx_1h", "rvol_1h", "squeeze_1h", "vol_eff",
    "chg_30m", "chg_10m", "chg_5m", "bp_5m", "body_15m", "dir_15m", "pm_drift_5m",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score", "vwap_distance_pct", "stoch_k",
    "ichi_bear", "cloud_thick_pct",
    "flag_signal", "flag_bull_bars_ago", "flag_bear_bars_ago",
    "flag_bull_pole_pct", "flag_bear_pole_pct",
    "sigma_swing_high_1pct", "sigma_dist_high_1pct",
    "macro_regime_bull", "macro_regime_sdwy", "macro_regime_bear",
]

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False, usecols=lambda c: c in ALL_COLS)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "p_market", "resolved_yes"])
print(f"total rows: {len(df)}  tickers: {df['contract_ticker'].nunique()}")

unc = df[df["p_market"].between(0.35, 0.65)].copy()
print(f"uncertain zone (p_market 0.35-0.65): n={len(unc)}  tickers={unc['contract_ticker'].nunique()}\n")

SIGNAL_COLS = [c for c in ALL_COLS if c not in ("logged_at", "contract_ticker", "p_market", "resolved_yes", "offset_pct")]


def tk_edge_test(sub, col, mask_desc, mask):
    part = sub[mask]
    if len(part) < 30:
        return None
    tk = part.groupby("contract_ticker")["resolved_yes"].mean()
    if len(tk) < 15:
        return None
    n_tk = len(tk)
    wr = tk.mean()
    boots = [tk.sample(frac=1, replace=True, random_state=i).mean() for i in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # is 0.5 (coin flip) outside the CI?
    sig = not (lo <= 0.5 <= hi)
    return dict(col=col, desc=mask_desc, n=len(part), tk=n_tk, wr=wr, ci_lo=lo, ci_hi=hi, sig=sig)


results = []
for col in SIGNAL_COLS:
    if col not in unc.columns:
        continue
    s = pd.to_numeric(unc[col], errors="coerce")
    valid = s.notna()
    if valid.sum() < 100:
        continue
    vals = s[valid]
    tmp = unc[valid].copy()
    tmp["_v"] = vals

    # numeric signals: test top/bottom quintile
    nunique = vals.nunique()
    if nunique <= 6:
        # categorical/small-cardinality: test each level
        for lvl in sorted(vals.unique()):
            r = tk_edge_test(tmp, col, f"={lvl}", tmp["_v"] == lvl)
            if r:
                results.append(r)
    else:
        q20, q80 = vals.quantile([0.2, 0.8])
        r_lo = tk_edge_test(tmp, col, f"bottom20% (<{q20:.3g})", tmp["_v"] <= q20)
        r_hi = tk_edge_test(tmp, col, f"top20% (>{q80:.3g})", tmp["_v"] >= q80)
        if r_lo:
            results.append(r_lo)
        if r_hi:
            results.append(r_hi)

print(f"=== signals with 95% CI excluding 0.5 (real candidate edge), uncertain zone only ===")
print(f"{'signal':<22s} {'bucket':<26s} {'n':>5s} {'tk':>4s} {'WR':>7s} {'95% CI':>16s}")
sig_results = sorted([r for r in results if r["sig"]], key=lambda r: abs(r["wr"] - 0.5), reverse=True)
for r in sig_results:
    print(f"{r['col']:<22s} {r['desc']:<26s} {r['n']:5d} {r['tk']:4d} {r['wr']:7.1%} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")
if not sig_results:
    print("  (none)")

print(f"\n=== all tested (for reference), sorted by |WR-0.5| ===")
all_sorted = sorted(results, key=lambda r: abs(r["wr"] - 0.5), reverse=True)
for r in all_sorted[:30]:
    flag = " ***" if r["sig"] else ""
    print(f"{r['col']:<22s} {r['desc']:<26s} {r['n']:5d} {r['tk']:4d} {r['wr']:7.1%}{flag}")

print(f"\ntotal signal x bucket combos tested: {len(results)}")
print("\nDONE_S11")
