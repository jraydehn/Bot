"""
S0 -- SOL exploratory sweep. Confirm (or refute) SOL's own-return and
cross-asset autocorrelation character BEFORE picking any candidate feature
groups. Per feedback_eth_sol_model_approach: never assume BTC/ETH's finding
(mean-reversion, negative IC) transfers to SOL -- SOL has previously shown
the OPPOSITE (mean-reversion is real for SOL too per project_sol_hourly_deepdive,
but must be re-confirmed fresh on long history per feedback_test_on_long_history).
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RB = "reform_results/pup_v2_rebuild_20260704"
sol = pd.read_parquet(f"{RB}/hist_SOLUSDT_1h.parquet").sort_index()
btc = pd.read_parquet(f"{RB}/hist_BTCUSDT_1h.parquet").sort_index()
eth = pd.read_parquet(f"{RB}/hist_ETHUSDT_1h.parquet").sort_index()

sol_c = sol["close"].astype(float)
btc_c = btc["close"].astype(float)
eth_c = eth["close"].astype(float)

print(f"SOL history: {sol_c.index[0]} -> {sol_c.index[-1]}  ({len(sol_c)} bars)")

sol_ret = np.log(sol_c / sol_c.shift(1))
sol_fwd = np.log(sol_c.shift(-1) / sol_c)  # next-hour return (target proxy)
sol_fwd_dir = (sol_fwd > 0).astype(float)

df = pd.DataFrame({"sol_ret": sol_ret, "fwd_dir": sol_fwd_dir, "fwd_ret": sol_fwd}).dropna()
df["year"] = df.index.year


def ic_by_year(x, y, years):
    rows = []
    for yr in years:
        m = (df["year"] == yr)
        if m.sum() < 100:
            continue
        xx, yy = x[m], y[m]
        ic, p = stats.spearmanr(xx, yy)
        rows.append((yr, ic, p, m.sum()))
    return rows


years = sorted(df["year"].unique())
print("\n=== SOL own-return IC on next-hour direction, by year ===")
for yr, ic, p, n in ic_by_year(df["sol_ret"], df["fwd_dir"], years):
    print(f"  {yr}: IC={ic:+.4f}  p={p:.4f}  n={n}")

overall_ic, overall_p = stats.spearmanr(df["sol_ret"], df["fwd_dir"])
print(f"  ALL: IC={overall_ic:+.4f}  p={overall_p:.4g}  n={len(df)}")

# Cross-asset: BTC and ETH returns aligned to SOL's timestamps
btc_ret = np.log(btc_c / btc_c.shift(1)).reindex(df.index, method="ffill")
eth_ret = np.log(eth_c / eth_c.shift(1)).reindex(df.index, method="ffill")
df["btc_ret"] = btc_ret
df["eth_ret"] = eth_ret

for name, col in [("BTC ret -> SOL fwd dir", "btc_ret"), ("ETH ret -> SOL fwd dir", "eth_ret")]:
    print(f"\n=== {name}, by year ===")
    sub = df.dropna(subset=[col])
    for yr in years:
        m = (sub["year"] == yr)
        if m.sum() < 100:
            continue
        ic, p = stats.spearmanr(sub[col][m], sub["fwd_dir"][m])
        print(f"  {yr}: IC={ic:+.4f}  p={p:.4f}  n={m.sum()}")
    ic, p = stats.spearmanr(sub[col], sub["fwd_dir"])
    print(f"  ALL: IC={ic:+.4f}  p={p:.4g}  n={len(sub)}")

# Multi-lag autocorrelation of SOL's own return (1h, 4h, 24h lookback -> next 1h)
print("\n=== SOL multi-lag own-return autocorrelation (lookback -> next-1h dir) ===")
for lag_h in [1, 2, 4, 8, 24, 168]:
    lagged = np.log(sol_c / sol_c.shift(lag_h)).reindex(df.index)
    sub = pd.DataFrame({"x": lagged, "y": df["fwd_dir"]}).dropna()
    ic, p = stats.spearmanr(sub["x"], sub["y"])
    print(f"  lookback={lag_h}h: IC={ic:+.4f}  p={p:.4g}  n={len(sub)}")
