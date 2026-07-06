"""
S0 -- ETH-specific exploratory sweep, BEFORE assuming any BTC winning
feature group transfers. Tests, on ETH's own 2020-2026 hourly history:
  1. Does recent hourly momentum predict continuation or reversal at
     1h-ahead (basic autocorrelation structure)?
  2. Does the SAME intra-hour microstructure pattern that won for BTC
     (ret_last15 mean-reverts into the next hour) hold for ETH, or is it
     different/absent?
  3. Cross-asset lead-lag: does BTC's or SOL's recent return predict
     ETH's next-hour direction (the reverse of BTC's B group, which used
     ETH/SOL to predict BTC)?
  4. Per-year stability of all of the above (never trust a single window).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

eth = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_ETHUSDT_1h.parquet")["close"].astype(float).sort_index()
btc = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet")["close"].astype(float).sort_index()
sol = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_SOLUSDT_1h.parquet")["close"].astype(float).sort_index()

lr_eth = np.log(eth / eth.shift(1))
label = (eth.shift(-1) > eth).astype(float)  # next-hour direction
label[eth.shift(-1).isna()] = np.nan

print("=== 1. Autocorrelation of ETH hourly log returns (lag 1-24) ===")
for lag in [1, 2, 3, 6, 12, 24]:
    ac = lr_eth.autocorr(lag)
    print(f"  lag={lag:2d}h: autocorr={ac:+.4f}")

print("\n=== 2. Does lr_eth (this hour's return) predict next-hour direction? Per year ===")
df = pd.DataFrame({"lr": lr_eth, "label": label}).dropna()
df["year"] = df.index.year
for yr, g in df.groupby("year"):
    ic = spearmanr(g["lr"], g["label"]).statistic
    print(f"  {yr}: IC={ic:+.4f}  n={len(g)}")
overall_ic = spearmanr(df["lr"], df["label"]).statistic
print(f"  OVERALL: IC={overall_ic:+.4f}  n={len(df)}")

print("\n=== 3. Cross-asset lead-lag: BTC/SOL 1h return -> ETH next-hour direction ===")
lr_btc = np.log(btc / btc.shift(1))
lr_sol = np.log(sol / sol.shift(1))
cross = pd.DataFrame({"lr_btc": lr_btc, "lr_sol": lr_sol, "label": label}).dropna()
cross["year"] = cross.index.year
for name, col in [("BTC->ETH", "lr_btc"), ("SOL->ETH", "lr_sol")]:
    print(f"  {name}:")
    for yr, g in cross.groupby("year"):
        ic = spearmanr(g[col], g["label"]).statistic
        print(f"    {yr}: IC={ic:+.4f}  n={len(g)}")
    ic_all = spearmanr(cross[col], cross["label"]).statistic
    print(f"    OVERALL: IC={ic_all:+.4f}")
