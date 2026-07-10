"""
S11 -- causal macro-regime posterior backfill, 2024-01 -> now, hourly.
EXACT live parity with _compute_macro_regime_probs (paper_trade_runner.py):
features ret_24h/ret_72h/rv24/sharpe_24h from 1h closes (same min_periods,
same fillna), decode = trailing 80 FEATURE bars ending at hour H, scaler +
predict_proba, take the LAST row's posterior. Effective time = H's bar
close (open + 1h): decoding at any 15m decision uses the last completed
1h bar, exactly as live.

Caveat recorded: the macro HMM pkl carries no training-window metadata; it
was fit on data that includes part of the evaluation era. It is unsupervised
(no outcome labels; 3 broad vol-tier states) so leakage risk into the
p_up tables is low, but 2026 "OOS" results carry this asterisk.
"""
import pickle
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

pay = pickle.load(open("reform_results/hmm_macro_regime_btc.pkl", "rb"))
model, scaler, label_names = pay["model"], pay["scaler"], pay["label_names"]

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2023-11-01"]
h1 = df1m["close"].resample("1h").last().dropna()
log_ret = np.log(h1 / h1.shift(1))
ret_24h = log_ret.rolling(24, min_periods=12).sum()
ret_72h = log_ret.rolling(72, min_periods=36).sum()
rv24 = log_ret.rolling(24, min_periods=12).std()
roll_mean = log_ret.rolling(24, min_periods=12).mean()
sharpe_24h = (roll_mean / rv24.replace(0, np.nan)).fillna(0.0)
feat = pd.DataFrame({"ret_24h": ret_24h, "ret_72h": ret_72h,
                     "rv24": rv24, "sharpe_24h": sharpe_24h}).dropna()
feat = feat[feat.index >= "2023-12-15"]
X_all = scaler.transform(feat.values.astype(float))
print(f"1h feature bars: {len(feat)}  {feat.index.min()} -> {feat.index.max()}")

rows = []
idx = feat.index
for i in range(80, len(feat)):
    w = X_all[i - 79:i + 1]                    # trailing 80 bars incl. current
    post = model.predict_proba(w)[-1]
    rows.append((idx[i], post[2], post[1], post[0]))   # label_names: 2=Bull,1=Sideways,0=Bear
reg = pd.DataFrame(rows, columns=["bar_open", "p_bull", "p_sdwy", "p_bear"])
reg["effective"] = reg["bar_open"] + pd.Timedelta("1h")
reg.to_csv(f"{OUT}/macro_regime_posteriors_1h.csv", index=False)
print(f"posteriors: {len(reg)}  {reg['bar_open'].min()} -> {reg['bar_open'].max()}")
reg["top"] = reg[["p_bull", "p_sdwy", "p_bear"]].idxmax(axis=1)
print("occupancy:", (reg["top"].value_counts(normalize=True) * 100).round(1).to_dict())
reg["yr"] = reg["bar_open"].dt.year
print(reg.groupby("yr")["top"].value_counts(normalize=True).round(3).to_string())
print("DONE_S11")
