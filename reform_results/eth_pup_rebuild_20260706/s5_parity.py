"""
S5 -- Parity validation for eth_p_up_v1_model.py (mandatory pre-wiring
check, same discipline as the BTC rebuild's s9_parity.py). At N historical
hour boundaries T, build features via the LIVE module's _assemble() using
TRUNCATED trailing windows that mirror what the live fetches actually
provide (ETH 1000 bars, BTC 60 bars, XRP/DOGE/ADA 30 bars each), and
compare per-feature + model-output against the training dataset.
"""
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc"
sys.path.insert(0, PROJ)
import eth_p_up_v1_model as V1

REBUILD = f"{PROJ}/reform_results/pup_v2_rebuild_20260704"
OUT = f"{PROJ}/reform_results/eth_pup_rebuild_20260706"

ds = pd.read_parquet(f"{OUT}/eth_dataset.parquet").sort_index()
FEATS = pd.read_parquet(f"{OUT}/eth_p_up_v1_20260706.pkl")["features"] if False else None
import pickle
pipe = pickle.load(open(f"{OUT}/eth_p_up_v1_20260706.pkl", "rb"))
FEATS = pipe["features"]

eth_full = pd.read_parquet(f"{REBUILD}/hist_ETHUSDT_1h.parquet")
btc_full = pd.read_parquet(f"{REBUILD}/hist_BTCUSDT_1h.parquet")
xrp_full = pd.read_parquet(f"{REBUILD}/hist_XRPUSDT_1h.parquet")
doge_full = pd.read_parquet(f"{REBUILD}/hist_DOGEUSDT_1h.parquet")
ada_full = pd.read_parquet(f"{REBUILD}/hist_ADAUSDT_1h.parquet")

rng = np.random.default_rng(11)
cand = ds.index[(ds.index >= "2025-07-01") & (ds.index <= "2026-07-03")]
sample = pd.DatetimeIndex(sorted(rng.choice(cand, size=200, replace=False)))

rows, preds_mod = [], []
for T in sample:
    b1 = eth_full[eth_full.index <= T].tail(1000)[["open", "high", "low", "close", "volume"]]
    btc60 = btc_full[btc_full.index <= T].tail(60)[["open", "high", "low", "close", "volume"]]
    xrp30 = xrp_full[xrp_full.index <= T].tail(30)["close"]
    doge30 = doge_full[doge_full.index <= T].tail(30)["close"]
    ada30 = ada_full[ada_full.index <= T].tail(30)["close"]
    f = V1._assemble(T, b1, btc60["close"], xrp30, doge30, ada30)
    f["_T"] = T
    rows.append(f)
    vec = np.array([[f.get(k, np.nan) for k in FEATS]], dtype=float)
    preds_mod.append(float(pipe["clf"].predict_proba(vec)[0, 1]))

live = pd.DataFrame(rows).set_index("_T")
train = ds.loc[sample, FEATS]

print(f"parity over {len(sample)} hour boundaries ({sample[0].date()} -> {sample[-1].date()})\n")
print(f"{'feature':<22} {'spearman':>9} {'mean_abs_diff':>14} {'live_nonnull':>12} {'train_nonnull':>13}")
for f in FEATS:
    a = live[f].astype(float); b = train[f].astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        corr = float("nan")
    else:
        corr = a[mask].corr(b[mask], method="spearman")
    mad = (a[mask] - b[mask]).abs().mean() if mask.sum() else float("nan")
    print(f"{f:<22} {corr:9.4f} {mad:14.6f} {a.notna().sum():12d} {b.notna().sum():13d}")

pred_mod = pd.Series(preds_mod, index=sample)
pred_train_direct = pd.Series(
    pipe["clf"].predict_proba(train.fillna(train.mean()).values)[:, 1], index=sample)
print(f"\nmodel output pearson (live-assembled feats vs training-row feats): "
      f"{pred_mod.corr(pred_train_direct):.4f}")
print(f"live pred range: [{pred_mod.min():.4f}, {pred_mod.max():.4f}]")
