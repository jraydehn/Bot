"""
S5 -- Parity validation for sol_p_up_v1_model.py (mandatory pre-wiring
check). At N historical hour boundaries T, build features via the LIVE
module's _assemble() using a TRUNCATED trailing window (SOL 1000 bars,
mirroring the live fetch), and compare per-feature + model-output against
the training dataset. SOL's model needs no cross-asset/alt-coin fetch at
all (pure group-A), so this is simpler than BTC/ETH's parity checks.
"""
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc"
sys.path.insert(0, PROJ)
import sol_p_up_v1_model as V1

REBUILD = f"{PROJ}/reform_results/pup_v2_rebuild_20260704"
OUT = f"{PROJ}/reform_results/sol_pup_rebuild_20260706"

ds = pd.read_parquet(f"{OUT}/sol_dataset.parquet").sort_index()
import pickle
pipe = pickle.load(open(f"{OUT}/sol_p_up_v1_20260706.pkl", "rb"))
FEATS = pipe["features"]

sol_full = pd.read_parquet(f"{REBUILD}/hist_SOLUSDT_1h.parquet")

rng = np.random.default_rng(11)
cand = ds.index[(ds.index >= "2025-07-01") & (ds.index <= "2026-07-03")]
sample = pd.DatetimeIndex(sorted(rng.choice(cand, size=200, replace=False)))

rows, preds_mod = [], []
for T in sample:
    b1 = sol_full[sol_full.index <= T].tail(1000)[["open", "high", "low", "close", "volume"]]
    f = V1._assemble(T, b1)
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
