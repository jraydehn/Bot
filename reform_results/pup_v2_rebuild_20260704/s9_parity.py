#!/usr/bin/env python3
"""S9 — Parity validation for btc_p_up_v3_model.py (mandatory pre-report).

At N historical hour boundaries T, build features via the LIVE module's
_assemble() using truncated trailing windows that mirror what the live
fetches provide (BTC 1h tail(1000), 15m tail(1000), ETH/SOL closes tail(40),
1m slice [T-240h, T+1h)) and compare per-feature against the training
dataset (extended_dataset.parquet). Also compares model outputs:
  (a) artifact clf on module vectors vs artifact clf on training rows
      (isolates feature-construction parity), and
  (b) module output vs walk-forward OOS preds (weekly-refit models — bounded
      disagreement expected by construction).
"""
import sys, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(PROJ))
import btc_p_up_v3_model as V3

ds = pd.read_parquet(HERE / "extended_dataset.parquet").sort_index()
wf = pd.read_parquet(HERE / "wf_preds_FINAL.parquet")["p"]
pipe = pickle.load(open(HERE / "btc_p_up_v3_20260704.pkl", "rb"))
FEATS = pipe["features"]

b1_full = pd.read_parquet(HERE / "hist_BTCUSDT_1h.parquet")
m15_full = pd.read_parquet(HERE / "hist_BTCUSDT_15m.parquet")
m1_full = pd.read_parquet(HERE / "hist_BTCUSDT_1m.parquet")
eth_full = pd.read_parquet(HERE / "hist_ETHUSDT_1h.parquet")["close"]
sol_full = pd.read_parquet(HERE / "hist_SOLUSDT_1h.parquet")["close"]

rng = np.random.default_rng(11)
cand = ds.index[(ds.index >= "2025-07-01") & (ds.index <= "2026-07-03")]
sample = pd.DatetimeIndex(sorted(rng.choice(cand, size=320, replace=False)))

rows, preds_mod = [], []
for T in sample:
    b1 = b1_full[b1_full.index <= T].tail(1000)[["open", "high", "low", "close", "volume"]]
    m15 = m15_full[m15_full.index <= T].tail(1000)[["open", "high", "low", "close", "volume"]]
    eth_c = eth_full[eth_full.index <= T].tail(40)
    sol_c = sol_full[sol_full.index <= T].tail(40)
    m1 = m1_full[(m1_full.index >= T - pd.Timedelta(hours=240)) &
                 (m1_full.index < T + pd.Timedelta(hours=1))][
                 ["open", "high", "low", "close", "volume"]]
    f = V3._assemble(T, b1, m15, eth_c, sol_c, m1)
    f["_T"] = T
    rows.append(f)
    vec = np.array([[f.get(k, np.nan) for k in FEATS]], dtype=float)
    preds_mod.append(float(pipe["clf"].predict_proba(vec)[0, 1]))

live = pd.DataFrame(rows).set_index("_T")
train = ds.loc[sample, FEATS]

print(f"parity over {len(sample)} hour boundaries "
      f"({sample[0].date()} -> {sample[-1].date()})\n")
print(f"{'feature':<22} {'spearman':>9} {'pearson':>9} {'max|diff|':>10} {'n':>4}  note")
res = []
for k in FEATS:
    a = live[k].astype(float)
    b = train[k].astype(float)
    ok = a.notna() & b.notna()
    n = int(ok.sum())
    if n < 20:
        print(f"{k:<22} {'—':>9} {'—':>9} {'—':>10} {n:>4}  insufficient non-NaN")
        continue
    if a[ok].nunique() <= 1 or b[ok].nunique() <= 1:
        match = (a[ok] == b[ok]).mean()
        print(f"{k:<22} {'const':>9} {'—':>9} {'—':>10} {n:>4}  degenerate; exact-match={match:.3f}")
        continue
    sp = spearmanr(a[ok], b[ok]).statistic
    pe = np.corrcoef(a[ok], b[ok])[0, 1]
    md = (a[ok] - b[ok]).abs().max()
    exact = (a[ok].round(10) == b[ok].round(10)).mean()
    note = "EXACT" if exact > 0.99 else ("ok" if sp >= 0.99 else "REVIEW")
    res.append({"feature": k, "spearman": sp, "pearson": pe, "n": n})
    print(f"{k:<22} {sp:>9.4f} {pe:>9.4f} {md:>10.3g} {n:>4}  {note}")

# NaN coverage
nn = live[FEATS].notna().mean()
low = nn[nn < 0.95]
if len(low):
    print("\nfeatures with <95% coverage in live build:")
    print(low.round(3).to_string())

# output parity
pm = pd.Series(preds_mod, index=sample)
X_train = train.values.astype(float)
p_train = pd.Series(pipe["clf"].predict_proba(X_train)[:, 1], index=sample)
ok = pm.notna() & p_train.notna()
print(f"\nOUTPUT parity (artifact clf: module features vs training features): "
      f"pearson={np.corrcoef(pm[ok], p_train[ok])[0,1]:.4f}  "
      f"spearman={spearmanr(pm[ok], p_train[ok]).statistic:.4f}  "
      f"mean|diff|={ (pm[ok]-p_train[ok]).abs().mean():.4f}")
w = wf.reindex(sample)
ok2 = pm.notna() & w.notna()
print(f"OUTPUT vs walk-forward OOS preds (weekly-refit models): "
      f"pearson={np.corrcoef(pm[ok2], w[ok2])[0,1]:.4f}  "
      f"spearman={spearmanr(pm[ok2], w[ok2]).statistic:.4f}  n={int(ok2.sum())}")
