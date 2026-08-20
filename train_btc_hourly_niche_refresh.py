"""BTC hourly niche REFRESH challenger — 2026-08-20.

Motivation: the frozen niche model (train 05-18..06-19) went silent on
08-13 when BTC broke out of its training range — 8 days at zero fire rate
vs a validated ~6.7/day (fire-rate collapse = the canonical staleness
tell). Refused band population made +$2,854/338 since 08-13.

Recipe: IDENTICAL to btc_hourly_lgbm_niche_20260728 (same LGBM params,
same 49 features incl. derived z_moneyness) — ONLY the training window
moves. Discipline per feedback_lgbm_single_fit_not_a_finding:
  - 6 seeds (report spread, deploy the median-val seed)
  - walk-forward sanity fold (train ..07-15 -> test 07-16..07-31)
  - TRAIN 05-18..07-31, early-stop val 08-01..08-09,
    UNTOUCHED holdout 08-10.. (melt-up + reversal; the frozen model's
    live record covers it, the refresh never saw it)
Eval = the niche band rule, flat $100 fee-net:
  YES only, pm in [0.35,0.65], edge >= 0.06 (original rule)
  + the NICHE-CAL variant pm [0.32,0.45], edge 0.06..0.20
Also reports fire rate on the 08-13+ drought window.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent
OUT = BASE / "models" / "btc_hourly_lgbm_niche_refresh_20260820.pkl"

with open(BASE / "models" / "btc_hourly_lgbm_niche_20260728.pkl", "rb") as f:
    _old = pickle.load(f)
FEATS = _old["features"]
PARAMS = {k: v for k, v in _old["model"].get_params().items()
          if k not in ("random_state", "n_jobs")}

df = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
for c in set(FEATS + ["p_market", "tau_minutes", "strike", "spot", "resolved_yes"]) - {"z_moneyness"}:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
with np.errstate(divide="ignore", invalid="ignore"):
    df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(df["tau_minutes"].clip(lower=1))
df = df.dropna(subset=["dt", "p_market", "resolved_yes"])
df = df[df["tau_minutes"] > 0].sort_values("dt").reset_index(drop=True)
print(f"resolved scans: {len(df)}  {df['dt'].min():%m-%d}..{df['dt'].max():%m-%d}")

def window(a, b):
    return df[(df["dt"] >= a) & (df["dt"] < b)]

def band_book(d, p, lo, hi, e0, e1=None):
    m = (d["p_market"] >= lo) & (d["p_market"] <= hi)
    edge = p - d["p_market"] - 0.07 * d["p_market"] * (1 - d["p_market"])
    m &= edge >= e0
    if e1 is not None:
        m &= edge <= e1
    t = d[m].sort_values("dt").drop_duplicates("contract_ticker", keep="first")
    if not len(t):
        return 0, 0.0, 0.0
    pm = t["p_market"]
    win = t["resolved_yes"] == 1
    net = np.where(win, 100 * (1 - pm) / pm, -100.0) - (100 / pm) * 0.07 * pm * (1 - pm)
    return len(t), float(net.sum()), float(win.mean())

def fit(train, val, seed):
    m = LGBMClassifier(**PARAMS, random_state=seed, n_jobs=-1)
    m.fit(train[FEATS], (train["resolved_yes"] == 1).astype(int),
          eval_set=[(val[FEATS], (val["resolved_yes"] == 1).astype(int))],
          eval_metric="binary_logloss",
          callbacks=__import__("lightgbm").early_stopping(50, verbose=False) and
                    [__import__("lightgbm").early_stopping(50, verbose=False)])
    return m

TRAIN = window("2026-05-18", "2026-08-01")
VAL = window("2026-08-01", "2026-08-10")
HOLD = window("2026-08-10", "2026-12-01")
DROUGHT = window("2026-08-13", "2026-12-01")
print(f"train n={len(TRAIN)}  val n={len(VAL)}  holdout n={len(HOLD)}")

print("\n--- walk-forward sanity fold (train ..07-15 -> test 07-16..07-31, seed 1) ---")
wf_tr = window("2026-05-18", "2026-07-08")
wf_va = window("2026-07-08", "2026-07-16")
wf_te = window("2026-07-16", "2026-08-01")
mwf = fit(wf_tr, wf_va, 1)
pwf = mwf.predict_proba(wf_te[FEATS])[:, 1]
n, s, w = band_book(wf_te, pwf, 0.35, 0.65, 0.06)
print(f"  wf test band book: n={n} net={s:+,.0f} WR={w*100:.0f}%")

print("\n--- 6-seed refresh (train ..07-31, val 08-01..09, HOLDOUT 08-10+) ---")
rows = []
models = {}
for seed in (1, 2, 3, 4, 5, 6):
    m = fit(TRAIN, VAL, seed)
    ph = m.predict_proba(HOLD[FEATS])[:, 1]
    n1, s1, w1 = band_book(HOLD, ph, 0.35, 0.65, 0.06)
    n2, s2, w2 = band_book(HOLD, ph, 0.32, 0.45, 0.06, 0.20)
    pdr = m.predict_proba(DROUGHT[FEATS])[:, 1]
    nd, sd, wd = band_book(DROUGHT, pdr, 0.35, 0.65, 0.06)
    # val book for seed selection (NOT holdout)
    pv = m.predict_proba(VAL[FEATS])[:, 1]
    nv, sv, wv = band_book(VAL, pv, 0.35, 0.65, 0.06)
    rows.append((seed, nv, sv, n1, s1, w1, n2, s2, nd, sd))
    models[seed] = m
    print(f"  seed {seed}: val n={nv:3d} {sv:+7,.0f} | HOLD band n={n1:3d} {s1:+7,.0f} WR{w1*100:3.0f}% "
          f"| cal-band n={n2:3d} {s2:+7,.0f} | drought(08-13+) n={nd:3d} {sd:+7,.0f}")

hold_nets = [r[4] for r in rows]
print(f"\nholdout band net across seeds: min {min(hold_nets):+,.0f} / median "
      f"{sorted(hold_nets)[3]:+,.0f} / max {max(hold_nets):+,.0f}")
# deploy candidate: median VAL-book seed (selection on val only)
sel = sorted(rows, key=lambda r: r[2])[len(rows) // 2][0]
print(f"deploy candidate = seed {sel} (median val net)")
with open(OUT, "wb") as f:
    pickle.dump({"model": models[sel], "features": FEATS,
                 "seed": sel, "train_end": "2026-08-01",
                 "val": "2026-08-01..10", "holdout": "2026-08-10+"}, f)
print(f"saved {OUT.name}")
