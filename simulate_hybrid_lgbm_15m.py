#!/usr/bin/env python3
"""
simulate_hybrid_lgbm_15m.py

Compare three configurations on paper_trades_btc15m.csv (flat $10/trade):
  A) Current split:   YES=zdrift-empirical,   NO=leaky LGBM
  B) Leaky symmetric: YES=leaky LGBM,         NO=leaky LGBM
  C) Hybrid branch:   YES=reform LGBM (signal-only), NO=leaky LGBM

All rows must be resolved (resolved_yes in {0,1}).
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

EDGE_THRESH = 0.04
BET_SIZE    = 10.0
RESULTS_DIR = Path("results")

# ── Load data ────────────────────────────────────────────────────────────────

df = pd.read_csv(RESULTS_DIR / "paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)

for c in ["spot", "floor_strike", "offset_pct", "tau_minutes", "p_market",
          "p_model_15m", "realized_vol_annual"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["spot", "floor_strike", "offset_pct", "tau_minutes",
                        "p_market", "realized_vol_annual"]).reset_index(drop=True)
print(f"Rows: {len(df)}  YES rate: {df['resolved_yes'].mean():.1%}")

# ── Load models ──────────────────────────────────────────────────────────────

with open("reform_results/btc_lgbm_15m.pkl", "rb") as f:
    reform_obj = pickle.load(f)
reform_clf      = reform_obj["clf"]
reform_features = reform_obj["features"]
reform_platt    = reform_obj.get("platt")

with open("models/lgbm_15m_btc.pkl", "rb") as f:
    leaky_clf = pickle.load(f)  # CalibratedClassifierCV

LEAKY_FEATURES = [
    "offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m",
    "stoch_k_15m", "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m",
    "vol_ratio_5m", "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
    "consec_dir_1h", "vol_ratio_1h", "realized_vol_annual",
]

# ── Compute derived leaky features ──────────────────────────────────────────

MINS_PER_YEAR = 252 * 390
df["sigma_tau"] = df["realized_vol_annual"] * np.sqrt(df["tau_minutes"] / MINS_PER_YEAR)
df["z_score"]   = np.log(df["floor_strike"] / df["spot"]) / df["sigma_tau"].replace(0, np.nan)
df["body_5m"]   = df.get("body_5m", pd.Series(0.0, index=df.index)).fillna(0.0)
df["dir_5m"]    = df.get("dir_5m",  pd.Series(0.0, index=df.index)).fillna(0.0)

# ── Build feature matrices ───────────────────────────────────────────────────

def fill_features(df, features):
    X = pd.DataFrame(index=df.index)
    for f in features:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
        else:
            X[f] = 0.0
    return X

X_reform = fill_features(df, reform_features)
X_leaky  = fill_features(df, LEAKY_FEATURES)

# ── Score models ─────────────────────────────────────────────────────────────

# Reform: dict with clf (LGBMClassifier) + optional Platt scaler
p_reform_raw = reform_clf.predict_proba(X_reform)[:, 1]
if reform_platt is not None:
    p_reform = reform_platt.predict_proba(p_reform_raw.reshape(-1, 1))[:, 1]
else:
    p_reform = p_reform_raw

# Leaky: CalibratedClassifierCV
p_leaky = leaky_clf.predict_proba(X_leaky)[:, 1]

df["p_reform"] = np.clip(p_reform, 0.01, 0.99)
df["p_leaky"]  = np.clip(p_leaky,  0.01, 0.99)

# ── P&L helpers ──────────────────────────────────────────────────────────────

def pnl_yes(pm, resolved): return BET_SIZE * (1 / pm - 1) if resolved == 1 else -BET_SIZE
def pnl_no(pm, resolved):  return BET_SIZE * (1 / (1 - pm) - 1) if resolved == 0 else -BET_SIZE


def simulate(df, p_yes_col, p_no_col, label):
    """Use p_yes_col for YES edge, p_no_col for NO edge."""
    trades = []
    for _, row in df.iterrows():
        pm       = float(row["p_market"])
        p_yes_m  = float(row[p_yes_col])
        p_no_m   = float(row[p_no_col])
        resolved = int(row["resolved_yes"])
        orig_side = str(row.get("side", "")).upper()

        edge_yes = p_yes_m - pm
        edge_no  = (1 - p_no_m) - (1 - pm)  # = pm - p_no_m

        if edge_yes >= EDGE_THRESH:
            side = "YES"
            pnl  = pnl_yes(pm, resolved)
        elif edge_no >= EDGE_THRESH:
            side = "NO"
            pnl  = pnl_no(pm, resolved)
        else:
            side = "pass"
            pnl  = 0.0

        trades.append({"side": side, "pnl": pnl, "resolved": resolved,
                       "pm": pm, "p_yes": p_yes_m, "p_no": p_no_m})

    tdf = pd.DataFrame(trades)
    return tdf


def report(tdf, label):
    acted = tdf[tdf["side"] != "pass"]
    yes_t = tdf[tdf["side"] == "YES"]
    no_t  = tdf[tdf["side"] == "NO"]

    def stats(sub, side_name):
        if len(sub) == 0:
            return f"  {side_name}: 0 trades"
        wins   = (sub["pnl"] > 0).sum()
        losses = (sub["pnl"] < 0).sum()
        wr     = wins / len(sub)
        be_wr  = sub["pm"].mean() if side_name == "YES" else (1 - sub["pm"]).mean()
        pnl    = sub["pnl"].sum()
        return (f"  {side_name}: n={len(sub):3d}  WR={wr:.1%}  "
                f"BE={be_wr:.1%}  PnL=${pnl:+.0f}")

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(stats(yes_t, "YES"))
    print(stats(no_t,  "NO"))
    total_pnl = acted["pnl"].sum()
    total_trades = len(acted)
    print(f"  TOTAL: n={total_trades}  PnL=${total_pnl:+.0f}")


# ── Configuration A: current split (zdrift YES, leaky NO) ───────────────────
# p_model_15m column = the actual scored value at decision time (zdrift for YES, leaky for NO)
# We replicate by using p_model_15m as-is for both sides, which is what the runner actually did.
tdf_a = simulate(df, "p_model_15m", "p_model_15m", "A: Current (zdrift-YES / leaky-NO)")
report(tdf_a, "A: Current split (zdrift-YES / leaky-NO)")

# ── Configuration B: leaky symmetric ────────────────────────────────────────
tdf_b = simulate(df, "p_leaky", "p_leaky", "B: Leaky symmetric")
report(tdf_b, "B: Leaky LGBM symmetric (YES + NO)")

# ── Configuration C: hybrid branch ──────────────────────────────────────────
tdf_c = simulate(df, "p_reform", "p_leaky", "C: Hybrid (reform-YES / leaky-NO)")
report(tdf_c, "C: Hybrid branch (reform-YES / leaky-NO)")

# ── Breakdown: where C differs from A ───────────────────────────────────────
print(f"\n{'═'*60}")
print("  Decision differences  A vs C")
print(f"{'─'*60}")
diff = pd.DataFrame({
    "side_a": tdf_a["side"], "side_c": tdf_c["side"],
    "pnl_a": tdf_a["pnl"],  "pnl_c": tdf_c["pnl"],
    "resolved": tdf_a["resolved"], "pm": tdf_a["pm"],
})
changed = diff[diff["side_a"] != diff["side_c"]]
print(f"  Rows where decision changes: {len(changed)}")
for a_side, c_side in [("YES", "pass"), ("pass", "YES"), ("NO", "pass"), ("pass", "NO"),
                        ("YES", "NO"), ("NO", "YES")]:
    sub = changed[(changed["side_a"] == a_side) & (changed["side_c"] == c_side)]
    if len(sub) == 0:
        continue
    wins_c = (sub["pnl_c"] > 0).sum()
    wins_a = (sub["pnl_a"] > 0).sum()
    print(f"  A={a_side:4s} → C={c_side:4s}:  n={len(sub):3d}  "
          f"PnL_A=${sub['pnl_a'].sum():+.0f}  PnL_C=${sub['pnl_c'].sum():+.0f}")

# ── Reform model calibration on YES trades ──────────────────────────────────
print(f"\n{'─'*60}")
print("  Reform model calibration (all rows):")
buckets = np.arange(0.0, 1.01, 0.1)
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = (df["p_reform"] >= lo) & (df["p_reform"] < hi)
    n = mask.sum()
    if n < 10:
        continue
    actual = df.loc[mask, "resolved_yes"].mean()
    pred   = df.loc[mask, "p_reform"].mean()
    print(f"    [{lo:.1f},{hi:.1f})  n={n:3d}  pred={pred:.2f}  actual={actual:.2f}  Δ={actual-pred:+.2f}")
