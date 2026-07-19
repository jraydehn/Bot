#!/usr/bin/env python3
"""
sim_btc_build15m_retrain_compare_20260719.py — Head-to-head of the new
build_15m_model.py BTC retrain vs the prior live model, evaluated against
REAL Kalshi market prices and REAL resolved outcomes from
results/btc_scan_archive_15m.csv (NOT the synthetic log-normal proxy that
build_15m_model.py's own eval_pnl() uses -- that proxy is nearly the same
quantity the model is trained to predict, so its "PnL" is not a real test).

Uses the exact same z_score construction as live inference
(compute_p_model_15m in paper_trade_runner_15m.py: blended realized+implied
vol, not the training script's pure-realized-vol z), so this reflects what
the model would actually see in production.
"""
import pickle
import math
import numpy as np
import pandas as pd
from probability_engine import implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT_BY_ASSET

EDGE_THRESHOLD = 0.04
FLAT_BET = 100.0
MINS_PER_YEAR = 525600.0

FEATURE_COLS = [
    "offset_pct", "z_score",
    "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m", "vol_ratio_5m",
    "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h", "consec_dir_1h",
    "vol_ratio_1h", "realized_vol_annual",
]

df = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df = df.drop_duplicates(subset=["contract_ticker"], keep="first").copy()
df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", format="mixed", utc=True)
df = df.reset_index(drop=True)  # natural file order is chronological

n = len(df)
i_test = int(n * 0.80)
df_te = df.iloc[i_test:].copy()
_ts = df_te["logged_at"].dropna()
print(f"Held-out window (last 20% of scan archive, real Kalshi data): n={len(df_te)}  "
      f"({_ts.iloc[0] if len(_ts) else '?'} → {_ts.iloc[-1] if len(_ts) else '?'})")

# ── Build inference features exactly as compute_p_model_15m does ────────────
def build_feats(row):
    spot = float(row["spot"]); strike = float(row["strike"])
    tau_min = float(row["tau_minutes"]); pm = float(row["p_market"])
    rv_ann = row.get("realized_vol_annual")
    rv_ann = float(rv_ann) if pd.notna(rv_ann) else 0.3
    vol_realized = rv_ann / math.sqrt(MINS_PER_YEAR)
    try:
        vol_imp = implied_vol_from_price(pm, spot, strike, tau_min)
    except Exception:
        vol_imp = vol_realized
    weight = REALIZED_VOL_WEIGHT_BY_ASSET.get("BTC", 0.35)
    vol_eff = blend_vol(vol_realized, vol_imp, weight=weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z = math.log(strike / spot) / sigma_tau if spot > 0 else 0.0
    offset_pct = (strike / spot - 1.0) * 100.0 if spot > 0 else 0.0

    def g(col, default):
        v = row.get(col)
        return float(v) if pd.notna(v) else default

    return {
        "offset_pct": offset_pct, "z_score": z,
        "bp_15m": g("bp_15m", 0.5), "body_15m": g("body_15m", 0.0),
        "dir_15m": g("dir_15m", 0.0), "chg_15m": g("chg_15m", 0.0),
        "stoch_k_15m": g("stoch_k_15m", 50.0),
        "bp_5m": g("bp_5m", 0.5), "body_5m": 0.0, "dir_5m": 0.0,  # not logged separately in scan archive
        "chg_5m": g("chg_5m", 0.0), "stoch_k_5m": g("stoch_k_5m", 50.0),
        "vol_ratio_5m": g("vol_ratio_5m", 1.0),
        "chg_1h": g("chg_1h", 0.0), "bp_1h": g("bp_1h", 0.5),
        "stoch_k_1h": g("stoch_k_1h", 50.0), "ema_bias_1h": g("ema_bias_1h", 0.0),
        "consec_dir_1h": g("consec_dir_1h", 0.0), "vol_ratio_1h": g("vol_ratio_1h", 1.0),
        "realized_vol_annual": rv_ann,
    }

feat_rows = df_te.apply(build_feats, axis=1)
X_te = pd.DataFrame(list(feat_rows))[FEATURE_COLS]

with open("models/lgbm_15m_btc_pre_retrain_20260719.pkl", "rb") as f:
    old_model = pickle.load(f)
with open("models/lgbm_15m_btc.pkl", "rb") as f:
    new_model = pickle.load(f)

p_old = old_model.predict_proba(X_te)[:, 1]
p_new = new_model.predict_proba(X_te)[:, 1]

pm = df_te["p_market"].values
res = df_te["resolved_yes"].values


def pick_and_pnl(p_yes):
    edge_yes = p_yes - pm
    edge_no = pm - p_yes
    side = np.where(edge_yes >= edge_no, "yes", "no")
    edge = np.where(side == "yes", edge_yes, edge_no)
    trade = edge >= EDGE_THRESHOLD
    pnl = np.where(
        side == "yes",
        np.where(res == 1, FLAT_BET * (1 / np.clip(pm, 1e-6, None) - 1), -FLAT_BET),
        np.where(res == 0, FLAT_BET * (1 / np.clip(1 - pm, 1e-6, None) - 1), -FLAT_BET),
    )
    pnl = np.where(trade, pnl, 0.0)
    win = trade & (((side == "yes") & (res == 1)) | ((side == "no") & (res == 0)))
    loss = trade & ~win
    be = np.where(side == "yes", pm, 1 - pm)
    return trade, side, pnl, win, loss, be


tr_old, side_old, pnl_old, win_old, loss_old, be_old = pick_and_pnl(p_old)
tr_new, side_new, pnl_new, win_new, loss_new, be_new = pick_and_pnl(p_new)

print()
print("=" * 70)
print("OLD model (live pre-retrain, models/lgbm_15m_btc_pre_retrain_20260719.pkl):")
print(f"  Trades: {tr_old.sum()}  Wins: {win_old.sum()}  Losses: {loss_old.sum()}  "
      f"WR: {100*win_old[tr_old].sum()/max(tr_old.sum(),1):.1f}%  "
      f"avg BE: {100*be_old[tr_old].mean():.1f}%  Net PnL: ${pnl_old.sum():+.2f}")

print()
print("NEW model (build_15m_model.py BTC retrain, 2yr price data through 07-19):")
print(f"  Trades: {tr_new.sum()}  Wins: {win_new.sum()}  Losses: {loss_new.sum()}  "
      f"WR: {100*win_new[tr_new].sum()/max(tr_new.sum(),1):.1f}%  "
      f"avg BE: {100*be_new[tr_new].mean():.1f}%  Net PnL: ${pnl_new.sum():+.2f}")

print()
print("=" * 70)
both_trade = tr_old & tr_new
only_old = tr_old & ~tr_new
only_new = tr_new & ~tr_old
side_agree = both_trade & (side_old == side_new)
side_disagree = both_trade & (side_old != side_new)

print(f"  Both traded, same side:  n={side_agree.sum():4d}  "
      f"old PnL=${pnl_old[side_agree].sum():+.2f}  new PnL=${pnl_new[side_agree].sum():+.2f}")
print(f"  Both traded, opp side:   n={side_disagree.sum():4d}  "
      f"old PnL=${pnl_old[side_disagree].sum():+.2f}  new PnL=${pnl_new[side_disagree].sum():+.2f}")
print(f"  Only OLD traded: n={only_old.sum():4d}  old PnL=${pnl_old[only_old].sum():+.2f}")
print(f"  Only NEW traded: n={only_new.sum():4d}  new PnL=${pnl_new[only_new].sum():+.2f}")
print()
print(f"  Net $ delta (new - old): ${pnl_new.sum() - pnl_old.sum():+.2f}")
print(f"  Wins   old={win_old.sum()}  new={win_new.sum()}  delta={win_new.sum()-win_old.sum():+d}")
print(f"  Losses old={loss_old.sum()}  new={loss_new.sum()}  delta={loss_new.sum()-loss_old.sum():+d}")
