#!/usr/bin/env python3
"""
sim_btc_lgbm_retrain_compare_20260719.py — Head-to-head comparison of the new
BTC 15m LGBM retrain (reform_results/btc_lgbm_15m.pkl) vs the currently-live
model (models/lgbm_15m_btc.pkl), on the SAME chronological held-out test split
used by train_btc_lgbm_15m.py (last 20% of results/btc_scan_archive_15m.csv,
first-occurrence-per-ticker deduped).

Old model's p_model_yes was recorded live in the scan archive at eval time
(zero lookahead by construction). New model's p_model_yes is generated fresh
here from the same feature columns. Both use the same EDGE_THRESHOLD=0.04 and
flat $100/contract bet sizing (feedback_flat_bankroll_backtest) so the two
picks are directly comparable.
"""
import pickle
import numpy as np
import pandas as pd

EDGE_THRESHOLD = 0.04
FLAT_BET = 100.0

FEATURES = [
    "body_15m", "dir_15m", "bp_5m", "bp_1h",
    "stoch_k_5m", "stoch_k_15m", "stoch_k_1h",
    "chg_1m", "chg_5m", "chg_15m", "chg_1h",
    "vwap_dist", "vol_ratio", "ema_bias",
    "consec_dir_1h", "dir_1h", "donchian_breakout_1h", "engulfing_1h",
    "stoch_cross_1h", "realized_vol_annual", "composite_p_up",
    "liq_score", "liq_bias", "oi_chg_pct",
]

df = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df = df.drop_duplicates(subset=["contract_ticker"], keep="first").copy()
for c in FEATURES:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", format="mixed", utc=True)
df = df.reset_index(drop=True)  # natural CSV row order is already chronological (verified 0 inversions)

n = len(df)
i_test = int(n * 0.80)
df_te = df.iloc[i_test:].copy()
_ts = df_te["logged_at"].dropna()
print(f"Test split: n={len(df_te)}  "
      f"({_ts.iloc[0] if len(_ts) else '?'} → {_ts.iloc[-1] if len(_ts) else '?'})")

# New model predictions
with open("reform_results/btc_lgbm_15m.pkl", "rb") as f:
    new_pipe = pickle.load(f)
X_te = df_te[new_pipe["features"]].values.astype(float)
p_new_raw = new_pipe["clf"].predict_proba(X_te)[:, 1]
logits = np.log(np.clip(p_new_raw, 1e-6, 1-1e-6) / np.clip(1-p_new_raw, 1e-6, 1-1e-6))
p_new = new_pipe["platt"].predict_proba(logits.reshape(-1, 1))[:, 1]
df_te["p_model_yes_new"] = p_new

# Old model predictions: already recorded live in the archive (p_model_yes)
df_te["p_model_yes_old"] = pd.to_numeric(df_te["p_model_yes"], errors="coerce")

pm = df_te["p_market"].values
res = df_te["resolved_yes"].values


def pick_and_pnl(p_yes_col):
    p_yes = df_te[p_yes_col].values
    edge_yes = p_yes - pm
    edge_no = pm - p_yes  # [2026-07-19 fix] was (1-pm)-(1-p_yes), which algebraically
    # equals p_yes-pm == edge_yes, not pm-p_yes -- silently forced side="yes" on every
    # near-tie via the >= comparison. Model's NO-side edge = P(NO)-price(NO) = (1-p_yes)-(1-pm).
    side = np.where(edge_yes >= edge_no, "yes", "no")
    edge = np.where(side == "yes", edge_yes, edge_no)
    trade = edge >= EDGE_THRESHOLD
    pnl = np.where(
        side == "yes",
        np.where(res == 1, FLAT_BET * (1 / np.clip(pm, 1e-6, None) - 1), -FLAT_BET),
        np.where(res == 0, FLAT_BET * (1 / np.clip(1 - pm, 1e-6, None) - 1), -FLAT_BET),
    )
    pnl = np.where(trade, pnl, 0.0)
    win = trade & (
        ((side == "yes") & (res == 1)) | ((side == "no") & (res == 0))
    )
    loss = trade & ~win
    return trade, side, pnl, win, loss


tr_old, side_old, pnl_old, win_old, loss_old = pick_and_pnl("p_model_yes_old")
tr_new, side_new, pnl_new, win_new, loss_new = pick_and_pnl("p_model_yes_new")

print()
print("=" * 70)
print("OLD model (currently live, trained pre-05-13):")
print(f"  Trades: {tr_old.sum()}  Wins: {win_old.sum()}  Losses: {loss_old.sum()}  "
      f"WR: {100*win_old.sum()/max(tr_old.sum(),1):.1f}%  Net PnL: ${pnl_old.sum():+.2f}")

print()
print("NEW model (retrained today on 4,693 rows through current archive):")
print(f"  Trades: {tr_new.sum()}  Wins: {win_new.sum()}  Losses: {loss_new.sum()}  "
      f"WR: {100*win_new.sum()/max(tr_new.sum(),1):.1f}%  Net PnL: ${pnl_new.sum():+.2f}")

print()
print("=" * 70)
print("Overlap / delta analysis:")
both_trade = tr_old & tr_new
only_old = tr_old & ~tr_new
only_new = tr_new & ~tr_old
side_agree = both_trade & (side_old == side_new)
side_disagree = both_trade & (side_old != side_new)

print(f"  Both traded, same side:  n={side_agree.sum():4d}  "
      f"old PnL=${pnl_old[side_agree].sum():+.2f}  new PnL=${pnl_new[side_agree].sum():+.2f}")
print(f"  Both traded, opp side:   n={side_disagree.sum():4d}  "
      f"old PnL=${pnl_old[side_disagree].sum():+.2f}  new PnL=${pnl_new[side_disagree].sum():+.2f}")
print(f"  Only OLD traded (new passed): n={only_old.sum():4d}  old PnL=${pnl_old[only_old].sum():+.2f} "
      f"(this $ is LOST if we switch to new model)")
print(f"  Only NEW traded (old passed): n={only_new.sum():4d}  new PnL=${pnl_new[only_new].sum():+.2f} "
      f"(this $ is GAINED if we switch to new model)")

print()
print(f"  Net $ delta (new - old): ${pnl_new.sum() - pnl_old.sum():+.2f}")
print(f"  Wins  old={win_old.sum()}  new={win_new.sum()}  delta={win_new.sum()-win_old.sum():+d}")
print(f"  Losses old={loss_old.sum()}  new={loss_new.sum()}  delta={loss_new.sum()-loss_old.sum():+d}")
