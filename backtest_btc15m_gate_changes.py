"""
BTC 15m gate backtest — May 11–28 combined paper trade history, with HMM regime backfill.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

# ─── Step 1: Load and combine paper trades ────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading paper trades")
print("=" * 70)

arch = pd.read_csv(BASE / "results/paper_trades_btc15m_archive_20260525_1432_pre_branched_drift.csv")
cur  = pd.read_csv(BASE / "results/paper_trades_btc15m.csv")

print(f"Archive rows: {len(arch)},  Current rows: {len(cur)}")

df = pd.concat([arch, cur], ignore_index=True)

# Drop duplicates — keep last (more recent file wins)
df = df.drop_duplicates(subset=["contract_ticker", "logged_at", "side"], keep="last")

# Filter
if "asset" in df.columns:
    df = df[df["asset"].str.upper() == "BTC"]

df = df[df["decision"] == "trade"]
df = df[df["resolved_yes"].notna()]

# Parse timestamps — drop rows with null logged_at (26 rows have no timestamp)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
n_before = len(df)
df = df.dropna(subset=["logged_at"])
n_dropped = n_before - len(df)
df = df.sort_values("logged_at").reset_index(drop=True)

# Computed fields
df["fee"] = 0.07 * df["p_market"].clip(0, 1).apply(lambda p: min(p, 1 - p))

# actual PnL — use would_pnl directly
# (already computed in CSV)

total = len(df)
date_min = df["logged_at"].min()
date_max = df["logged_at"].max()
yes_df = df[df["side"] == "yes"]
no_df  = df[df["side"] == "no"]

print(f"\nCOMBINED DATASET: {total} trades ({n_dropped} dropped — null logged_at), {date_min.date()} to {date_max.date()}")
print(f"  Overall WR:  {(df['would_pnl'] > 0).mean():.1%}")
print(f"  Total PnL:   ${df['would_pnl'].sum():.2f}")
print(f"\n  YES side: {len(yes_df)} trades | WR {(yes_df['would_pnl']>0).mean():.1%} | PnL ${yes_df['would_pnl'].sum():.2f}")
print(f"  NO  side: {len(no_df)} trades | WR {(no_df['would_pnl']>0).mean():.1%} | PnL ${no_df['would_pnl'].sum():.2f}")

# ─── Step 2: Backfill HMM regime labels ───────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Backfilling HMM regime labels")
print("=" * 70)

def load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)

hmm_1h_pkg  = load_pkl(BASE / "results/hmm_3state_btc_1h.pkl")
hmm_15m_pkg = load_pkl(BASE / "results/hmm_3state_btc_15m.pkl")
hmm_1h_model  = hmm_1h_pkg["model"]
hmm_15m_model = hmm_15m_pkg["model"]
state_to_name_1h  = hmm_1h_pkg["state_to_name"]
state_to_name_15m = hmm_15m_pkg["state_to_name"]
feat_cols_1h  = hmm_1h_pkg["feature_cols"]   # ['log_ret', 'realized_vol', 'ret_5bar']
feat_cols_15m = hmm_15m_pkg["feature_cols"]
print("HMM models loaded.")
print(f"  1h state→name: {state_to_name_1h}")
print(f"  15m state→name: {state_to_name_15m}")
print(f"  1h feature cols: {feat_cols_1h}")
print(f"  15m feature cols: {feat_cols_15m}")

# Load 1h data
print("Loading 1h Binance parquet...")
df_1h = pd.read_parquet(BASE / "data/binanceus_BTCUSDT_1h_1970-01-01_2026-05-28.parquet")
# Normalise index / timestamp column
if df_1h.index.name in ["timestamp", "open_time"]:
    df_1h = df_1h.reset_index()
for col in ["timestamp", "open_time", "time"]:
    if col in df_1h.columns:
        df_1h["ts"] = pd.to_datetime(df_1h[col], utc=True)
        break
df_1h = df_1h.sort_values("ts").set_index("ts")

log_ret_1h    = np.log(df_1h["close"] / df_1h["close"].shift(1))
rvol_1h       = log_ret_1h.rolling(20, min_periods=10).std()
ret5_1h       = np.log(df_1h["close"] / df_1h["close"].shift(5))
feat_1h       = pd.DataFrame({"log_ret": log_ret_1h, "realized_vol": rvol_1h, "ret_5bar": ret5_1h}).dropna()
# Align column order to model
feat_1h = feat_1h[feat_cols_1h]

print(f"  1h features: {len(feat_1h)} rows, {feat_1h.index.min()} – {feat_1h.index.max()}")
pred_1h = hmm_1h_model.predict(feat_1h.values)

# Use stored state_to_name mapping
regime_1h = pd.Series([state_to_name_1h[s] for s in pred_1h], index=feat_1h.index, name="hmm_1h")
print(f"  1h regime distribution: {dict(regime_1h.value_counts())}")

# Load 1m data, resample to 15m
print("Loading 1m Binance parquet and resampling to 15m...")
df_1m = pd.read_parquet(BASE / "data/binanceus_BTCUSDT_1m_1970-01-01_2026-05-28.parquet")
if df_1m.index.name in ["timestamp", "open_time"]:
    df_1m = df_1m.reset_index()
for col in ["timestamp", "open_time", "time"]:
    if col in df_1m.columns:
        df_1m["ts"] = pd.to_datetime(df_1m[col], utc=True)
        break
df_1m = df_1m.sort_values("ts").set_index("ts")
df_15m = df_1m["close"].resample("15T").last().dropna().to_frame()

log_ret_15m   = np.log(df_15m["close"] / df_15m["close"].shift(1))
rvol_15m      = log_ret_15m.rolling(20, min_periods=10).std()
ret5_15m      = np.log(df_15m["close"] / df_15m["close"].shift(5))
feat_15m      = pd.DataFrame({"log_ret": log_ret_15m, "realized_vol": rvol_15m, "ret_5bar": ret5_15m}).dropna()
feat_15m      = feat_15m[feat_cols_15m]

print(f"  15m features: {len(feat_15m)} rows, {feat_15m.index.min()} – {feat_15m.index.max()}")
pred_15m = hmm_15m_model.predict(feat_15m.values)

# Use stored state_to_name mapping
regime_15m = pd.Series([state_to_name_15m[s] for s in pred_15m], index=feat_15m.index, name="hmm_15m")
print(f"  15m regime distribution: {dict(regime_15m.value_counts())}")

# merge_asof to join HMM labels to paper trades
# Drop rows where logged_at is null (shouldn't happen, but guard)
df = df.dropna(subset=["logged_at"])
df_sorted = df.sort_values("logged_at").reset_index(drop=True)

r1h_df  = regime_1h.reset_index().rename(columns={"ts": "logged_at"}).sort_values("logged_at")
r15m_df = regime_15m.reset_index().rename(columns={"ts": "logged_at"}).sort_values("logged_at")

df_sorted = pd.merge_asof(df_sorted, r1h_df,  on="logged_at", direction="backward")
df_sorted = pd.merge_asof(df_sorted, r15m_df, on="logged_at", direction="backward")

print(f"\n  hmm_1h distribution:\n{df_sorted['hmm_1h'].value_counts().to_string()}")
print(f"\n  hmm_15m distribution:\n{df_sorted['hmm_15m'].value_counts().to_string()}")

df = df_sorted  # use this going forward

# ─── Step 3: Simulate scenarios ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — Scenario simulations")
print("=" * 70)

baseline_pnl = df["would_pnl"].sum()
baseline_wr  = (df["would_pnl"] > 0).mean()
baseline_n   = len(df)

def scenario_result(label, mask_block):
    """
    mask_block: boolean Series, True = block this trade (remove its PnL).
    Returns (n_trades, wr, pnl, delta, wins_blocked, losses_blocked)
    """
    blocked = df[mask_block]
    kept    = df[~mask_block]
    wins_blocked   = (blocked["would_pnl"] > 0).sum()
    losses_blocked = (blocked["would_pnl"] <= 0).sum()
    pnl   = kept["would_pnl"].sum()
    delta = pnl - baseline_pnl
    wr    = (kept["would_pnl"] > 0).mean() if len(kept) > 0 else float("nan")
    return len(kept), wr, pnl, delta, int(wins_blocked), int(losses_blocked)

# Gate A condition
gate_a_base = (df["side"] == "no") & (df["stoch_k_15m"] >= 80) & (df["ema_bias"] == 1)

rescue_a    = (df["bp_15m"] < 0.35) & (df["dir_15m"] == -1)

# Scenario 0 — baseline
s0_n, s0_wr, s0_pnl = baseline_n, baseline_wr, baseline_pnl

# Scenario 1 — Gate A no rescue
mask_s1 = gate_a_base
s1 = scenario_result("Gate A only", mask_s1)

# Scenario 2 — Gate A with rescue
mask_s2 = gate_a_base & ~rescue_a
s2 = scenario_result("Gate A + rescue", mask_s2)

# Scenario 3 — HMM regime gates only
# Block YES where hmm_1h == "Bear"
hmm_bear_1h_block  = (df["side"] == "yes") & (df["hmm_1h"] == "Bear")
# Block YES where hmm_15m == "Bear" AND composite_p_up > 0.488
hmm_bear_15m_block = (df["side"] == "yes") & (df["hmm_15m"] == "Bear") & (df["composite_p_up"] > 0.488)
mask_s3_new = hmm_bear_1h_block | hmm_bear_15m_block
s3_new = scenario_result("HMM new labels", mask_s3_new)

# Same gates with OLD stored markov labels
# markov_regime_1h / markov_regime_15m — check column values
print(f"\n  Old markov_regime_1h distribution:\n{df['markov_regime_1h'].value_counts().to_string()}")
print(f"\n  Old markov_regime_15m distribution:\n{df['markov_regime_15m'].value_counts().to_string()}")

old_bear_1h_block  = (df["side"] == "yes") & (df["markov_regime_1h"].str.lower() == "bear")
old_bear_15m_block = (df["side"] == "yes") & (df["markov_regime_15m"].str.lower() == "bear") & (df["composite_p_up"] > 0.488)
mask_s3_old = old_bear_1h_block | old_bear_15m_block
s3_old = scenario_result("HMM old labels", mask_s3_old)

# Scenario 4 — Gate A rescue + HMM combined
mask_s4 = mask_s2 | mask_s3_new
s4 = scenario_result("Gate A+rescue + HMM", mask_s4)

# Print scenario table
print("\n" + "-" * 90)
print(f"{'Scenario':<30} {'N_kept':>7} {'WR':>7} {'PnL':>10} {'Delta':>10} {'W_blk':>7} {'L_blk':>7}")
print("-" * 90)
print(f"{'Scenario 0 — Baseline':<30} {s0_n:>7} {s0_wr:>7.1%} {s0_pnl:>10.2f} {'—':>10} {'—':>7} {'—':>7}")
for label, res in [
    ("Scenario 1 — Gate A only",      s1),
    ("Scenario 2 — Gate A+rescue",    s2),
    ("Scenario 3 — HMM new",          s3_new),
    ("Scenario 3b — HMM old",         s3_old),
    ("Scenario 4 — GateA+HMM combo",  s4),
]:
    n, wr, pnl, delta, wb, lb = res
    print(f"  {label:<28} {n:>7} {wr:>7.1%} {pnl:>10.2f} {delta:>+10.2f} {wb:>7} {lb:>7}")
print("-" * 90)

# ─── Step 4: Weekly breakdown ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Weekly breakdown (Baseline vs Gate A+rescue)")
print("=" * 70)

weeks = [
    ("May 11–17", pd.Timestamp("2026-05-11", tz="UTC"), pd.Timestamp("2026-05-18", tz="UTC")),
    ("May 18–24", pd.Timestamp("2026-05-18", tz="UTC"), pd.Timestamp("2026-05-25", tz="UTC")),
    ("May 25–28", pd.Timestamp("2026-05-25", tz="UTC"), pd.Timestamp("2026-05-29", tz="UTC")),
]

print(f"\n{'Week':<12} {'Base_N':>7} {'Base_PnL':>10} {'GateA_N':>8} {'GateA_PnL':>10} {'Delta':>10}")
print("-" * 62)
for wname, wstart, wend in weeks:
    mask_week = (df["logged_at"] >= wstart) & (df["logged_at"] < wend)
    wdf = df[mask_week]
    if len(wdf) == 0:
        print(f"{wname:<12} {'—':>7}")
        continue
    # Baseline
    b_pnl = wdf["would_pnl"].sum()
    b_n   = len(wdf)
    # Gate A rescue applied
    wg_a = gate_a_base[mask_week] & ~rescue_a[mask_week]
    wkept = wdf[~wg_a]
    g_pnl = wkept["would_pnl"].sum()
    g_n   = len(wkept)
    delta = g_pnl - b_pnl
    print(f"  {wname:<10} {b_n:>7} {b_pnl:>10.2f} {g_n:>8} {g_pnl:>10.2f} {delta:>+10.2f}")

# ─── HMM regime comparison detail ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("HMM REGIME COMPARISON — Old vs New labels")
print("=" * 70)

# YES trades only for regime gate comparison
yes_trades = df[df["side"] == "yes"].copy()

old_bear_flag = (yes_trades["markov_regime_1h"].str.lower() == "bear") | \
                ((yes_trades["markov_regime_15m"].str.lower() == "bear") & (yes_trades["composite_p_up"] > 0.488))
new_bear_flag = (yes_trades["hmm_1h"] == "Bear") | \
                ((yes_trades["hmm_15m"] == "Bear") & (yes_trades["composite_p_up"] > 0.488))

newly_blocked = yes_trades[ new_bear_flag & ~old_bear_flag]
released      = yes_trades[~new_bear_flag &  old_bear_flag]
both_block    = yes_trades[ new_bear_flag &  old_bear_flag]
neither       = yes_trades[~new_bear_flag & ~old_bear_flag]

print(f"\n  YES trades total: {len(yes_trades)}")
print(f"  Old labels distribution — Bear: {old_bear_flag.sum()}, Non-Bear: {(~old_bear_flag).sum()}")
print(f"  New HMM distribution  — Bear: {new_bear_flag.sum()}, Non-Bear: {(~new_bear_flag).sum()}")
print(f"\n  Newly blocked (HMM catches, old didn't): {len(newly_blocked)} trades")
if len(newly_blocked):
    print(f"    wins would block: {(newly_blocked['would_pnl']>0).sum()}")
    print(f"    losses would block: {(newly_blocked['would_pnl']<=0).sum()}")
    print(f"    PnL of these trades: ${newly_blocked['would_pnl'].sum():.2f}  (removing = saves losses)")
print(f"\n  Released (old blocked, HMM wouldn't): {len(released)} trades")
if len(released):
    print(f"    wins that would be recovered: {(released['would_pnl']>0).sum()}")
    print(f"    losses that would be kept: {(released['would_pnl']<=0).sum()}")
    print(f"    PnL of these trades: ${released['would_pnl'].sum():.2f}")
print(f"\n  Both methods block: {len(both_block)} trades | PnL: ${both_block['would_pnl'].sum():.2f}")
print(f"  Neither blocks:     {len(neither)} trades | PnL: ${neither['would_pnl'].sum():.2f}")

net_effect_switch = released["would_pnl"].sum() - newly_blocked["would_pnl"].sum()
print(f"\n  Net PnL effect of switching old→HMM labels: ${net_effect_switch:+.2f}")
print("  (positive = HMM regime labels better)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
