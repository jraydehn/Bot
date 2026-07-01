"""
Train VWAP multi-timeframe HMM, backfill states to btc_scan_archive_15m.csv,
then analyze WR/PnL per state.

Features (5 dims, all aligned to 15m bars):
  vwap_dist_1m   — (close - rolling_20bar_vwap_1m) / vwap * 100  (20 min VWAP)
  vwap_dist_5m   — same on 5m bars (100 min VWAP)
  vwap_dist_15m  — same on 15m bars (5 hr VWAP)
  vwap_vel_1m    — 1m diff of vwap_dist_1m at the last 1m bar per 15m period
  vwap_spread    — vwap_dist_1m - vwap_dist_15m  (cross-TF agreement)
"""

import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("pip install hmmlearn scikit-learn")

BASE = Path(__file__).parent
PARQUET = BASE / "data" / "binanceus_BTCUSDT_1m_1970-01-01_2026-07-01.parquet"
ARCH_PATH = BASE / "results" / "btc_scan_archive_15m.csv"
STATES_OUT = BASE / "results" / "btc_vwap_hmm_states_15m.csv"  # sidecar, not modified by runner
MODEL_OUT = BASE / "models" / "hmm_vwap_mtf_btc_15m.pkl"
FEAT_COLS = ["vwap_dist_1m", "vwap_dist_5m", "vwap_dist_15m", "vwap_vel_1m", "vwap_spread"]


# ── helpers ────────────────────────────────────────────────────────────────────

def rolling_vwap_dist(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tv = (tp * df["volume"]).rolling(n, min_periods=n).sum()
    cum_v = df["volume"].rolling(n, min_periods=n).sum()
    vwap = cum_tv / cum_v.replace(0, np.nan)
    return (df["close"] - vwap) / vwap.replace(0, np.nan) * 100


def build_feature_matrix(df1m: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 VWAP features aligned to 15m bar timestamps."""
    RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"}

    df5m = df1m.resample("5min").agg(RESAMPLE_AGG).dropna()
    df15m = df1m.resample("15min").agg(RESAMPLE_AGG).dropna()

    dist_1m = rolling_vwap_dist(df1m, 20)   # 20-min VWAP
    dist_5m = rolling_vwap_dist(df5m, 20)   # 100-min VWAP
    dist_15m = rolling_vwap_dist(df15m, 20)  # 5-hr VWAP
    vel_1m = dist_1m.diff()                  # 1m momentum of VWAP divergence

    feat = pd.DataFrame(index=df15m.index)
    feat["vwap_dist_15m"] = dist_15m
    feat["vwap_dist_5m"] = dist_5m.resample("15min").last()
    feat["vwap_dist_1m"] = dist_1m.resample("15min").last()
    feat["vwap_vel_1m"] = vel_1m.resample("15min").last()
    feat["vwap_spread"] = feat["vwap_dist_1m"] - feat["vwap_dist_15m"]
    return feat.dropna()


# ── load & build features ──────────────────────────────────────────────────────

print("Loading 1m parquet …")
df1m = pd.read_parquet(PARQUET)
df1m = df1m[df1m.index >= "2024-01-01"].sort_index()
print(f"  {len(df1m):,} 1m bars  "
      f"({df1m.index.min().date()} → {df1m.index.max().date()})")

print("Building feature matrix …")
feat = build_feature_matrix(df1m)
print(f"  {len(feat):,} 15m bars with complete features")

# ── train HMM (BIC selection) ──────────────────────────────────────────────────

scaler = StandardScaler()
X = scaler.fit_transform(feat[FEAT_COLS].values)

# Split into continuous sequences (gap > 30 min → new seq)
ts = feat.index
gaps = pd.Series(ts).diff().dt.total_seconds().fillna(0).values
split_mask = gaps > 1800
seq_starts = [0] + list(np.where(split_mask)[0])
seq_ends = seq_starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(seq_starts, seq_ends) if e - s >= 3]
valid_idx = [i for s, e in zip(seq_starts, seq_ends) if e - s >= 3 for i in range(s, e)]
X_seq = X[valid_idx]
print(f"  {len(lengths)} sequences, {len(X_seq)} observations (avg len {np.mean(lengths):.0f})")

print("\nBIC model selection:")
best_bic, best_n, best_model = np.inf, 4, None
for n in range(2, 9):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag",
                        n_iter=300, random_state=42, tol=1e-4)
        m.fit(X_seq, lengths=lengths)
        ll = m.score(X_seq, lengths=lengths)
        n_params = n * n + n * len(FEAT_COLS) * 2
        bic = -2 * ll + n_params * np.log(len(X_seq))
        print(f"  n={n}: BIC={bic:>12.1f}  LL={ll:.4f}")
        if bic < best_bic:
            best_bic, best_n, best_model = bic, n, m
    except Exception as e:
        print(f"  n={n}: FAILED — {e}")

print(f"\nSelected: {best_n} states (BIC={best_bic:.1f})")
model = best_model

# ── decode full sequence ───────────────────────────────────────────────────────

states_all = model.predict(X_seq, lengths=lengths)
feat_valid = feat.iloc[valid_idx].copy()
feat_valid["vwap_hmm_state"] = states_all

# print state profiles
print("\n=== State profiles (feature means) ===")
state_sizes = []
for s in range(best_n):
    m_sub = feat_valid[feat_valid["vwap_hmm_state"] == s]
    pct = 100 * len(m_sub) / len(feat_valid)
    state_sizes.append((s, len(m_sub)))
    means = m_sub[FEAT_COLS].mean()
    print(f"\nState {s}  n={len(m_sub):,} ({pct:.1f}%)")
    for c in FEAT_COLS:
        print(f"  {c:20s}: {means[c]:+.4f}")

# ── save model ────────────────────────────────────────────────────────────────

pkg = {"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": best_n}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nSaved model → {MODEL_OUT}")

# ── backfill scan archive ──────────────────────────────────────────────────────

def parse_logged_at_mixed(series: pd.Series) -> pd.Series:
    """Handle mix of UTC-aware ('2026-05-21 21:42:31+00:00') and
    timezone-naive ('2026-07-01 19:17:04') strings. Returns UTC Series."""
    def _to_utc(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_to_utc(v) for v in series], utc=True)

print(f"\nLoading scan archive: {ARCH_PATH}")
arch = pd.read_csv(ARCH_PATH, low_memory=False)
arch["logged_at"] = parse_logged_at_mixed(arch["logged_at"])
# Fallback: use close_ts - tau_minutes for rows with null logged_at
null_la = arch["logged_at"].isna()
if null_la.any():
    close_ts = parse_logged_at_mixed(arch.loc[null_la, "close_ts"])
    tau_min = pd.to_numeric(arch.loc[null_la, "tau_minutes"], errors="coerce").fillna(7)
    arch.loc[null_la, "logged_at"] = close_ts - pd.to_timedelta(tau_min, unit="m")
valid_ts = arch["logged_at"].notna()
print(f"  {len(arch):,} rows  valid_ts={valid_ts.sum():,}  "
      f"({arch.loc[valid_ts, 'logged_at'].min().date()} → "
      f"{arch.loc[valid_ts, 'logged_at'].max().date()})")

# Build sidecar: logged_at + contract_ticker + vwap_hmm_state
state_ts = feat_valid[["vwap_hmm_state"]].reset_index()
state_ts.columns = ["bar_ts", "vwap_hmm_state"]
# Ensure bar_ts is UTC-aware to match arch_valid["logged_at"]
if state_ts["bar_ts"].dt.tz is None:
    state_ts["bar_ts"] = state_ts["bar_ts"].dt.tz_localize("UTC")
else:
    state_ts["bar_ts"] = state_ts["bar_ts"].dt.tz_convert("UTC")
state_ts = state_ts.sort_values("bar_ts")

arch_valid = arch[valid_ts].sort_values("logged_at").reset_index(drop=True)
sidecar = pd.merge_asof(
    arch_valid[["logged_at", "contract_ticker"]],
    state_ts,
    left_on="logged_at",
    right_on="bar_ts",
    direction="backward",
).drop(columns=["bar_ts"])

null_states = sidecar["vwap_hmm_state"].isna().sum()
print(f"  States assigned: {len(sidecar) - null_states:,}  null: {null_states}")
sidecar.to_csv(STATES_OUT, index=False)
print(f"  Sidecar → {STATES_OUT}  (main archive unchanged)")

# Join for analysis
arch_valid = arch_valid.merge(
    sidecar[["logged_at", "contract_ticker", "vwap_hmm_state"]],
    on=["logged_at", "contract_ticker"], how="left")

# ── analysis ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EFFICACY ANALYSIS — resolved rows only")
print("=" * 60)

arch_res = arch_valid[arch_valid["resolved_yes"].notna() & (arch_valid["resolved_yes"] != "")].copy()
for col in ["resolved_yes", "p_market", "p_model_yes", "p_model_no", "offset_pct"]:
    arch_res[col] = pd.to_numeric(arch_res[col], errors="coerce")
arch_res = arch_res.dropna(subset=["resolved_yes", "p_market", "vwap_hmm_state"])

# Determine model-preferred side
arch_res["edge_yes"] = arch_res["p_model_yes"] - arch_res["p_market"]
arch_res["edge_no"] = arch_res["p_model_no"] - (1 - arch_res["p_market"])
arch_res["model_side"] = np.where(
    arch_res["edge_yes"] >= arch_res["edge_no"], "YES", "NO"
)

overall_yes_wr = arch_res.loc[arch_res["model_side"] == "YES", "resolved_yes"].mean()
overall_no_wr = 1 - arch_res.loc[arch_res["model_side"] == "NO", "resolved_yes"].mean()
n_yes_total = (arch_res["model_side"] == "YES").sum()
n_no_total = (arch_res["model_side"] == "NO").sum()

print(f"\nBaseline  YES: WR={overall_yes_wr:.3f} n={n_yes_total}  "
      f"|  NO: WR={overall_no_wr:.3f} n={n_no_total}")

# Per state × side
print(f"\n{'State':<7} {'Side':<5} {'n':>6} {'WR':>6} {'BEven':>6} {'ΔWR':>7}  {'Verdict'}")
print("-" * 55)

results = []
for s in sorted(arch_res["vwap_hmm_state"].unique()):
    sub_s = arch_res[arch_res["vwap_hmm_state"] == s]
    for side, base_wr in [("YES", overall_yes_wr), ("NO", overall_no_wr)]:
        sub = sub_s[sub_s["model_side"] == side]
        if len(sub) < 10:
            continue
        if side == "YES":
            wr = sub["resolved_yes"].mean()
            be_wr = sub["p_market"].mean()  # breakeven = avg pm for YES
        else:
            wr = 1 - sub["resolved_yes"].mean()
            be_wr = 1 - sub["p_market"].mean()  # breakeven = avg (1-pm) for NO
        delta = wr - base_wr
        edge = wr - be_wr
        if edge < -0.05:
            verdict = "BLOCK ✗"
        elif edge > 0.05 and delta > 0.05:
            verdict = "BOOST ✓"
        else:
            verdict = "neutral"
        print(f"  {s:<5} {side:<5} {len(sub):>6} {wr:>6.3f} {be_wr:>6.3f} {delta:>+7.3f}  {verdict}")
        results.append(dict(state=s, side=side, n=len(sub), wr=wr, be_wr=be_wr,
                            delta=delta, edge=edge))

# summary: top candidates
print("\n=== Top gate candidates (|ΔWR| > 0.08, n >= 20) ===")
candidates = []
for r in sorted(results, key=lambda x: abs(x["delta"]), reverse=True):
    if abs(r["delta"]) > 0.08 and r["n"] >= 20:
        action = "BLOCK" if r["edge"] < -0.05 else "BOOST"
        print(f"  State {r['state']} {r['side']:3s}: WR={r['wr']:.3f}  "
              f"BE={r['be_wr']:.3f}  edge={r['edge']:+.3f}  "
              f"n={r['n']}  → {action}")
        candidates.append(r)

# ── week-by-week consistency ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("WEEK-BY-WEEK CONSISTENCY (close_ts reference)")
print("=" * 60)

arch_res["close_ts_dt"] = parse_logged_at_mixed(arch_res["close_ts"])
arch_res["iso_week"] = (arch_res["close_ts_dt"].dt.isocalendar().year.astype(str)
                        + "-W" + arch_res["close_ts_dt"].dt.isocalendar().week
                                                         .astype(str).str.zfill(2))
all_weeks = sorted(arch_res["iso_week"].dropna().unique())

# Only run week breakdown for candidates with |ΔWR| > 0.08
for r in candidates:
    s, side = r["state"], r["side"]
    side_mask = arch_res["model_side"] == side
    sub_all = arch_res[side_mask]
    sub_state = sub_all[sub_all["vwap_hmm_state"] == s]
    print(f"\nState {s} {side} (overall ΔWR={r['delta']:+.3f}, n={r['n']}):")
    print(f"  {'Week':<9} {'n':>5} {'WR':>6} {'Base':>6} {'ΔWR':>7}")
    n_pos, n_neg, n_wks = 0, 0, 0
    for wk in all_weeks:
        wk_sub = sub_state[sub_state["iso_week"] == wk]
        wk_base = sub_all[sub_all["iso_week"] == wk]
        if len(wk_sub) < 5:
            continue
        n_wks += 1
        wr = wk_sub["resolved_yes"].mean() if side == "YES" else 1 - wk_sub["resolved_yes"].mean()
        base = wk_base["resolved_yes"].mean() if side == "YES" else 1 - wk_base["resolved_yes"].mean()
        delta_wk = wr - base
        flag = "✓" if delta_wk * r["delta"] > 0 else "✗"
        if delta_wk * r["delta"] > 0:
            n_pos += 1
        else:
            n_neg += 1
        print(f"  {wk:<9} {len(wk_sub):>5} {wr:>6.3f} {base:>6.3f} {delta_wk:>+7.3f}  {flag}")
    if n_wks > 0:
        print(f"  Directionally consistent: {n_pos}/{n_wks} weeks")

# ── MCPT ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MCPT — Monte Carlo Permutation Test (n_perm=2000)")
print("=" * 60)

np.random.seed(42)
N_PERM = 2000

def mcpt(df: pd.DataFrame, state: int, side: str) -> dict:
    sub = df[df["model_side"] == side].copy()
    get_wr = (lambda g: g["resolved_yes"].mean()) if side == "YES" \
             else (lambda g: 1 - g["resolved_yes"].mean())
    obs_wr = get_wr(sub[sub["vwap_hmm_state"] == state])
    base_wr = get_wr(sub)
    obs_delta = obs_wr - base_wr

    state_arr = sub["vwap_hmm_state"].values
    res_arr = sub["resolved_yes"].values
    n_state = (state_arr == state).sum()

    perm_deltas = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_states = np.random.permutation(state_arr)
        mask = perm_states == state
        perm_wr = res_arr[mask].mean() if side == "YES" else 1 - res_arr[mask].mean()
        perm_deltas[i] = perm_wr - base_wr

    std = perm_deltas.std()
    z = obs_delta / (std + 1e-9)
    # One-sided p-value in direction of observed effect
    p = (perm_deltas <= obs_delta).mean() if obs_delta < 0 \
        else (perm_deltas >= obs_delta).mean()
    return dict(state=state, side=side, n=n_state, obs_wr=obs_wr,
                base_wr=base_wr, delta=obs_delta, z=z, p=p)

print(f"\n{'State':<7} {'Side':<5} {'n':>5} {'WR':>6} {'ΔWR':>7} {'z':>7} {'p':>7}  Sig?")
print("-" * 55)
for r in candidates:
    res = mcpt(arch_res, r["state"], r["side"])
    sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 \
          else "*" if res["p"] < 0.05 else ""
    print(f"  {res['state']:<5} {res['side']:<5} {res['n']:>5} "
          f"{res['obs_wr']:>6.3f} {res['delta']:>+7.3f} "
          f"{res['z']:>7.2f} {res['p']:>7.4f}  {sig}")

# ── PnL simulation (Kelly-sized, matches live runner methodology) ──────────────

print("\n" + "=" * 60)
print("PnL SIMULATION — Kelly-sized, $2k bankroll, 7% fee")
print("=" * 60)

KELLY_MULT = 0.08    # matches paper_trade_runner_15m.py
KELLY_CAP  = 0.06
BANKROLL   = 2000.0
FEE_RATE   = 0.07    # 7% on min(pm, 1-pm)
MIN_EDGE   = 0.0     # include any positive-edge bet (gate analysis, not selection)

def fee_cost(pm):
    return FEE_RATE * min(float(pm), 1 - float(pm))

def kelly_pnl(pm, p_model, side, resolved_yes):
    pm = float(pm)
    if side == "YES":
        edge    = float(p_model) - pm
        pm_risk = pm
    else:
        edge    = pm - float(p_model)
        pm_risk = 1 - pm
    if edge <= MIN_EDGE or pm_risk <= 0:
        return None  # no bet
    frac = min(edge / pm_risk * KELLY_MULT, KELLY_CAP)
    bet  = frac * BANKROLL
    f    = fee_cost(pm)
    if side == "YES":
        return bet * (1 - pm - f) if resolved_yes == 1 else -bet * (pm + f)
    else:
        return bet * (pm - f) if resolved_yes == 0 else -bet * (1 - pm + f)

# Load full archive fresh, deduplicate to one row per contract (last logged)
arch_full = pd.read_csv(ARCH_PATH, low_memory=False)
arch_full["logged_at"] = parse_logged_at_mixed(arch_full["logged_at"])
null_la2 = arch_full["logged_at"].isna()
if null_la2.any():
    close_ts2 = parse_logged_at_mixed(arch_full.loc[null_la2, "close_ts"])
    tau2 = pd.to_numeric(arch_full.loc[null_la2, "tau_minutes"], errors="coerce").fillna(7)
    arch_full.loc[null_la2, "logged_at"] = close_ts2 - pd.to_timedelta(tau2, unit="m")

# Deduplicate: one row per contract (last scan = most accurate signals)
arch_dedup = (arch_full.sort_values("logged_at")
                        .groupby("contract_ticker", as_index=False)
                        .last())

# Filter to resolved
for col in ["resolved_yes", "p_market", "p_model_yes", "p_model_no"]:
    arch_dedup[col] = pd.to_numeric(arch_dedup[col], errors="coerce")
arch_dedup = arch_dedup[arch_dedup["resolved_yes"].notna()].copy()
print(f"Unique resolved contracts: {len(arch_dedup):,}")

# Join VWAP HMM states from sidecar (match by contract_ticker)
sc = pd.read_csv(STATES_OUT)
sc["vwap_hmm_state"] = pd.to_numeric(sc["vwap_hmm_state"], errors="coerce")
sc_dedup = sc.sort_values("logged_at").groupby("contract_ticker", as_index=False).last()
arch_dedup = arch_dedup.merge(
    sc_dedup[["contract_ticker", "vwap_hmm_state"]], on="contract_ticker", how="left")
has_state = arch_dedup["vwap_hmm_state"].notna().sum()
print(f"Contracts with VWAP HMM state: {has_state:,} / {len(arch_dedup):,}")

# Compute baseline Kelly PnL (best side per contract)
def best_side_pnl(row):
    ey = float(row.p_model_yes) - float(row.p_market)
    en = float(row.p_market)    - float(row.p_model_no)
    if ey >= en and ey > MIN_EDGE:
        p = kelly_pnl(row.p_market, row.p_model_yes, "YES", row.resolved_yes)
        return p, "YES"
    elif en > ey and en > MIN_EDGE:
        p = kelly_pnl(row.p_market, row.p_model_no,  "NO",  row.resolved_yes)
        return p, "NO"
    return None, None

pnl_list, side_list = zip(*[best_side_pnl(r) for r in arch_dedup.itertuples()])
arch_dedup["kelly_pnl"] = pnl_list
arch_dedup["model_side"] = side_list

taken = arch_dedup[arch_dedup["kelly_pnl"].notna()].copy()
baseline_pnl = taken["kelly_pnl"].sum()
baseline_n   = len(taken)
baseline_wr  = (taken["kelly_pnl"] > 0).mean()
print(f"Baseline  trades={baseline_n:,}  WR={baseline_wr:.1%}  PnL=${baseline_pnl:+.2f}")

# Gate simulation using dynamic block candidates from this run
block_results = [r for r in results
                 if r["n"] >= 20 and r["edge"] < -0.03 and abs(r["delta"]) > 0.05]

def build_mask(df, state_side_pairs):
    m = pd.Series(False, index=df.index)
    for s, side in state_side_pairs:
        m |= (df["vwap_hmm_state"] == s) & (df["model_side"] == side)
    return m

def sim_gate(df, block_mask, label, base_pnl):
    blocked = df[block_mask]
    kept    = df[~block_mask]
    n_blk   = len(blocked)
    w_blk   = (blocked["kelly_pnl"] > 0).sum()
    l_blk   = (blocked["kelly_pnl"] < 0).sum()
    pnl_blk = blocked["kelly_pnl"].sum()
    pnl_aft = kept["kelly_pnl"].sum()
    delta   = pnl_aft - base_pnl
    return dict(label=label, n_blk=n_blk, w_blk=w_blk, l_blk=l_blk,
                pnl_blk=pnl_blk, pnl_aft=pnl_aft, delta=delta)

print(f"\n{'Scenario':<44} {'n_blk':>5} {'W_blk':>6} {'L_blk':>6} "
      f"{'$blocked':>9} {'$after':>9} {'Δ$':>8}")
print("-" * 92)

sim_results = []
for r in sorted(block_results, key=lambda x: x["delta"]):
    mask  = build_mask(taken, [(r["state"], r["side"])])
    label = f"Block St{r['state']} {r['side']} (WR={r['wr']:.3f} BE={r['be_wr']:.3f})"
    sr    = sim_gate(taken, mask, label, baseline_pnl)
    sim_results.append(sr)
    print(f"  {sr['label']:<42} {sr['n_blk']:>5} {sr['w_blk']:>6} {sr['l_blk']:>6} "
          f"{sr['pnl_blk']:>+9.2f} {sr['pnl_aft']:>+9.2f} {sr['delta']:>+8.2f}")

if block_results:
    all_pairs  = [(r["state"], r["side"]) for r in block_results]
    combo_mask = build_mask(taken, all_pairs)
    csr        = sim_gate(taken, combo_mask,
                          f"COMBO all {len(block_results)} blocks", baseline_pnl)
    print(f"  {csr['label']:<42} {csr['n_blk']:>5} {csr['w_blk']:>6} {csr['l_blk']:>6} "
          f"{csr['pnl_blk']:>+9.2f} {csr['pnl_aft']:>+9.2f} {csr['delta']:>+8.2f}")
    print(f"\n  Baseline ${baseline_pnl:+.2f} → After ${csr['pnl_aft']:+.2f}  "
          f"(Δ={csr['delta']:+.2f},  {csr['n_blk']} blocked: "
          f"{csr['w_blk']} wins / {csr['l_blk']} losses)")

print("\nDone.")
