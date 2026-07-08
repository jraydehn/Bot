"""
Train VWAP multi-timeframe HMM for SOL 15m, mirroring
train_backfill_vwap_hmm_15m.py's exact methodology and window scale (same
contract timeframe as BTC 15m, so reusing the SAME window lengths is
justified by matching horizon -- NOT assuming BTC's specific state EFFECTS
transfer; the HMM is trained fresh on SOL's own price history and every
state's efficacy is independently discovered from scratch).

SOL 15m is currently LIVE -- this script is research/backtest only, no
wiring into the live runner without separate validation + explicit approval.

Features (5 dims, all aligned to 15m bars): identical definitions to BTC's
version, computed from SOL's own 1m price history.
  vwap_dist_1m, vwap_dist_5m, vwap_dist_15m, vwap_vel_1m, vwap_spread

Uses sol_scan_archive_15m.csv's REAL p_model_yes/p_model_no columns
directly for the efficacy analysis (avoids the strike-adjustment bug found
and fixed in the BTC-hourly build, where composite_p_up had to be manually
converted to a strike-specific probability).
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
PARQUET = sorted(BASE.glob("data/binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
ARCH_PATH = BASE / "results" / "sol_scan_archive_15m.csv"
STATES_OUT = BASE / "results" / "sol_vwap_hmm_states_15m.csv"
MODEL_OUT = BASE / "models" / "hmm_vwap_mtf_sol_15m.pkl"
FEAT_COLS = ["vwap_dist_1m", "vwap_dist_5m", "vwap_dist_15m", "vwap_vel_1m", "vwap_spread"]


def rolling_vwap_dist(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tv = (tp * df["volume"]).rolling(n, min_periods=n).sum()
    cum_v = df["volume"].rolling(n, min_periods=n).sum()
    vwap = cum_tv / cum_v.replace(0, np.nan)
    return (df["close"] - vwap) / vwap.replace(0, np.nan) * 100


def build_feature_matrix(df1m: pd.DataFrame) -> pd.DataFrame:
    RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"}
    df5m = df1m.resample("5min").agg(RESAMPLE_AGG).dropna()
    df15m = df1m.resample("15min").agg(RESAMPLE_AGG).dropna()

    dist_1m = rolling_vwap_dist(df1m, 20)
    dist_5m = rolling_vwap_dist(df5m, 20)
    dist_15m = rolling_vwap_dist(df15m, 20)
    vel_1m = dist_1m.diff()

    feat = pd.DataFrame(index=df15m.index)
    feat["vwap_dist_15m"] = dist_15m
    feat["vwap_dist_5m"] = dist_5m.resample("15min").last()
    feat["vwap_dist_1m"] = dist_1m.resample("15min").last()
    feat["vwap_vel_1m"] = vel_1m.resample("15min").last()
    feat["vwap_spread"] = feat["vwap_dist_1m"] - feat["vwap_dist_15m"]
    return feat.dropna()


print(f"Loading 1m parquet: {PARQUET.name} …")
df1m = pd.read_parquet(PARQUET)
df1m = df1m.sort_index()
print(f"  {len(df1m):,} 1m bars  ({df1m.index.min().date()} -> {df1m.index.max().date()})")

print("Building feature matrix …")
feat = build_feature_matrix(df1m)
print(f"  {len(feat):,} 15m bars with complete features")

scaler = StandardScaler()
X = scaler.fit_transform(feat[FEAT_COLS].values)

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

states_all = model.predict(X_seq, lengths=lengths)
feat_valid = feat.iloc[valid_idx].copy()
feat_valid["vwap_hmm_state"] = states_all

print("\n=== State profiles (feature means) ===")
for s in range(best_n):
    m_sub = feat_valid[feat_valid["vwap_hmm_state"] == s]
    pct = 100 * len(m_sub) / len(feat_valid)
    means = m_sub[FEAT_COLS].mean()
    print(f"\nState {s}  n={len(m_sub):,} ({pct:.1f}%)")
    for c in FEAT_COLS:
        print(f"  {c:20s}: {means[c]:+.4f}")

pkg = {"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": best_n}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nSaved model -> {MODEL_OUT}")

# ── backfill scan archive ──────────────────────────────────────────────────────

def parse_logged_at_mixed(series):
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
null_la = arch["logged_at"].isna()
if null_la.any():
    close_ts = parse_logged_at_mixed(arch.loc[null_la, "close_ts"])
    tau_min = pd.to_numeric(arch.loc[null_la, "tau_minutes"], errors="coerce").fillna(7)
    arch.loc[null_la, "logged_at"] = close_ts - pd.to_timedelta(tau_min, unit="m")
valid_ts = arch["logged_at"].notna()
print(f"  {len(arch):,} rows  valid_ts={valid_ts.sum():,}  "
      f"({arch.loc[valid_ts, 'logged_at'].min()} -> {arch.loc[valid_ts, 'logged_at'].max()})")

state_ts = feat_valid[["vwap_hmm_state"]].reset_index()
state_ts.columns = ["bar_ts", "vwap_hmm_state"]
if state_ts["bar_ts"].dt.tz is None:
    state_ts["bar_ts"] = state_ts["bar_ts"].dt.tz_localize("UTC")
else:
    state_ts["bar_ts"] = state_ts["bar_ts"].dt.tz_convert("UTC")
state_ts = state_ts.sort_values("bar_ts")

arch_valid = arch[valid_ts].sort_values("logged_at").reset_index(drop=True)
sidecar = pd.merge_asof(
    arch_valid[["logged_at", "contract_ticker"]],
    state_ts, left_on="logged_at", right_on="bar_ts", direction="backward",
).drop(columns=["bar_ts"])
null_states = sidecar["vwap_hmm_state"].isna().sum()
print(f"  States assigned: {len(sidecar) - null_states:,}  null: {null_states}")
sidecar.to_csv(STATES_OUT, index=False)
print(f"  Sidecar -> {STATES_OUT}")
print(sidecar["vwap_hmm_state"].value_counts(dropna=False).sort_index())

arch_valid = arch_valid.drop(columns=["vwap_hmm_state"], errors="ignore").merge(
    sidecar[["logged_at", "contract_ticker", "vwap_hmm_state"]],
    on=["logged_at", "contract_ticker"], how="left")

# ── analysis using REAL p_model_yes/p_model_no (no strike-adjustment bug) ───────

print("\n" + "=" * 60)
print("EFFICACY ANALYSIS — resolved rows only")
print("=" * 60)

arch_res = arch_valid[arch_valid["resolved_yes"].notna() & (arch_valid["resolved_yes"] != "")].copy()
for col in ["resolved_yes", "p_market", "p_model_yes", "p_model_no"]:
    arch_res[col] = pd.to_numeric(arch_res[col], errors="coerce")
arch_res = arch_res.dropna(subset=["resolved_yes", "p_market", "p_model_yes", "p_model_no", "vwap_hmm_state"])
print(f"usable resolved+state rows: {len(arch_res):,}")

arch_res["edge_yes"] = arch_res["p_model_yes"] - arch_res["p_market"]
arch_res["edge_no"] = arch_res["p_model_no"] - (1 - arch_res["p_market"])
arch_res["model_side"] = np.where(arch_res["edge_yes"] >= arch_res["edge_no"], "YES", "NO")

overall_yes_wr = arch_res.loc[arch_res["model_side"] == "YES", "resolved_yes"].mean()
overall_no_wr = 1 - arch_res.loc[arch_res["model_side"] == "NO", "resolved_yes"].mean()
overall_yes_be = arch_res.loc[arch_res["model_side"] == "YES", "p_market"].mean()
overall_no_be = 1 - arch_res.loc[arch_res["model_side"] == "NO", "p_market"].mean()
n_yes, n_no = (arch_res["model_side"] == "YES").sum(), (arch_res["model_side"] == "NO").sum()
print(f"\nBaseline  YES: WR={overall_yes_wr:.3f} BE={overall_yes_be:.3f} n={n_yes}  "
      f"|  NO: WR={overall_no_wr:.3f} BE={overall_no_be:.3f} n={n_no}")

print(f"\n{'State':<7} {'Side':<5} {'n':>6} {'WR':>6} {'BEven':>6} {'edge':>7} {'ΔWR':>7}  Verdict")
print("-" * 62)
results = []
for s in sorted(arch_res["vwap_hmm_state"].unique()):
    sub_s = arch_res[arch_res["vwap_hmm_state"] == s]
    for side, base_wr in [("YES", overall_yes_wr), ("NO", overall_no_wr)]:
        sub = sub_s[sub_s["model_side"] == side]
        if len(sub) < 10:
            continue
        if side == "YES":
            wr = sub["resolved_yes"].mean(); be_wr = sub["p_market"].mean()
        else:
            wr = 1 - sub["resolved_yes"].mean(); be_wr = 1 - sub["p_market"].mean()
        delta = wr - base_wr
        edge = wr - be_wr
        verdict = "BLOCK cand." if edge < -0.05 else ("BOOST cand." if edge > 0.05 and delta > 0.05 else "neutral")
        print(f"  {int(s):<5} {side:<5} {len(sub):>6} {wr:>6.3f} {be_wr:>6.3f} {edge:>+7.3f} {delta:>+7.3f}  {verdict}")
        results.append(dict(state=int(s), side=side, n=len(sub), wr=wr, be_wr=be_wr, delta=delta, edge=edge))

print("\n=== Gate candidates (|edge| > 0.05, n >= 15) ===")
candidates = [r for r in results if abs(r["edge"]) > 0.05 and r["n"] >= 15]
for r in sorted(candidates, key=lambda x: x["edge"]):
    action = "BLOCK" if r["edge"] < 0 else "BOOST"
    print(f"  State {r['state']} {r['side']:3s}: WR={r['wr']:.3f} BE={r['be_wr']:.3f} "
          f"edge={r['edge']:+.3f} ΔWR={r['delta']:+.3f} n={r['n']} -> {action}")

# ── bootstrap significance ────────────────────────────────────────────────────
print("\n=== Bootstrap significance (trade-level, n_boot=4000) ===")
rng = np.random.default_rng(11)
arch_res["be"] = np.where(arch_res["model_side"] == "YES", arch_res["p_market"], 1 - arch_res["p_market"])
arch_res["won"] = np.where(arch_res["model_side"] == "YES", arch_res["resolved_yes"] == 1,
                           arch_res["resolved_yes"] == 0)
arch_res["trade_edge"] = arch_res["won"].astype(float) - arch_res["be"]

def boot_p(edges, n_boot=4000):
    e = edges.values; n = len(e)
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()

for r in sorted(candidates, key=lambda x: x["edge"]):
    sub = arch_res[(arch_res["vwap_hmm_state"] == r["state"]) & (arch_res["model_side"] == r["side"])]
    m, lo, hi, p = boot_p(sub["trade_edge"])
    p_report = p if r["edge"] < 0 else (1 - p)
    print(f"  State {r['state']} {r['side']}: edge_mean={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
          f"P(wrong-direction)={p_report:.4f}")

# ── week-by-week consistency ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WEEK-BY-WEEK CONSISTENCY")
print("=" * 60)
arch_res["close_ts_dt"] = parse_logged_at_mixed(arch_res["close_ts"])
arch_res["iso_week"] = (arch_res["close_ts_dt"].dt.isocalendar().year.astype(str)
                        + "-W" + arch_res["close_ts_dt"].dt.isocalendar().week.astype(str).str.zfill(2))
all_weeks = sorted(arch_res["iso_week"].dropna().unique())
for r in candidates:
    s, side = r["state"], r["side"]
    sub_all = arch_res[arch_res["model_side"] == side]
    sub_state = sub_all[sub_all["vwap_hmm_state"] == s]
    print(f"\nState {s} {side} (overall edge={r['edge']:+.3f}, n={r['n']}):")
    n_pos, n_wks = 0, 0
    for wk in all_weeks:
        wk_sub = sub_state[sub_state["iso_week"] == wk]
        if len(wk_sub) < 5:
            continue
        n_wks += 1
        wr = wk_sub["resolved_yes"].mean() if side == "YES" else 1 - wk_sub["resolved_yes"].mean()
        be = wk_sub["p_market"].mean() if side == "YES" else 1 - wk_sub["p_market"].mean()
        wk_edge = wr - be
        flag = "OK" if wk_edge * r["edge"] > 0 else "x"
        if wk_edge * r["edge"] > 0:
            n_pos += 1
        print(f"    {wk}: n={len(wk_sub)}  WR={wr:.3f}  BE={be:.3f}  edge={wk_edge:+.3f}  {flag}")
    if n_wks:
        print(f"  Directionally consistent: {n_pos}/{n_wks} weeks")

arch_res.to_csv("reform_results/vwap_hmm_sol15m_20260708/full_analysis.csv", index=False)
print("\nDone.")
