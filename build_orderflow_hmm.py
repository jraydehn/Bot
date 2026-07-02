#!/usr/bin/env python3
"""
build_orderflow_hmm.py

Builds a per-asset Gaussian HMM on order flow positioning features:
  ls_long_pct    — % of accounts long (retail crowding / positioning)
  oi_chg_pct     — open interest change % (fresh money entering/exiting)
  liq_bias       — net liquidation direction bias (-1=bearish liq, +1=bullish liq)
  vpin_score     — VPIN direction indicator (-1/0/+1)
  funding_bias   — funding rate direction (-1/0/+1)
  obi_score      — order book imbalance score (-1/0/+1)

Features read directly from scan archives — no price data needed.
State count selected via BIC over range 2..8. Saves per-asset models to:
  models/hmm_orderflow_{btc,eth,sol}.pkl

Backfills two new shadow columns into all archives + paper trades:
  hmm_of_state   int   0..N-1  (Viterbi hard state)
  hmm_of_prob    float 0..1    (posterior P(current state))

Usage:
  python3 build_orderflow_hmm.py               # train + backfill + analyse
  python3 build_orderflow_hmm.py --dry-run     # skip writes
  python3 build_orderflow_hmm.py --no-train    # skip training, load saved models
  python3 build_orderflow_hmm.py --asset BTC   # single asset
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("pip install hmmlearn")

BASE    = Path(__file__).parent
RESULTS = BASE / "results"
MODELS  = BASE / "models"
MODELS.mkdir(exist_ok=True)

ASSETS = {
    "BTC": {"archives": ["btc_scan_archive.csv", "paper_trades.csv"]},
    "ETH": {"archives": ["eth_scan_archive.csv", "paper_trades_eth.csv"]},
    "SOL": {"archives": ["sol_scan_archive.csv", "paper_trades_sol.csv"]},
}

FEAT_COLS = ["ls_long_pct", "oi_chg_pct", "liq_bias", "vpin_score", "funding_bias", "obi_score"]
NEW_CSV_COLS = ["hmm_of_state", "hmm_of_prob"]


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_matrix(asset: str) -> pd.DataFrame:
    """
    Load scan archive for this asset, clean order flow features, and aggregate
    to 1h bars. Returns DataFrame indexed by UTC bar timestamp.
    """
    fname = ASSETS[asset]["archives"][0]
    df = pd.read_csv(RESULTS / fname, low_memory=False)

    ts_col = "logged_at" if "logged_at" in df.columns else "close_ts"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    # Parse and clean each feature
    df["ls_long_pct"]  = pd.to_numeric(df["ls_long_pct"],  errors="coerce").clip(0.0,   100.0)
    df["oi_chg_pct"]   = pd.to_numeric(df["oi_chg_pct"],   errors="coerce")
    df["liq_bias"]     = pd.to_numeric(df["liq_bias"],     errors="coerce").clip(-1.0,    1.0)
    df["vpin_score"]   = pd.to_numeric(df["vpin_score"],   errors="coerce")
    df["funding_bias"] = pd.to_numeric(df["funding_bias"], errors="coerce")
    df["obi_score"]    = pd.to_numeric(df["obi_score"],    errors="coerce")

    # Winsorize oi_chg_pct at 1st / 99th percentile to suppress extreme data events
    p1, p99 = df["oi_chg_pct"].quantile([0.01, 0.99])
    df["oi_chg_pct"] = df["oi_chg_pct"].clip(p1, p99)

    # Aggregate to 1h bars (multiple contracts may be scanned in the same hour)
    df["bar_ts"] = df[ts_col].dt.floor("1h")
    agg = df.groupby("bar_ts")[FEAT_COLS].mean()
    agg.index = pd.DatetimeIndex(agg.index)
    if agg.index.tz is None:
        agg.index = agg.index.tz_localize("UTC")
    else:
        agg.index = agg.index.tz_convert("UTC")

    print(f"  Feature matrix: {len(agg):,} hourly bars  ({agg.index[0].date()} → {agg.index[-1].date()})")
    null_counts = agg.isna().sum()
    if null_counts.sum() > 0:
        print(f"  NaN counts: {null_counts[null_counts > 0].to_dict()}")

    return agg.dropna()


# ── HMM training ─────────────────────────────────────────────────────────────

def fit_hmm(feat_matrix: pd.DataFrame, max_states: int = 8) -> tuple:
    """
    Fit GaussianHMM for n_states in 2..max_states, select by BIC.
    Returns (best_model, scaler, best_n_states, state_sequence_df).
    """
    X_raw = feat_matrix[FEAT_COLS].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    lengths = [len(X)]

    best_bic   = np.inf
    best_n     = 2
    best_model = None

    print(f"  BIC selection (n=2..{max_states}):")
    for n in range(2, max_states + 1):
        results = []
        for seed in range(5):
            try:
                m = GaussianHMM(
                    n_components=n,
                    covariance_type="diag",
                    n_iter=200,
                    tol=1e-4,
                    random_state=seed,
                    verbose=False,
                )
                m.fit(X, lengths)
                ll = m.score(X, lengths)
                d  = X.shape[1]
                T  = len(X)
                k  = n * (n - 1) + 2 * n * d
                bic = -2 * ll + k * np.log(T)
                results.append((bic, m, ll))
            except Exception:
                pass
        if not results:
            continue
        bic, m, ll = min(results, key=lambda x: x[0])
        marker = " ← best" if bic < best_bic else ""
        print(f"    n={n}  BIC={bic:.1f}  logL={ll:.1f}{marker}")
        if bic < best_bic:
            best_bic  = bic
            best_n    = n
            best_model = m

    states     = best_model.predict(X, lengths)
    log_post   = best_model.predict_proba(X, lengths)
    probs      = np.exp(log_post)
    asgn_probs = probs[np.arange(len(states)), states]

    state_df = pd.DataFrame({
        "hmm_of_state": states.astype(int),
        "hmm_of_prob":  np.round(asgn_probs, 4),
    }, index=feat_matrix.index)

    return best_model, scaler, best_n, state_df


# ── State characterization ────────────────────────────────────────────────────

def characterize_states(feat_matrix: pd.DataFrame, state_df: pd.DataFrame,
                         asset: str, n_states: int) -> None:
    joined = feat_matrix.join(state_df, how="inner")
    print(f"\n  State characterization ({asset}, {n_states} states):")
    header = (f"  {'st':<4}  {'ls_long':>7}  {'oi_chg':>7}  "
              f"{'liq_bias':>9}  {'vpin':>5}  {'fund':>5}  {'obi':>5}  {'n':>6}")
    print(header)
    for s in range(n_states):
        sub = joined[joined["hmm_of_state"] == s]
        if len(sub) == 0:
            continue
        print(f"  {s:<4}  "
              f"{sub['ls_long_pct'].mean():>7.2f}  "
              f"{sub['oi_chg_pct'].mean():>+7.4f}  "
              f"{sub['liq_bias'].mean():>+9.4f}  "
              f"{sub['vpin_score'].mean():>+5.2f}  "
              f"{sub['funding_bias'].mean():>+5.2f}  "
              f"{sub['obi_score'].mean():>+5.2f}  "
              f"{len(sub):>6}")


# ── Backfill ──────────────────────────────────────────────────────────────────

def backfill_csv(csv_path: Path, state_df: pd.DataFrame, dry_run: bool) -> None:
    if not csv_path.exists():
        print(f"    SKIP (not found): {csv_path.name}")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    ts_col = "logged_at" if "logged_at" in df.columns else "close_ts"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    for c in NEW_CSV_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    state_ts  = state_df.index.sort_values()
    state_arr = state_df.loc[state_ts]

    already = df["hmm_of_state"].notna().sum()
    filled  = 0
    for idx in df.index:
        row_ts = df.at[idx, ts_col]
        if pd.isna(row_ts):
            continue
        if pd.notna(df.at[idx, "hmm_of_state"]):
            continue
        prior = state_ts[state_ts < row_ts]
        if len(prior) == 0:
            continue
        bar = prior[-1]
        df.at[idx, "hmm_of_state"] = int(state_arr.at[bar, "hmm_of_state"])
        df.at[idx, "hmm_of_prob"]  = float(state_arr.at[bar, "hmm_of_prob"])
        filled += 1

    print(f"    {csv_path.name}: already={already:,}  filled={filled:,}")
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"    saved → {csv_path.name}")


# ── MCPT ─────────────────────────────────────────────────────────────────────

def mcpt(full_edges: np.ndarray, n_in: int, obs_mean: float,
         n_perm: int = 5000, rng: np.random.Generator = None) -> tuple:
    """
    Proper MCPT: sample n_in indices from full edge array, compute mean,
    repeat n_perm times.
    p_block = fraction of perm means <= obs_mean  (low p = state is bad = block candidate)
    p_rescue = fraction of perm means >= obs_mean (high p = state is good = rescue candidate)
    Returns (z_score, p_block, p_rescue)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    perm_means = np.array([rng.choice(full_edges, size=n_in, replace=False).mean()
                           for _ in range(n_perm)])
    pop_mean = full_edges.mean()
    pop_std  = perm_means.std()
    z = (obs_mean - pop_mean) / (pop_std + 1e-12)
    p_block  = float((perm_means <= obs_mean).mean())
    p_rescue = float((perm_means >= obs_mean).mean())
    return round(z, 3), round(p_block, 4), round(p_rescue, 4)


# ── Trade analysis ────────────────────────────────────────────────────────────

def analyse_trades(asset: str, state_df: pd.DataFrame, n_states: int) -> None:
    """Print WR/edge/PnL by order-flow HMM state + MCPT on each state."""
    paper_fname = [f for f in ASSETS[asset]["archives"] if "paper_trade" in f]
    if not paper_fname:
        return

    df = pd.read_csv(RESULTS / paper_fname[0], low_memory=False)
    for c in df.columns:
        if c not in ("side", "asset", "contract_ticker", "logged_at", "decision",
                     "p_market_source", "close_ts", "decision_time", "loss_category"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[(df.get("kelly_fraction", pd.Series(dtype=float)) > 0) &
            df["resolved_yes"].notna()].copy()

    if df.empty:
        print(f"  {asset}: no resolved trades with kelly_fraction>0")
        return

    if "hmm_of_state" not in df.columns or df["hmm_of_state"].isna().all():
        print(f"  {asset}: hmm_of_state not yet backfilled into paper trades")
        return

    rng = np.random.default_rng(42)

    print(f"\n  {'═'*68}")
    print(f"  {asset} — trade outcomes by order-flow HMM state")
    print(f"  {'═'*68}")

    for side in ["yes", "no"]:
        sub = df[df["side"] == side].dropna(subset=["hmm_of_state"]).copy()
        if len(sub) < 5:
            continue

        if side == "yes":
            sub["edge"] = sub["resolved_yes"] - sub["p_market"]
            sub["won"]  = sub["resolved_yes"].astype(float)
        else:
            sub["edge"] = (1 - sub["resolved_yes"]) - (1 - sub["p_market"])
            sub["won"]  = (1 - sub["resolved_yes"])

        base_wr   = sub["won"].mean()
        base_edge = sub["edge"].mean()
        base_pnl  = sub["would_pnl"].sum()
        ic, ic_p  = spearmanr(sub["hmm_of_state"], sub["edge"])
        full_edges = sub["edge"].values

        print(f"\n  {side.upper()} baseline: n={len(sub)}  WR={base_wr:.1%}  "
              f"edge={base_edge:+.1%}  PnL=${base_pnl:+.0f}")
        print(f"  IC(state,edge)={ic:+.4f}  p={ic_p:.4f}{'  ***' if ic_p < 0.05 else ''}")
        print(f"  {'st':<4}  {'n':>5}  {'WR':>7}  {'edge':>8}  {'PnL':>9}  "
              f"{'z':>7}  {'p_blk':>7}  {'p_rsc':>7}  note")

        for s in range(n_states):
            sg = sub[sub["hmm_of_state"] == s]
            if len(sg) < 5:
                continue
            wr_s  = sg["won"].mean()
            eg_s  = sg["edge"].mean()
            pnl_s = sg["would_pnl"].sum()
            z, p_blk, p_rsc = mcpt(full_edges, len(sg), eg_s, rng=rng)

            note = ""
            if p_blk <= 0.05:
                note = "  *** BLOCK candidate"
            elif p_rsc >= 0.95:
                note = "  *** RESCUE/CONVICTION"

            print(f"  {s:<4}  {len(sg):>5}  {wr_s:>7.1%}  {eg_s:>+8.1%}  "
                  f"{pnl_s:>+9.0f}  {z:>+7.2f}  {p_blk:>7.4f}  {p_rsc:>7.4f}{note}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--no-train",   action="store_true",
                    help="Skip training, load saved models")
    ap.add_argument("--asset",      default="ALL",
                    choices=["BTC", "ETH", "SOL", "ALL"])
    ap.add_argument("--max-states", type=int, default=8)
    args = ap.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.asset == "ALL" else [args.asset]

    for asset in assets:
        pkl_path = MODELS / f"hmm_orderflow_{asset.lower()}.pkl"

        print(f"\n{'='*60}")
        print(f"  {asset} — Order Flow HMM")
        print(f"{'='*60}")

        if args.no_train and pkl_path.exists():
            pkg = pickle.load(open(pkl_path, "rb"))
            model       = pkg["model"]
            scaler      = pkg["scaler"]
            n_states    = pkg["n_states"]
            state_df    = pkg["state_df"]
            feat_matrix = pkg["feat_matrix"]
            print(f"  Loaded saved model ({n_states} states) from {pkl_path.name}")
            characterize_states(feat_matrix, state_df, asset, n_states)
        else:
            print(f"  Building feature matrix from scan archive ...")
            feat_matrix = build_feature_matrix(asset)

            print(f"\n  Fitting HMM ...")
            model, scaler, n_states, state_df = fit_hmm(feat_matrix, max_states=args.max_states)
            print(f"  Best: {n_states} states")

            characterize_states(feat_matrix, state_df, asset, n_states)

            if not args.dry_run:
                pkg = {
                    "model":       model,
                    "scaler":      scaler,
                    "n_states":    n_states,
                    "feat_cols":   FEAT_COLS,
                    "state_df":    state_df,
                    "feat_matrix": feat_matrix,
                    "asset":       asset,
                }
                pickle.dump(pkg, open(pkl_path, "wb"))
                print(f"  Saved → {pkl_path}")

        # Backfill archives
        print(f"\n  Backfilling archives ...")
        for fname in ASSETS[asset]["archives"]:
            backfill_csv(RESULTS / fname, state_df, args.dry_run)

        # Trade analysis
        analyse_trades(asset, state_df, n_states)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
