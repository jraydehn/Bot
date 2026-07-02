#!/usr/bin/env python3
"""
build_microstructure_hmm.py

Builds a per-asset Gaussian HMM on microstructure features:
  ou_theta       — log-return mean-reversion speed (how fast returns revert to mean)
  hurst_exponent — R/S persistence; H>0.5=trending, H<0.5=mean-reverting
  autocorr1_30   — lag-1 autocorrelation of 1h returns (60-bar window)
  kalman_velocity — Kalman-filtered 1h return trend
  rvol_24h       — realized vol (std of last 24 1h log-returns)

State count selected via BIC over range 2..8. Saves per-asset models to:
  models/hmm_microstructure_{btc,eth,sol}.pkl

Backfills two new shadow columns into all hourly archives + paper trades:
  hmm_ms_state   int   0..N-1  (Viterbi hard state)
  hmm_ms_prob    float 0..1    (posterior P(current state))

Finally prints IC analysis and WR/PnL breakdown by state vs trade outcomes.

Usage:
  python3 build_microstructure_hmm.py             # train + backfill + analyse
  python3 build_microstructure_hmm.py --dry-run   # skip writes
  python3 build_microstructure_hmm.py --no-train  # skip training, use saved models
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
DATA    = BASE / "data"
RESULTS = BASE / "results"
MODELS  = BASE / "models"
MODELS.mkdir(exist_ok=True)

ASSETS = {
    "BTC": {"ticker": "BTCUSDT",
            "archives": ["btc_scan_archive.csv", "paper_trades.csv"]},
    "ETH": {"ticker": "ETHUSDT",
            "archives": ["eth_scan_archive.csv", "paper_trades_eth.csv"]},
    "SOL": {"ticker": "SOLUSDT",
            "archives": ["sol_scan_archive.csv", "paper_trades_sol.csv"]},
}

FEAT_COLS = ["ou_theta", "hurst_exponent", "autocorr1_30", "kalman_velocity", "rvol_24h"]

NEW_CSV_COLS = ["hmm_ms_state", "hmm_ms_prob"]


# ── Signal computation ────────────────────────────────────────────────────────

def _lag1_ac(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    x = arr[:-1] - arr[:-1].mean()
    y = arr[1:]  - arr[1:].mean()
    denom = float(np.sqrt((x**2).sum() * (y**2).sum()))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def _hurst(lr: np.ndarray) -> float:
    wins = [8, 16, 32, 64]
    pts = []
    for w in wins:
        if len(lr) < w:
            continue
        seg  = lr[-w:]
        mean = seg.mean()
        dev  = np.cumsum(seg - mean)
        r    = dev.max() - dev.min()
        s    = seg.std(ddof=1)
        if s > 0 and r > 0:
            pts.append((np.log(w), np.log(r / s)))
    if len(pts) < 2:
        return float("nan")
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    h  = float(np.polyfit(xs, ys, 1)[0])
    return float(np.clip(h, 0.0, 1.0))


def _ou_theta(lr: np.ndarray) -> float:
    buf = lr[-48:] if len(lr) >= 48 else lr
    if len(buf) < 10:
        return float("nan")
    y = buf - buf.mean()
    phi = float(np.dot(y[:-1], y[1:]) / (np.dot(y[:-1], y[:-1]) + 1e-12))
    phi = float(np.clip(phi, -0.9999, 0.9999))
    return float(np.clip(-np.log(abs(phi)), 0.0, 10.0))


def _kalman_vel(lr: np.ndarray) -> float:
    buf = lr[-48:] if len(lr) >= 48 else lr
    if len(buf) < 5:
        return float("nan")
    Q = np.array([[1e-5, 0.0], [0.0, 1e-5]])
    R = float(np.var(buf)) + 1e-10
    x = np.array([buf[0], 0.0])
    P = np.eye(2) * 0.1
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    for obs in buf:
        x = F @ x
        P = F @ P @ F.T + Q
        K = P @ H.T / (float(H @ P @ H.T) + R)
        x = x + K.flatten() * (obs - float(H @ x))
        P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
    return round(float(x[1]), 6)


def compute_feature_row(lr: np.ndarray) -> dict:
    """Compute all 5 microstructure features from a 1h log-return array."""
    if len(lr) < 10:
        return {c: float("nan") for c in FEAT_COLS}
    return {
        "ou_theta":       _ou_theta(lr),
        "hurst_exponent": _hurst(lr),
        "autocorr1_30":   _lag1_ac(lr[-60:] if len(lr) >= 60 else lr),
        "kalman_velocity": _kalman_vel(lr),
        "rvol_24h":        float(np.std(lr[-24:] if len(lr) >= 24 else lr, ddof=1)),
    }


# ── Price data ────────────────────────────────────────────────────────────────

def load_1h_series(ticker: str) -> pd.Series:
    """Return 1h close price series (UTC-indexed, all available data)."""
    files_1m = sorted(DATA.glob(f"binanceus_{ticker}_1m_2024-01-01_*.parquet"))
    files_1h = sorted(DATA.glob(f"binanceus_{ticker}_1h_1970-01-01_*.parquet"))

    if files_1m:
        df = pd.read_parquet(files_1m[-1], columns=["close"])
        df.index = pd.to_datetime(df.index, utc=True)
        s_1m = df["close"].resample("1h").last().dropna()
        latest_1m = s_1m.index.max()

        if files_1h:
            df_1h = pd.read_parquet(files_1h[-1], columns=["close"])
            df_1h.index = pd.to_datetime(df_1h.index, utc=True)
            latest_1h = df_1h.index.max()
            if (latest_1h - latest_1m).days > 7:
                ext = df_1h.loc[df_1h.index > latest_1m, "close"]
                combined = pd.concat([s_1m, ext]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                print(f"  {ticker}: {len(combined):,} 1h bars (1m→{latest_1m.date()} + 1h ext to {latest_1h.date()})")
                return combined
        print(f"  {ticker}: {len(s_1m):,} 1h bars (from 1m, up to {latest_1m.date()})")
        return s_1m

    if files_1h:
        df = pd.read_parquet(files_1h[-1], columns=["close"])
        df.index = pd.to_datetime(df.index, utc=True)
        s = df["close"].dropna()
        print(f"  {ticker}: {len(s):,} 1h bars (1h parquet)")
        return s

    raise FileNotFoundError(f"No price data for {ticker}")


# ── Feature matrix construction ───────────────────────────────────────────────

def build_feature_matrix(price_series: pd.Series, min_bars: int = 70) -> pd.DataFrame:
    """
    For every 1h bar (starting from bar min_bars), compute all 5 features
    using only data available UP TO that bar. Returns a DataFrame indexed
    by UTC timestamp with columns = FEAT_COLS.
    """
    lr = np.log(price_series / price_series.shift(1)).dropna().values
    timestamps = price_series.index[1:]  # lr[i] corresponds to timestamps[i]

    rows = []
    ts_out = []
    for i in range(min_bars, len(lr)):
        avail = lr[:i]  # strictly prior bars
        feat  = compute_feature_row(avail)
        rows.append(feat)
        ts_out.append(timestamps[i])

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_out, tz="UTC"))
    return df.dropna()


# ── HMM training ─────────────────────────────────────────────────────────────

def fit_hmm(feat_matrix: pd.DataFrame, max_states: int = 8) -> tuple:
    """
    Fit GaussianHMM for n_states in 2..max_states, select by BIC.
    Returns (best_model, scaler, best_n_states, state_sequence_df).
    state_sequence_df has columns [hmm_ms_state, hmm_ms_prob] indexed by timestamp.
    """
    X_raw = feat_matrix[FEAT_COLS].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    lengths = [len(X)]

    best_bic  = np.inf
    best_n    = 2
    best_model = None

    print(f"  BIC selection (n=2..{max_states}):")
    for n in range(2, max_states + 1):
        results = []
        for seed in range(5):  # multiple restarts for stability
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
                # BIC = -2*ll + k*log(T)
                # k = n*(n-1) + 2*n*d  (transition + emission params)
                d = X.shape[1]
                T = len(X)
                k = n * (n - 1) + 2 * n * d
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

    # Viterbi path on full series
    states = best_model.predict(X, lengths)
    # Posterior state probabilities
    log_post = best_model.predict_proba(X, lengths)  # (T, n_states)
    probs    = np.exp(log_post)
    assigned_probs = probs[np.arange(len(states)), states]

    state_df = pd.DataFrame({
        "hmm_ms_state": states.astype(int),
        "hmm_ms_prob":  np.round(assigned_probs, 4),
    }, index=feat_matrix.index)

    return best_model, scaler, best_n, state_df


# ── State characterization ────────────────────────────────────────────────────

def characterize_states(feat_matrix: pd.DataFrame, state_df: pd.DataFrame,
                         asset: str, n_states: int) -> None:
    """Print mean feature values per state to interpret what each state means."""
    joined = feat_matrix.join(state_df, how="inner")
    print(f"\n  State characterization ({asset}, {n_states} states):")
    cols_show = FEAT_COLS
    header = f"  {'state':<6}" + "".join(f"  {c[:12]:>12}" for c in cols_show) + "   n_bars"
    print(header)
    for s in range(n_states):
        sub = joined[joined["hmm_ms_state"] == s]
        vals = "".join(f"  {sub[c].mean():>+12.4f}" for c in cols_show)
        print(f"  {s:<6}{vals}  {len(sub):>6}")


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

    # Match each archive row to the most recent prior HMM timestamp
    # state_df is indexed by UTC timestamp of bar completion
    state_ts  = state_df.index.sort_values()
    state_arr = state_df.loc[state_ts]

    already = df["hmm_ms_state"].notna().sum()
    filled  = 0
    for idx in df.index:
        row_ts = df.at[idx, ts_col]
        if pd.isna(row_ts):
            continue
        if pd.notna(df.at[idx, "hmm_ms_state"]):
            continue
        # Find the most recent HMM bar completed before this row's timestamp
        prior = state_ts[state_ts < row_ts]
        if len(prior) == 0:
            continue
        bar = prior[-1]
        df.at[idx, "hmm_ms_state"] = int(state_arr.at[bar, "hmm_ms_state"])
        df.at[idx, "hmm_ms_prob"]  = float(state_arr.at[bar, "hmm_ms_prob"])
        filled += 1

    print(f"    {csv_path.name}: already={already:,} filled={filled:,}")
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"    saved → {csv_path.name}")


# ── Correlation analysis ──────────────────────────────────────────────────────

def analyse_trades(asset: str, state_df: pd.DataFrame, n_states: int) -> None:
    """Load paper trades for this asset and show WR/edge/PnL broken down by HMM state."""
    fnames = ASSETS[asset]["archives"]
    paper  = [f for f in fnames if "paper_trades" in f or "paper_trade" in f]
    if not paper:
        return

    df = pd.read_csv(RESULTS / paper[0], low_memory=False)
    for c in df.columns:
        if c not in ("side", "asset", "contract_ticker", "logged_at", "decision",
                     "p_market_source", "close_ts", "decision_time", "loss_category"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["kelly_fraction"] > 0) & df["resolved_yes"].notna()].copy()

    if "hmm_ms_state" not in df.columns or df["hmm_ms_state"].isna().all():
        print(f"  {asset}: hmm_ms_state not yet backfilled into paper trades")
        return

    print(f"\n  {'═'*60}")
    print(f"  {asset} — trade outcomes by microstructure HMM state")
    print(f"  {'═'*60}")

    for side in ["yes", "no"]:
        sub = df[df["side"] == side].dropna(subset=["hmm_ms_state"])
        if len(sub) < 5:
            continue
        if side == "yes":
            edge_col = sub["resolved_yes"] - sub["p_market"]
        else:
            edge_col = (1 - sub["resolved_yes"]) - (1 - sub["p_market"])
        base_wr   = sub["resolved_yes"].mean() if side == "yes" else (1 - sub["resolved_yes"]).mean()
        base_edge = edge_col.mean()
        base_pnl  = sub["would_pnl"].sum()

        # Spearman IC between state label and edge — test if ordering matters
        ic, ic_p = spearmanr(sub["hmm_ms_state"], edge_col)

        print(f"\n  {side.upper()} baseline: n={len(sub)}  WR={base_wr:.1%}  "
              f"edge={base_edge:+.1%}  PnL=${base_pnl:+.0f}")
        print(f"  IC(state,edge)={ic:+.4f}  p={ic_p:.4f}{'  ***' if ic_p<0.05 else '  (ns)'}")
        print(f"  {'state':<6}  {'n':>5}  {'WR':>7}  {'edge':>8}  {'PnL':>9}  {'freq':>6}")

        for s in range(n_states):
            sg = sub[sub["hmm_ms_state"] == s]
            if len(sg) < 3:
                continue
            wr_s  = sg["resolved_yes"].mean() if side=="yes" else (1-sg["resolved_yes"]).mean()
            eg_s  = (sg["resolved_yes"]-sg["p_market"]).mean() if side=="yes" \
                    else ((1-sg["resolved_yes"])-(1-sg["p_market"])).mean()
            pnl_s = sg["would_pnl"].sum()
            freq  = len(sg) / len(sub)
            flag  = "  *** LOSS" if eg_s < -0.05 else ("  *** GOOD" if eg_s > 0.10 else "")
            print(f"  {s:<6}  {len(sg):>5}  {wr_s:>7.1%}  {eg_s:>+8.1%}  {pnl_s:>+9.0f}  "
                  f"{freq:>5.1%}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--no-train", action="store_true",
                    help="Skip training, load saved models")
    ap.add_argument("--asset",    default="ALL",
                    choices=["BTC", "ETH", "SOL", "ALL"])
    ap.add_argument("--max-states", type=int, default=8)
    args = ap.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.asset == "ALL" else [args.asset]

    for asset in assets:
        ticker  = ASSETS[asset]["ticker"]
        pkl_path = MODELS / f"hmm_microstructure_{asset.lower()}.pkl"

        print(f"\n{'='*60}")
        print(f"  {asset}")
        print(f"{'='*60}")

        if args.no_train and pkl_path.exists():
            pkg = pickle.load(open(pkl_path, "rb"))
            model    = pkg["model"]
            scaler   = pkg["scaler"]
            n_states = pkg["n_states"]
            state_df = pkg["state_df"]
            feat_matrix = pkg["feat_matrix"]
            print(f"  Loaded saved model ({n_states} states) from {pkl_path.name}")
        else:
            print(f"  Loading price data ...")
            price = load_1h_series(ticker)

            print(f"  Building feature matrix ...")
            feat_matrix = build_feature_matrix(price)
            print(f"  Feature matrix: {len(feat_matrix):,} rows × {len(FEAT_COLS)} features")
            print(f"  Date range: {feat_matrix.index[0].date()} → {feat_matrix.index[-1].date()}")
            print(f"  NaN counts: {feat_matrix.isna().sum().to_dict()}")

            print(f"\n  Fitting HMM ...")
            model, scaler, n_states, state_df = fit_hmm(
                feat_matrix, max_states=args.max_states
            )
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
                    "ticker":      ticker,
                }
                pickle.dump(pkg, open(pkl_path, "wb"))
                print(f"  Saved → {pkl_path}")

        # Backfill into archives
        print(f"\n  Backfilling archives ...")
        for fname in ASSETS[asset]["archives"]:
            backfill_csv(RESULTS / fname, state_df, args.dry_run)

        # Correlation analysis
        analyse_trades(asset, state_df, n_states)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
