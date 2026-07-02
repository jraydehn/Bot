#!/usr/bin/env python3
"""
build_voldirection_hmm.py

Builds a per-asset Gaussian HMM on vol + direction features — filling the gap
in the existing R0/R1 vol HMM which separates volatility *level* but conflates
direction. The four natural latent states:
  low_vol_trending   — quiet grind in one direction (YES/NO holds)
  low_vol_flat       — quiet consolidation, no direction (bets expire near price)
  high_vol_directed  — volatile but trending (edge exists if model is aligned)
  high_vol_chaotic   — violent whipsaw (all bets risky)

Features (all computed from 1h price bars):
  chg_1h        — 1h log return (signed direction)
  chg_3h        — 3h cumulative log return (momentum confirmation)
  rvol_1h       — std of last 24 1h log-returns (short-term realized vol)
  vol_ratio     — rvol_1h / rvol_168h  (vol relative to 7-day baseline; >1 = elevated)
  ema_trend     — (EMA8 - EMA21) / close (continuous trend direction, normalized)

State count selected via BIC over range 2..8. Saves per-asset models to:
  models/hmm_voldirection_{btc,eth,sol}.pkl

Backfills two new shadow columns into all archives + paper trades:
  hmm_vd_state   int   0..N-1  (Viterbi hard state)
  hmm_vd_prob    float 0..1    (posterior P(current state))

Usage:
  python3 build_voldirection_hmm.py               # train + backfill + analyse
  python3 build_voldirection_hmm.py --dry-run     # skip writes
  python3 build_voldirection_hmm.py --no-train    # skip training, load saved models
  python3 build_voldirection_hmm.py --asset BTC   # single asset
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

FEAT_COLS    = ["chg_1h", "chg_3h", "rvol_1h", "vol_ratio", "ema_trend"]
NEW_CSV_COLS = ["hmm_vd_state", "hmm_vd_prob"]


# ── Price loading (shared with build_microstructure_hmm.py) ───────────────────

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


# ── Feature construction ──────────────────────────────────────────────────────

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (standard EW, adjust=False)."""
    alpha = 2.0 / (span + 1)
    out = np.empty(len(series))
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def build_feature_matrix(price: pd.Series, min_bars: int = 175) -> pd.DataFrame:
    """
    Compute per-bar vol + direction features from 1h close price series.
    min_bars = 175 to ensure 168-bar rvol baseline is stable.
    All features use only data available at bar t (strictly causal).
    """
    c    = price.values
    idx  = price.index
    lr   = np.log(c[1:] / c[:-1])    # 1h log returns, length = len(c)-1
    ts   = idx[1:]                     # timestamps aligned to lr

    ema8  = _ema(c, 8)
    ema21 = _ema(c, 21)

    rows  = []
    ts_out = []

    for i in range(min_bars, len(lr)):
        # chg_1h: most recent 1h return
        chg_1h = float(lr[i])

        # chg_3h: 3-bar cumulative return
        chg_3h = float(lr[i - 2] + lr[i - 1] + lr[i]) if i >= 2 else chg_1h

        # rvol_1h: std of last 24 1h log-returns
        window_short = lr[max(0, i - 23): i + 1]
        rvol_1h = float(np.std(window_short, ddof=1)) if len(window_short) >= 4 else float("nan")

        # vol_ratio: rvol_1h / rvol_168h (7-day baseline)
        window_long = lr[max(0, i - 167): i + 1]
        rvol_long = float(np.std(window_long, ddof=1)) if len(window_long) >= 24 else float("nan")
        vol_ratio = float(rvol_1h / rvol_long) if (rvol_long and rvol_long > 0) else float("nan")

        # ema_trend: (EMA8 - EMA21) / close, normalized continuous direction
        ema_trend = float((ema8[i + 1] - ema21[i + 1]) / c[i + 1]) if c[i + 1] > 0 else float("nan")

        rows.append([chg_1h, chg_3h, rvol_1h, vol_ratio, ema_trend])
        ts_out.append(ts[i])

    df = pd.DataFrame(rows, columns=FEAT_COLS,
                      index=pd.DatetimeIndex(ts_out, tz="UTC"))
    return df.dropna()


# ── HMM training ─────────────────────────────────────────────────────────────

def fit_hmm(feat_matrix: pd.DataFrame, max_states: int = 8) -> tuple:
    """Fit GaussianHMM (diag) for n=2..max_states, select by BIC."""
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
                d   = X.shape[1]
                T   = len(X)
                k   = n * (n - 1) + 2 * n * d
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
            best_bic   = bic
            best_n     = n
            best_model = m

    states     = best_model.predict(X, lengths)
    log_post   = best_model.predict_proba(X, lengths)
    probs      = np.exp(log_post)
    asgn_probs = probs[np.arange(len(states)), states]

    state_df = pd.DataFrame({
        "hmm_vd_state": states.astype(int),
        "hmm_vd_prob":  np.round(asgn_probs, 4),
    }, index=feat_matrix.index)

    return best_model, scaler, best_n, state_df


# ── State characterization ────────────────────────────────────────────────────

def characterize_states(feat_matrix: pd.DataFrame, state_df: pd.DataFrame,
                         asset: str, n_states: int) -> None:
    joined = feat_matrix.join(state_df, how="inner")
    print(f"\n  State characterization ({asset}, {n_states} states):")
    header = (f"  {'st':<4}  {'chg_1h':>8}  {'chg_3h':>8}  "
              f"{'rvol_1h':>8}  {'vol_ratio':>9}  {'ema_trend':>10}  {'n':>6}  label")
    print(header)
    for s in range(n_states):
        sub = joined[joined["hmm_vd_state"] == s]
        if len(sub) == 0:
            continue
        # Derive human-readable label
        v_ratio = sub["vol_ratio"].mean()
        chg     = sub["chg_1h"].mean()
        ema     = sub["ema_trend"].mean()
        if v_ratio < 1.0:
            vol_label = "low_vol"
        else:
            vol_label = "high_vol"
        if abs(ema) > 0.0003:
            dir_label = "bull" if ema > 0 else "bear"
        elif abs(chg) > 0.001:
            dir_label = "up" if chg > 0 else "down"
        else:
            dir_label = "flat"
        label = f"{vol_label}_{dir_label}"
        print(f"  {s:<4}  {sub['chg_1h'].mean():>+8.5f}  {sub['chg_3h'].mean():>+8.5f}  "
              f"{sub['rvol_1h'].mean():>8.5f}  {sub['vol_ratio'].mean():>9.4f}  "
              f"{sub['ema_trend'].mean():>+10.6f}  {len(sub):>6}  {label}")


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

    already = df["hmm_vd_state"].notna().sum()
    filled  = 0
    for idx in df.index:
        row_ts = df.at[idx, ts_col]
        if pd.isna(row_ts):
            continue
        if pd.notna(df.at[idx, "hmm_vd_state"]):
            continue
        prior = state_ts[state_ts < row_ts]
        if len(prior) == 0:
            continue
        bar = prior[-1]
        df.at[idx, "hmm_vd_state"] = int(state_arr.at[bar, "hmm_vd_state"])
        df.at[idx, "hmm_vd_prob"]  = float(state_arr.at[bar, "hmm_vd_prob"])
        filled += 1

    print(f"    {csv_path.name}: already={already:,}  filled={filled:,}")
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"    saved → {csv_path.name}")


# ── MCPT ─────────────────────────────────────────────────────────────────────

def mcpt(full_edges: np.ndarray, n_in: int, obs_mean: float,
         n_perm: int = 5000, rng=None) -> tuple:
    if rng is None:
        rng = np.random.default_rng(42)
    perm_means = np.array([rng.choice(full_edges, size=n_in, replace=False).mean()
                           for _ in range(n_perm)])
    z      = (obs_mean - full_edges.mean()) / (perm_means.std() + 1e-12)
    p_blk  = float((perm_means <= obs_mean).mean())   # high = unusually GOOD (rescue)
    p_bad  = float((perm_means >= obs_mean).mean())   # high = unusually BAD  (block)
    return round(z, 3), round(p_blk, 4), round(p_bad, 4)


# ── Trade analysis ────────────────────────────────────────────────────────────

def analyse_trades(asset: str, state_df: pd.DataFrame, n_states: int) -> None:
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
        print(f"  {asset}: no resolved trades")
        return

    if "hmm_vd_state" not in df.columns or df["hmm_vd_state"].isna().all():
        print(f"  {asset}: hmm_vd_state not yet backfilled")
        return

    rng = np.random.default_rng(42)

    print(f"\n  {'═'*72}")
    print(f"  {asset} — trade outcomes by vol+direction HMM state")
    print(f"  {'═'*72}")

    for side in ["yes", "no"]:
        sub = df[df["side"] == side].dropna(subset=["hmm_vd_state"]).copy()
        if len(sub) < 5:
            continue

        sub["edge"] = ((sub["resolved_yes"] - sub["p_market"])
                       if side == "yes"
                       else (1 - sub["resolved_yes"]) - (1 - sub["p_market"]))
        sub["won"]  = sub["resolved_yes"] if side == "yes" else (1 - sub["resolved_yes"])
        full_edges  = sub["edge"].values.astype(float)

        base_wr   = sub["won"].mean()
        base_edge = sub["edge"].mean()
        base_pnl  = sub["would_pnl"].sum()
        ic, ic_p  = spearmanr(sub["hmm_vd_state"], sub["edge"])

        print(f"\n  {side.upper()} baseline: n={len(sub)}  WR={base_wr:.1%}  "
              f"edge={base_edge:+.1%}  PnL=${base_pnl:+.0f}")
        print(f"  IC(state,edge)={ic:+.4f}  p={ic_p:.4f}{'  ***' if ic_p < 0.05 else ''}")
        print(f"  {'st':<4}  {'n':>5}  {'WR':>7}  {'edge':>8}  {'PnL':>9}  "
              f"{'z':>7}  {'p_good':>7}  {'p_bad':>7}  note")

        for s in range(n_states):
            sg = sub[sub["hmm_vd_state"] == s]
            if len(sg) < 5:
                continue
            wr_s  = sg["won"].mean()
            eg_s  = sg["edge"].mean()
            pnl_s = sg["would_pnl"].sum()
            z, p_good, p_bad = mcpt(full_edges, len(sg), eg_s, rng=rng)

            note = ""
            if p_bad >= 0.95:
                note = "  *** BLOCK candidate"
            elif p_good >= 0.95:
                note = "  *** CONVICTION"

            print(f"  {s:<4}  {len(sg):>5}  {wr_s:>7.1%}  {eg_s:>+8.1%}  "
                  f"{pnl_s:>+9.0f}  {z:>+7.2f}  {p_good:>7.4f}  {p_bad:>7.4f}{note}")


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
        pkl_path = MODELS / f"hmm_voldirection_{asset.lower()}.pkl"

        print(f"\n{'='*60}")
        print(f"  {asset} — Vol + Direction HMM")
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
            ticker = ASSETS[asset]["ticker"]
            print(f"  Loading price data ...")
            price = load_1h_series(ticker)

            print(f"  Building feature matrix ...")
            feat_matrix = build_feature_matrix(price)
            print(f"  Feature matrix: {len(feat_matrix):,} rows × {len(FEAT_COLS)} features")
            print(f"  Date range: {feat_matrix.index[0].date()} → {feat_matrix.index[-1].date()}")
            null_counts = feat_matrix.isna().sum()
            if null_counts.sum() > 0:
                print(f"  NaN counts: {null_counts[null_counts > 0].to_dict()}")

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
                    "ticker":      ticker,
                }
                pickle.dump(pkg, open(pkl_path, "wb"))
                print(f"  Saved → {pkl_path}")

        print(f"\n  Backfilling archives ...")
        for fname in ASSETS[asset]["archives"]:
            backfill_csv(RESULTS / fname, state_df, args.dry_run)

        analyse_trades(asset, state_df, n_states)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
