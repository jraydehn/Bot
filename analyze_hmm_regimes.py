"""
analyze_hmm_regimes.py — Compare HMM regime labeling vs current ±2% threshold.

Trains GaussianHMM (2, 3, 4 states) on BTC-USD daily data (2022-present)
using log return + 20-day realized vol as emission features.

Outputs:
  - State transition matrices and emission means per model
  - Regime label comparison (HMM vs current threshold) for May 18-27
  - YES win rate + P&L per HMM state vs current regime on scan archive
  - results/hmm_regime_comparison_YYYYMMDD.csv
"""
import math, warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

warnings.filterwarnings("ignore")

RES_DIR = Path(__file__).parent / "results"
N_STATES_LIST = [2, 3, 4]
TRAIN_START   = "2021-01-01"
FEE_RATE      = 0.07
MIN_EDGE      = 0.005
BANKROLL      = 1_000.0
KELLY_MULT    = 0.30
KELLY_CAP     = 0.06
PM_MIN, PM_MAX   = 0.10, 0.90
TAU_MIN, TAU_MAX = 5.0, 150.0


# ── data ─────────────────────────────────────────────────────────────────────

def fetch_btc_daily():
    print("Fetching BTC-USD daily data from yfinance...")
    df = yf.download("BTC-USD", start=TRAIN_START, auto_adjust=True, progress=False)
    df = df["Close"].squeeze().dropna()
    df.name = "close"
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    print(f"  {len(df)} daily bars  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def build_features(close_s):
    """Return DataFrame with log_ret, realized_vol_20d, and ret_5d momentum."""
    lr   = np.log(close_s / close_s.shift(1))
    rv   = lr.rolling(20, min_periods=10).std()
    r5   = np.log(close_s / close_s.shift(5))   # 5-day momentum
    df   = pd.DataFrame({"log_ret": lr, "realized_vol": rv, "ret_5d": r5}).dropna()
    return df


def current_threshold_label(ret_20d):
    if ret_20d > 0.02:
        return "Bull"
    elif ret_20d < -0.02:
        return "Bear"
    return "Sideways"


# ── HMM ──────────────────────────────────────────────────────────────────────

def fit_hmm(feats, n_states, seed=42):
    X = feats[["log_ret", "realized_vol", "ret_5d"]].values
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=seed,
        tol=1e-5,
    )
    model.fit(X)
    states = model.predict(X)
    return model, states


def label_states(model, n_states):
    """Label HMM states by mean log_ret: highest = Bull, lowest = Bear, rest = Neutral_N."""
    means = model.means_[:, 0]  # log_ret dimension
    order = np.argsort(means)   # ascending: index 0 = lowest return state
    labels = {}
    if n_states == 2:
        labels[order[0]] = "Bear/Neutral"
        labels[order[1]] = "Bull/Neutral"
    elif n_states == 3:
        labels[order[0]] = "Bear"
        labels[order[1]] = "Sideways"
        labels[order[2]] = "Bull"
    else:
        labels[order[0]] = "Strong_Bear"
        labels[order[1]] = "Weak_Bear"
        labels[order[2]] = "Weak_Bull"
        labels[order[3]] = "Strong_Bull"
    return labels


def print_model_summary(model, state_labels, n_states, feats, states):
    X = feats[["log_ret", "realized_vol", "ret_5d"]].values
    print(f"\n{'─'*60}")
    print(f"  HMM  n_states={n_states}   log-likelihood={model.score(X):.1f}")
    print(f"  State means (log_ret, realized_vol, ret_5d):")
    for s in range(n_states):
        lbl = state_labels[s]
        m_r  = model.means_[s, 0]
        m_v  = model.means_[s, 1]
        m_m  = model.means_[s, 2]
        cnt  = (states == s).sum()
        pct  = cnt / len(states) * 100
        print(f"    [{s}] {lbl:<14}  ret={m_r:+.4f}  vol={m_v:.4f}  mom5={m_m:+.4f}  ({cnt:,}d  {pct:.0f}%)")
    print(f"  Transition matrix:")
    for i in range(n_states):
        row = "  ".join(f"{model.transmat_[i,j]:.2f}" for j in range(n_states))
        print(f"    {state_labels[i]:<14} → [{row}]")


# ── scan archive join ─────────────────────────────────────────────────────────

def load_archive():
    df = pd.read_csv(RES_DIR / "scan_archive_backfilled.csv", low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    df["date"] = df["logged_at"].dt.tz_convert("UTC").dt.date
    df = df[df["resolved_yes"].notna() & (df["resolved_yes"].astype(str).str.strip() != "")].copy()
    for col in ["p_market", "tau_minutes", "vol_eff", "resolved_yes", "spot", "strike"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["p_market", "tau_minutes", "vol_eff", "resolved_yes", "spot", "strike"])
    df = df[
        (df["p_market"] >= PM_MIN) & (df["p_market"] <= PM_MAX) &
        (df["tau_minutes"] >= TAU_MIN) & (df["tau_minutes"] <= TAU_MAX)
    ].copy()
    return df


def calc_pnl(side, pm, n, ry):
    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        return n * (1 - pm - fee) if ry == 1 else -n * (pm + fee)
    return n * (pm - fee) if ry == 0 else -n * (1 - pm + fee)


def best_edge_per_scan(df_sub):
    """Return best YES/NO edge bet per scan cycle (simplified, no gate logic)."""
    trades = []
    for ts, grp in df_sub.groupby("logged_at"):
        best_edge, best_row, best_side = MIN_EDGE, None, None
        for _, row in grp.iterrows():
            pm = row["p_market"]; ve = row["vol_eff"]; tau = row["tau_minutes"]
            ry = int(row["resolved_yes"])
            tau_h = max(tau / 60, 1 / 60)
            sig = ve * math.sqrt(tau_h)
            if sig <= 0:
                continue
            p_yes = float(norm.sf(math.log(row["strike"] / row["spot"]) / sig))
            fee = FEE_RATE * min(pm, 1 - pm)
            for side, edge, risk in [("yes", p_yes - pm - fee, pm),
                                      ("no",  pm - p_yes - fee, 1 - pm)]:
                if edge > best_edge and risk > 0:
                    n = min(edge / risk * KELLY_MULT, KELLY_CAP) * BANKROLL / risk
                    if n >= 0.01:
                        best_edge, best_side = edge, side
                        best_row = {"ts": ts, "side": side, "pm": pm, "edge": edge,
                                    "pnl": calc_pnl(side, pm, n, ry), "won": calc_pnl(side, pm, n, ry) > 0}
        if best_row:
            trades.append(best_row)
    return pd.DataFrame(trades)


def regime_stats(trades_df, regime_col, archive_df):
    """Compute YES WR and P&L per regime label using scan-by-scan best trade."""
    if trades_df.empty:
        return pd.DataFrame()
    merged = trades_df.merge(
        archive_df[["logged_at", regime_col]].drop_duplicates("logged_at"),
        left_on="ts", right_on="logged_at", how="left"
    )
    out = []
    for lbl, g in merged.groupby(regime_col):
        n = len(g); wr = g["won"].mean(); pnl = g["pnl"].sum()
        out.append({"regime": lbl, "trades": n, "wr": wr, "pnl": pnl,
                    "pnl_per_trade": pnl / n if n else 0})
    return pd.DataFrame(out).sort_values("regime")


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    close = fetch_btc_daily()
    feats = build_features(close)

    # Current ±2% threshold labels
    ret_20d = np.log(close / close.shift(20))
    threshold_labels = ret_20d.apply(current_threshold_label).rename("threshold")

    print(f"\n{'='*60}")
    print("  CURRENT ±2% THRESHOLD  (2021-present)")
    print(f"{'='*60}")
    for lbl, cnt in threshold_labels.value_counts().items():
        pct = cnt / len(threshold_labels) * 100
        print(f"  {lbl:<14} {cnt:>5,}d  ({pct:.0f}%)")

    # Fit HMMs
    hmm_results = {}
    for n in N_STATES_LIST:
        model, states = fit_hmm(feats, n)
        state_labels = label_states(model, n)
        print_model_summary(model, state_labels, n, feats, states)
        hmm_results[n] = (model, states, state_labels, feats.index)

    # Build day-level label DataFrame for the archive period
    arc = load_archive()
    arc_dates = sorted(arc["date"].unique())
    print(f"\n  Archive period: {min(arc_dates)} → {max(arc_dates)}")

    # Map HMM states to archive dates
    label_df = pd.DataFrame({"date": pd.to_datetime(feats.index).date})
    label_df["threshold"] = threshold_labels.values[:len(label_df)]

    for n, (model, states, state_labels, idx) in hmm_results.items():
        col = f"hmm_{n}state"
        label_df[col] = [state_labels[s] for s in states]

    label_df = label_df[label_df["date"].isin(arc_dates)].copy()

    print(f"\n{'='*60}")
    print("  REGIME LABELS  (archive window, by day)")
    print(f"{'='*60}")
    cols = ["date", "threshold"] + [f"hmm_{n}state" for n in N_STATES_LIST]
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(label_df[cols].to_string(index=False))

    # Join to archive and compute YES WR per regime
    arc = arc.merge(
        label_df[cols].rename(columns={"date": "arc_date"}),
        left_on="date", right_on="arc_date", how="left"
    )

    # Best-edge-per-scan trades (no gate filter — raw model edge)
    print(f"\n  Simulating ungated scan-by-scan trades...")
    trades = best_edge_per_scan(arc)
    print(f"  {len(trades)} total trades")

    print(f"\n{'='*60}")
    print("  YES WIN RATE + P&L BY REGIME  (ungated, best edge per scan)")
    print(f"{'='*60}")

    all_rows = []
    for col in ["threshold"] + [f"hmm_{n}state" for n in N_STATES_LIST]:
        stats = regime_stats(trades, col, arc)
        if stats.empty:
            continue
        print(f"\n  [{col}]")
        print(f"  {'Regime':<16} {'n':>6} {'WR':>7} {'P&L':>10} {'$/trade':>8}")
        print(f"  {'-'*50}")
        for _, r in stats.iterrows():
            print(f"  {r['regime']:<16} {r['trades']:>6} {r['wr']:>7.1%} "
                  f"{r['pnl']:>+10,.0f} {r['pnl_per_trade']:>+8.2f}")
        stats["model"] = col
        all_rows.append(stats)

    # Save
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out_path = RES_DIR / f"hmm_regime_comparison_{date.today().strftime('%Y%m%d')}.csv"
    out.to_csv(out_path, index=False)

    # Also save day-level labels
    label_path = RES_DIR / f"hmm_regime_labels_{date.today().strftime('%Y%m%d')}.csv"
    label_df[cols].to_csv(label_path, index=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {label_path}")


if __name__ == "__main__":
    run()
