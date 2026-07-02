"""
phase2_model.py — Train a p_yes calibration model from live Kalshi trade data.

Target:  resolved_yes (binary)
Edge:    p_lgbm - p_market - fee  (YES)  |  p_market - p_lgbm - fee  (NO)
Split:   chronological 70/30  (no random fold — time series)
Model:   shallow LightGBM, max_depth=3

Backtest compares the new model against:
  B  Pure lognormal (no drift)
  G  Pure lognormal + p_up_v2 gate (current best)
"""
import glob, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import calibration_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"

BANKROLL   = 1_000.0
KELLY_MULT = 0.30
KELLY_CAP  = 0.06
FEE_RATE   = 0.07
MIN_EDGE   = 0.005

MODEL_FEATURES = [
    # Donchian — strongest signals from Phase 1
    "dc_4h_n20_pos",
    "dc_4h_n20_break",
    "dc_4h_n55_pos",
    "dc_4h_n55_width",
    "dc_1h_n55_pos",
    "dc_1h_n20_pos",
    "dc_15m_n20_pos",
    "dc_15m_n55_pos",
    "dc_15m_n55_break",
    "dc_15m_n20_break",
    # Momentum / structure
    "stoch_k",
    "stoch_bias",
    "vwap_score",
    "vwap_distance_pct",
    "ema_stack_bias",
    "ema_stretch_score",
    "composite_rev",
    # Directional / vol
    "p_up_v2_backfilled",
    "rvol_inv_backfilled",
    # Market context
    "p_market",
    "tau_minutes",
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES_DIR / "paper_trades_archive_*.csv")))
    files += [str(RES_DIR / "paper_trades.csv")]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["logged_at"] = pd.to_datetime(combined["logged_at"], format="mixed", utc=True)
    combined = (combined
                .sort_values("logged_at")
                .drop_duplicates(subset=["contract_ticker", "logged_at", "side"], keep="last"))

    trades = combined[combined["decision"] == "trade"].copy()
    trades = trades[trades["resolved_yes"].notna()].copy()

    for col in ["p_market", "resolved_yes", "spot", "strike", "tau_minutes", "vol_eff"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(subset=["p_market", "resolved_yes", "spot", "strike",
                                   "tau_minutes", "vol_eff"])

    # Merge backfilled features
    pup_path = RES_DIR / "p_up_v2_backfilled.csv"
    if pup_path.exists():
        pup = pd.read_csv(pup_path)
        pup["logged_at"] = pd.to_datetime(pup["logged_at"], utc=True)
        trades = trades.merge(pup, on=["contract_ticker", "logged_at", "side"], how="left")

    # Coerce all model features to numeric
    for col in MODEL_FEATURES:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
        else:
            trades[col] = float("nan")

    trades["won"] = np.where(
        trades["side"] == "yes",
        trades["resolved_yes"] == 1,
        trades["resolved_yes"] == 0,
    )
    return trades.reset_index(drop=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def sigma_tau(row) -> float:
    tau_h = max(float(row["tau_minutes"]) / 60.0, 1/60)
    return float(row["vol_eff"]) * math.sqrt(tau_h)


def p_lognormal(spot, strike, sig_t) -> float:
    if sig_t <= 0:
        return 1.0 if spot > strike else 0.0
    z = math.log(strike / spot) / sig_t
    return float(norm.sf(z))


def kelly_size(edge, pm_risk) -> float:
    if pm_risk <= 0:
        return 0.0
    return min(edge / pm_risk * KELLY_MULT, KELLY_CAP) * BANKROLL / pm_risk


def compute_pnl(side, pm, n, resolved_yes) -> float:
    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        return n * (1 - pm - fee) if resolved_yes == 1 else -n * (pm + fee)
    else:
        return n * (pm - fee) if resolved_yes == 0 else -n * (1 - pm + fee)


# ── train ─────────────────────────────────────────────────────────────────────

def train(trades: pd.DataFrame):
    n_train = int(len(trades) * 0.70)
    train_df = trades.iloc[:n_train].copy()
    test_df  = trades.iloc[n_train:].copy()

    print(f"  Train: {len(train_df):,} trades  "
          f"({train_df['logged_at'].min().date()} → {train_df['logged_at'].max().date()})")
    print(f"  Test:  {len(test_df):,} trades  "
          f"({test_df['logged_at'].min().date()} → {test_df['logged_at'].max().date()})")

    X_train = train_df[MODEL_FEATURES].values.astype(float)
    y_train = train_df["resolved_yes"].values.astype(float)
    X_test  = test_df[MODEL_FEATURES].values.astype(float)

    feat_coverage = [(f, (~np.isnan(X_train[:, i])).mean())
                     for i, f in enumerate(MODEL_FEATURES)]
    print(f"\n  Feature coverage on train set:")
    for feat, cov in feat_coverage:
        mark = "" if cov > 0.7 else "  ← sparse"
        print(f"    {feat:<28} {cov:.0%}{mark}")

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    # Importance
    importances = sorted(zip(MODEL_FEATURES, clf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importances (gain):")
    for feat, imp in importances:
        bar = "█" * int(imp / max(i for _, i in importances) * 20)
        print(f"    {feat:<28} {imp:>6.0f}  {bar}")

    return clf, train_df, test_df


# ── backtest ──────────────────────────────────────────────────────────────────

def backtest_model(clf, trades: pd.DataFrame, label: str) -> pd.DataFrame:
    X = trades[MODEL_FEATURES].values.astype(float)
    trades = trades.copy()
    trades["p_lgbm"] = clf.predict_proba(X)[:, 1]

    results = []
    for _, row in trades.iterrows():
        pm   = float(row["p_market"])
        side = str(row["side"])
        p_m  = float(row["p_lgbm"])
        sig  = sigma_tau(row)
        if sig <= 0:
            continue

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE or pm_risk <= 0:
            continue

        n   = kelly_size(edge, pm_risk)
        if n < 0.01:
            continue
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        results.append({
            "model": label, "side": side, "pm": pm, "edge": edge,
            "p_lgbm": p_m, "n_cont": n, "pnl": pnl, "won": pnl > 0,
            "logged_at": row["logged_at"],
            "month": row["logged_at"].to_period("M"),
        })
    return pd.DataFrame(results)


def backtest_lognormal(trades: pd.DataFrame, label: str,
                       pup_gate: bool = False) -> pd.DataFrame:
    results = []
    for _, row in trades.iterrows():
        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            continue

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig)

        if pup_gate:
            pup = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
            if pd.isna(pup):
                continue
            if side == "yes" and pup <= 0.50:
                continue
            if side == "no" and pup >= 0.50:
                continue

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE or pm_risk <= 0:
            continue

        n   = kelly_size(edge, pm_risk)
        if n < 0.01:
            continue
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        results.append({
            "model": label, "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": pnl > 0,
            "logged_at": row["logged_at"],
            "month": row["logged_at"].to_period("M"),
        })
    return pd.DataFrame(results)


def print_summary(dfs: dict, test_df: pd.DataFrame):
    months = sorted(test_df["logged_at"].dt.to_period("M").unique())

    print(f"\n{'='*72}")
    print(f"  OUT-OF-SAMPLE BACKTEST  (test period only)")
    print(f"{'='*72}")
    print(f"  {'Model':<20} {'Trades':>7} {'WR':>7} {'P&L':>10} {'$/trade':>8}")
    print(f"  {'-'*56}")

    for label, df in dfs.items():
        if df.empty:
            print(f"  {label:<20}  no trades")
            continue
        n   = len(df)
        wr  = df["won"].mean()
        pnl = df["pnl"].sum()
        print(f"  {label:<20} {n:>7,} {wr:>7.1%} {pnl:>+10,.0f} {pnl/n:>+8.2f}")

    print(f"\n{'='*72}")
    print(f"  MONTHLY P&L (test period)")
    print(f"{'='*72}")
    headers = "".join(f"  {m:<14}" for m in dfs)
    print(f"  {'Month':<8}" + headers)
    print(f"  {'-'*(10 + 16*len(dfs))}")
    for m in months:
        row_str = f"  {str(m):<8}"
        for label, df in dfs.items():
            sub = df[df["month"] == m] if not df.empty else pd.DataFrame()
            if sub.empty:
                row_str += f"  {'—':>14}"
            else:
                row_str += f"  {sub['pnl'].sum():>+8,.0f} ({len(sub):>3})"
        print(row_str)


def calibration_report(clf, test_df: pd.DataFrame):
    X = test_df[MODEL_FEATURES].values.astype(float)
    p_pred = clf.predict_proba(X)[:, 1]
    y_true = test_df["resolved_yes"].values

    print(f"\n{'='*72}")
    print(f"  CALIBRATION  (predicted p_yes vs actual resolution rate)")
    print(f"{'='*72}")
    print(f"  {'Pred range':<18} {'N':>6} {'Pred mean':>10} {'Actual rate':>12}")
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() < 5:
            continue
        print(f"  {lo:.1f}–{hi:.1f}              {mask.sum():>6,} "
              f"{p_pred[mask].mean():>10.3f} {y_true[mask].mean():>12.3f}")


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading trades + backfilled features...")
    trades = load_trades()
    print(f"  {len(trades):,} resolved trades  "
          f"({trades['logged_at'].min().date()} → {trades['logged_at'].max().date()})")

    print("\nTraining model (chronological 70/30 split)...")
    clf, train_df, test_df = train(trades)

    print("\nBacktesting on test set...")
    results = {
        "K_lgbm":    backtest_model(clf, test_df, "K_lgbm"),
        "B_purelogN": backtest_lognormal(test_df, "B_purelogN", pup_gate=False),
        "G_pup2gate": backtest_lognormal(test_df, "G_pup2gate", pup_gate=True),
    }

    print_summary(results, test_df)
    calibration_report(clf, test_df)

    # Edge quartile analysis for K_lgbm
    dfK = results["K_lgbm"]
    if not dfK.empty:
        print(f"\n{'='*72}")
        print(f"  MODEL K — P&L BY EDGE QUARTILE")
        print(f"{'='*72}")
        dfK["edge_q"] = pd.qcut(dfK["edge"], 4, labels=["Q1 low","Q2","Q3","Q4 high"])
        for q, g in dfK.groupby("edge_q", observed=True):
            pnl = g["pnl"].sum(); wr = g["won"].mean(); n = len(g)
            print(f"  {str(q):<10}  n={n:,}  WR={wr:.1%}  P&L=${pnl:+,.0f}  "
                  f"mean_edge={g['edge'].mean():.3f}")

        print(f"\n{'='*72}")
        print(f"  MODEL K — YES vs NO SPLIT")
        print(f"{'='*72}")
        for side, g in dfK.groupby("side"):
            pnl = g["pnl"].sum(); wr = g["won"].mean(); n = len(g)
            be  = g["pm"].mean() if side == "yes" else 1 - g["pm"].mean()
            print(f"  {side:4s}  n={n:,}  WR={wr:.1%}  breakeven={be:.1%}  "
                  f"P&L=${pnl:+,.0f}  $/trade=${pnl/n:+.2f}")

    # Save model predictions on full dataset for further analysis
    X_all = trades[MODEL_FEATURES].values.astype(float)
    trades["p_lgbm"] = clf.predict_proba(X_all)[:, 1]
    trades[["contract_ticker", "logged_at", "side", "p_lgbm"]].to_csv(
        RES_DIR / "phase2_lgbm_predictions.csv", index=False)
    print(f"\n  Wrote results/phase2_lgbm_predictions.csv")


if __name__ == "__main__":
    run()
