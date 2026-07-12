"""
S9 -- does the existing SOL LGBM shadow model (reform_results/sol_lgbm.pkl,
trained 05-21, test AUC=0.872) carry real tradeable edge, or is that AUC
inflated by within-hour pseudo-replication (many simultaneously-scanned
strikes share near-identical composite_trend/rev features but correlated
resolved_yes outcomes since they're driven by the same underlying hourly
move)? Same rigorous methodology as s5/s8: ticker-clustered, and isolated
to the p_market 0.35-0.65 "genuinely uncertain" zone where trading edge
would actually need to exist.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_gbdt", "composite_p_up",
                            "p_market", "resolved_yes", "offset_pct"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "p_gbdt", "composite_p_up", "p_market", "resolved_yes"])
print(f"total rows with p_gbdt: {len(df)}  tickers: {df['contract_ticker'].nunique()}")


def tk_brier(p, y, ticker):
    t = pd.DataFrame({"p": p, "y": y, "tk": ticker})
    g = t.groupby("tk").agg(p=("p", "mean"), y=("y", "mean"))
    return float(np.mean((g["p"] - g["y"]) ** 2)), len(g)


# --- full population, row-level (replicates training script's un-clustered eval) ---
from sklearn.metrics import roc_auc_score
auc_row = roc_auc_score(df["resolved_yes"], df["p_gbdt"])
print(f"\nRow-level AUC on full archive (uncorrected for clustering): {auc_row:.4f}")

# --- ticker-clustered AUC: one row per ticker (mean p_gbdt, majority-vote outcome) ---
tk_all = df.groupby("contract_ticker").agg(p=("p_gbdt", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
auc_tk = roc_auc_score((tk_all["y"] >= 0.5).astype(int), tk_all["p"])
print(f"Ticker-clustered AUC (one vote per contract): {auc_tk:.4f}")

# --- Brier: p_gbdt vs p_market vs composite_p_up, full population, ticker-clustered ---
print(f"\n=== Brier score comparison, ticker-clustered, FULL population ===")
for col, label in [("p_gbdt", "p_gbdt (LGBM shadow)"), ("composite_p_up", "composite_p_up"), ("p_market", "p_market (the market)")]:
    brier, ntk = tk_brier(df[col], df["resolved_yes"], df["contract_ticker"])
    print(f"  {label:<28s} brier={brier:.4f}  tickers={ntk}")

# --- isolate to genuinely uncertain zone (p_market 0.35-0.65) ---
unc = df[df["p_market"].between(0.35, 0.65)].copy()
print(f"\n=== genuinely uncertain zone (p_market 0.35-0.65): n={len(unc)}  tickers={unc['contract_ticker'].nunique()} ===")
for col, label in [("p_gbdt", "p_gbdt (LGBM shadow)"), ("composite_p_up", "composite_p_up"), ("p_market", "p_market (the market)")]:
    brier, ntk = tk_brier(unc[col], unc["resolved_yes"], unc["contract_ticker"])
    corr = np.corrcoef(unc[col] - 0.5, unc["resolved_yes"])[0, 1]
    print(f"  {label:<28s} brier={brier:.4f}  corr(x-0.5,y)={corr:+.4f}  tickers={ntk}")

print(f"\n=== p_gbdt decile vs actual outcome, uncertain zone only ===")
unc["gbdt_decile"] = pd.qcut(unc["p_gbdt"], 10, duplicates="drop")
for d, sub in unc.groupby("gbdt_decile", observed=True):
    tk = sub.groupby("contract_ticker").agg(p=("p_gbdt", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    print(f"  {str(d):>22s}  n={len(sub):5d} tk={len(tk):4d}  mean_p_gbdt={tk['p'].mean():.3f}  actual_up%={tk['y'].mean():.3f}  avg_pm={tk['pm'].mean():.3f}")

# --- $ sim: bet where p_gbdt disagrees with p_market by > margin, uncertain zone ---
print(f"\n=== $ PnL proxy in uncertain zone: bet side p_gbdt favors when it beats p_market by margin ===")
for margin in [0.03, 0.05, 0.08]:
    edge_yes = unc["p_gbdt"] - unc["p_market"]
    edge_no = (1 - unc["p_gbdt"]) - (1 - unc["p_market"])
    take_yes = edge_yes > margin
    take_no = edge_no > margin
    bets = []
    if take_yes.sum() > 0:
        sub = unc[take_yes]
        bets.append(pd.DataFrame({"win": sub["resolved_yes"], "cost": sub["p_market"], "tk": sub["contract_ticker"]}))
    if take_no.sum() > 0:
        sub = unc[take_no]
        bets.append(pd.DataFrame({"win": 1 - sub["resolved_yes"], "cost": 1 - sub["p_market"], "tk": sub["contract_ticker"]}))
    if not bets:
        print(f"  margin={margin}: no bets")
        continue
    allbets = pd.concat(bets)
    tk = allbets.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    print(f"  margin={margin}: n={len(tk):4d}  WR={tk['win'].mean():.1%}  BE={tk['cost'].mean():.1%}  "
          f"total=${pnl.sum():.2f}  $/bet=${pnl.sum()/len(tk):.2f}")

print("\nDONE_S9")
