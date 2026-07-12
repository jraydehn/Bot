"""
S16 -- does the SOL 15m model actually work correctly when applied to the
HOURLY ladder, not just a single hand-picked example? SOL 15m contracts are
always exactly 1 strike per window; SOL hourly averages 5.4 (up to 9) --
a genuinely different selection structure, not just a different tau. This
retroactively applies compute_signals + compute_p_model_15m to the REAL
historical hourly tau<=20min population and checks calibration against
actual resolved_yes outcomes -- the same ground-truth-anchored method used
throughout this investigation, not just a sanity check that it runs.
"""
import glob
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import paper_trade_runner_15m as r15

r15._LGBM_MODELS["SOL"] = r15._load_15m_lgbm("SOL")

print("Loading SOL 1m/1h history...")
p1m = sorted(glob.glob("data/binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
p1h = sorted(glob.glob("data/binanceus_SOLUSDT_1h_2024-01-01_*.parquet"))[-1]
full_1m = pd.read_parquet(p1m).sort_index()
full_1m.index = pd.to_datetime(full_1m.index, utc=True)
full_1h = pd.read_parquet(p1h).sort_index()
full_1h.index = pd.to_datetime(full_1h.index, utc=True)

print("Loading hourly archive, tau<=20min candidates...")
dh = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                            "tau_minutes", "resolved_yes"])
dh["logged_at"] = pd.to_datetime(dh["logged_at"], utc=True, errors="coerce", format="mixed")
dh = dh.dropna(subset=["logged_at", "spot", "strike", "p_market", "tau_minutes", "resolved_yes"])
dh["p_market"] = pd.to_numeric(dh["p_market"], errors="coerce")
dh["tau_minutes"] = pd.to_numeric(dh["tau_minutes"], errors="coerce")
dh = dh[dh["tau_minutes"] <= 20].sort_values("logged_at")

# subsample unique timestamps for tractability -- every 8th, spread evenly,
# still representative (~1500 timestamps -> several thousand candidate rows)
uniq_ts = dh["logged_at"].drop_duplicates().sort_values().reset_index(drop=True)
sampled_ts = set(uniq_ts.iloc[::8])
sub = dh[dh["logged_at"].isin(sampled_ts)].copy()
print(f"sampled timestamps: {len(sampled_ts)}  candidate rows: {len(sub)}  "
      f"tickers: {sub['contract_ticker'].nunique()}")

results = []
n_err = 0
for ts, grp in sub.groupby("logged_at"):
    live_1m = full_1m[full_1m.index <= ts].tail(1000)
    live_1h = full_1h[full_1h.index <= ts].tail(250)
    if len(live_1m) < 30:
        continue
    try:
        sig = r15.compute_signals(live_1m, asset="SOL", live_1h=live_1h)
    except Exception:
        n_err += 1
        continue
    for row in grp.itertuples(index=False):
        try:
            p = r15.compute_p_model_15m(row.spot, row.strike, row.tau_minutes, sig, asset="SOL", p_market=row.p_market)
        except Exception:
            n_err += 1
            continue
        results.append(dict(contract_ticker=row.contract_ticker, p_market=row.p_market,
                             resolved_yes=row.resolved_yes, p_hybrid=p, tau=row.tau_minutes))

print(f"errors during compute: {n_err}")
res = pd.DataFrame(results)
print(f"successfully scored: {len(res)}  tickers: {res['contract_ticker'].nunique()}")

tk = res.groupby("contract_ticker").agg(p=("p_hybrid", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
brier_hybrid = float(np.mean((tk["p"] - tk["y"]) ** 2))
brier_pm = float(np.mean((tk["pm"] - tk["y"]) ** 2))
auc_ok = tk["y"].nunique() > 1
print(f"\n=== FULL population (all offsets, ground-truth outcomes) ===")
print(f"Brier -- 15m model applied to hourly ladder: {brier_hybrid:.4f}   p_market alone: {brier_pm:.4f}")

unc = res[res["p_market"].between(0.35, 0.65)]
tk_unc = unc.groupby("contract_ticker").agg(p=("p_hybrid", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
print(f"\n=== UNCERTAIN ZONE (p_market 0.35-0.65): n={len(unc)} tickers={len(tk_unc)} ===")
if len(tk_unc) >= 15:
    brier_u = float(np.mean((tk_unc["p"] - tk_unc["y"]) ** 2))
    brier_pm_u = float(np.mean((tk_unc["pm"] - tk_unc["y"]) ** 2))
    corr_h = np.corrcoef(tk_unc["p"] - 0.5, tk_unc["y"])[0, 1]
    corr_pm = np.corrcoef(tk_unc["pm"] - 0.5, tk_unc["y"])[0, 1]
    print(f"  Brier: 15m-hybrid={brier_u:.4f}  p_market={brier_pm_u:.4f}")
    print(f"  corr:  15m-hybrid={corr_h:+.4f}  p_market={corr_pm:+.4f}")
else:
    print("  too thin to evaluate")

print(f"\n=== decile check: hybrid p vs actual outcome, full population ===")
res["decile"] = pd.qcut(res["p_hybrid"], 10, duplicates="drop")
for d, g in res.groupby("decile", observed=True):
    t = g.groupby("contract_ticker").agg(p=("p_hybrid", "mean"), y=("resolved_yes", "mean"))
    print(f"  {str(d):>22s}  n={len(g):5d} tk={len(t):4d}  mean_p={t['p'].mean():.3f}  actual={t['y'].mean():.3f}")

print(f"\n=== $ sim, uncertain zone, margin-based ===")
for margin in [0.03, 0.05]:
    edge_yes = unc["p_hybrid"] - unc["p_market"]
    edge_no = (1 - unc["p_hybrid"]) - (1 - unc["p_market"])
    ty, tn = edge_yes > margin, edge_no > margin
    bets = []
    if ty.sum() > 0:
        s = unc[ty]; bets.append(pd.DataFrame({"win": s["resolved_yes"], "cost": s["p_market"], "tk": s["contract_ticker"]}))
    if tn.sum() > 0:
        s = unc[tn]; bets.append(pd.DataFrame({"win": 1 - s["resolved_yes"], "cost": 1 - s["p_market"], "tk": s["contract_ticker"]}))
    if not bets:
        print(f"  margin={margin}: no bets"); continue
    ab = pd.concat(bets); tkb = ab.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    nc = 100.0 / tkb["cost"]; pnl = np.where(tkb["win"] >= 0.5, nc * (1 - tkb["cost"]), -nc * tkb["cost"])
    print(f"  margin={margin}: n={len(tkb):4d}  WR={tkb['win'].mean():.1%}  BE={tkb['cost'].mean():.1%}  total=${pnl.sum():.2f}")

print("\nDONE_S16")
