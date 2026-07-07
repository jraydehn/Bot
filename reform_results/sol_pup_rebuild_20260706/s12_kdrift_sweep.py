"""
S12 -- Fresh k_drift sweep for sol_p_up_v1 as the score_to_p_model drift
source (replacing the naive reuse of DRIFT_MULTIPLIER["SOL"]=0.20, which
s11 showed is miscalibrated for sol_p_up_v1's narrower output range).

Distinguishing this from the standing caution in feedback_k_drift_calibration
(don't sweep k on synthetic sim data / <8-day archives / no real p_up logged):
this sweep uses REAL Kalshi market prices (p_market from actual historical
scans, not a synthetic offset grid built from our own vol formula) and
127,577 real scanned candidates with real resolved outcomes over ~6.5 weeks
-- the two specific problems that memory flagged (synthetic prices, <2k rows)
don't apply here. Still only ~6.5 weeks of history, so this uses a proper
temporal train/test split (not just in-sample best-k) and reports both,
flagging the window-length caveat honestly rather than treating it as a
final answer.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE

OUT = "reform_results/sol_pup_rebuild_20260706"
rng = np.random.default_rng(7)

df = pd.read_csv(f"{OUT}/sim_pup_swap_full.csv", low_memory=False)
df["logged_at_parsed"] = pd.to_datetime(df["logged_at_parsed"], utc=True, errors="coerce")
df = df.dropna(subset=["p_up_new", "spot", "strike", "sigma_tau", "p_market", "resolved_yes"]).copy()
df = df.sort_values("logged_at_parsed")
print(f"rows: {len(df)}  ({df['logged_at_parsed'].min()} -> {df['logged_at_parsed'].max()})")

split_ts = df["logged_at_parsed"].quantile(0.70, interpolation="nearest")
train = df[df["logged_at_parsed"] <= split_ts].copy()
test = df[df["logged_at_parsed"] > split_ts].copy()
print(f"train: {len(train)} rows through {train['logged_at_parsed'].max()}")
print(f"test:  {len(test)} rows from {test['logged_at_parsed'].min()}  (held out)")

z_strike_train = np.log(train["strike"] / train["spot"]) / train["sigma_tau"]
z_strike_test = np.log(test["strike"] / test["spot"]) / test["sigma_tau"]
pup_z_train = norm.ppf(train["p_up_new"].clip(0.001, 0.999))
pup_z_test = norm.ppf(test["p_up_new"].clip(0.001, 0.999))

K_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00]


def sim(sub, z_strike, pup_z, k):
    z_drift = pup_z * k
    z_adj = z_strike - z_drift
    p_model = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    pm = sub["p_market"].values
    fee = np.array([kalshi_fee(float(x)) for x in pm])
    edge_yes = p_model - pm
    edge_no = pm - p_model
    net_yes = edge_yes - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD
    net_no = edge_no - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD
    take = np.maximum(net_yes, net_no) >= MIN_NET_EDGE
    side_yes = net_yes >= net_no
    if take.sum() == 0:
        return {"n": 0, "wr": np.nan, "be": np.nan, "edge": np.nan, "pnl": 0.0, "pos_weeks": np.nan}
    resolved = sub["resolved_yes"].values.astype(float)
    side_is_yes = side_yes[take]
    price_side = np.where(side_is_yes, pm[take], 1 - pm[take])
    won = np.where(side_is_yes, resolved[take] == 1, resolved[take] == 0)
    contracts = 50.0 / price_side
    pnl_arr = np.where(won, contracts * (1 - price_side), -50.0)
    wk = sub.loc[sub.index[take], "yw"] if "yw" in sub.columns else None
    n = int(take.sum())
    wr = float(won.mean())
    be = float(price_side.mean())
    pnl = float(pnl_arr.sum())
    pos_weeks = np.nan
    if wk is not None:
        wk_pnl = pd.Series(pnl_arr, index=wk.index).groupby(wk).sum()
        pos_weeks = float((wk_pnl > 0).mean())
    return {"n": n, "wr": wr, "be": be, "edge": wr - be, "pnl": pnl, "pos_weeks": pos_weeks}


print("\n=== TRAIN (in-sample selection) ===")
train_rows = []
for k in K_GRID:
    r = sim(train, z_strike_train, pup_z_train, k)
    r["k"] = k
    train_rows.append(r)
    print(f"  k={k:.2f}  n={r['n']:5d}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
          f"edge={r['edge']:+.3f}  PnL=${r['pnl']:9.2f}  pos_weeks={r['pos_weeks']:.2f}" if r["n"] else f"  k={k:.2f} n=0")

train_df = pd.DataFrame(train_rows)
best_k = float(train_df.loc[train_df["pnl"].idxmax(), "k"])
print(f"\nbest k by train PnL: {best_k}")

print("\n=== TEST (held-out, same k grid) ===")
test_rows = []
for k in K_GRID:
    r = sim(test, z_strike_test, pup_z_test, k)
    r["k"] = k
    test_rows.append(r)
    print(f"  k={k:.2f}  n={r['n']:5d}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
          f"edge={r['edge']:+.3f}  PnL=${r['pnl']:9.2f}  pos_weeks={r['pos_weeks']:.2f}" if r["n"] else f"  k={k:.2f} n=0")

test_df = pd.DataFrame(test_rows)
print(f"\ntrain-selected k={best_k} on TEST: {test_df[test_df['k']==best_k].to_dict('records')}")
best_k_test = float(test_df.loc[test_df["pnl"].idxmax(), "k"])
print(f"best k by TEST PnL alone (for comparison, NOT the selection criterion): {best_k_test}")

train_df.to_csv(f"{OUT}/kdrift_sweep_train.csv", index=False)
test_df.to_csv(f"{OUT}/kdrift_sweep_test.csv", index=False)
print(f"\nsaved sweep tables")
