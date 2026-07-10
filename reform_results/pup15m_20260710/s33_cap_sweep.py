"""
S33 -- simulate tightening the empirical z_drift cap (currently 0.5,
saturated 73.8% of decisions) before touching anything live.

Method:
1. Reconstruct the CAUSAL raw (uncapped) z_drift trajectory exactly as
   compute_zdrift_empirical_15m does: alpha*z_short(10) + (1-alpha)*z_long(30)
   from ALL resolved trades (any side) up to each point in time.
2. Join this trajectory to EVERY candidate in btc_scan_archive_15m.csv
   (zero-lookahead, backward asof) -- not just taken trades, so the full
   selection process can be replayed.
3. For each candidate cap value, re-derive p_model_yes via the REAL
   compute_p_yes_zdrift_15m (imported directly, no hand math) with
   z_drift = clip(raw, -cap, cap); p_model_no = 1 - p_model_yes (the
   runner's non-coherent-fallback convention); recompute edges; select
   best side/edge per ticker (best across scan cycles -- same approximation
   used throughout this session's backtests, not exact cooldown replication).
4. Compare: trade count, side mix, WR, BE, $, and touch-rate (path
   reconstruction) across cap in {0.50 (current), 0.35, 0.25, 0.20, 0.15, 0.10}.
"""
import warnings
import sys
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
OUT = "reform_results/pup15m_20260710"

import paper_trade_runner_15m as R  # noqa: E402

EDGE_THRESH = 0.04
CAPS = [0.50, 0.35, 0.25, 0.20, 0.15, 0.10]

# ---- 1. causal raw z_drift trajectory ----
df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
res = df.dropna(subset=["resolved_yes", "spot", "realized_vol_annual", "tau_minutes",
                        "spot_at_expiry", "decision_time"]).copy()
res = res[(res["spot"] > 0) & (res["realized_vol_annual"] > 0) & (res["tau_minutes"] > 0) & (res["spot_at_expiry"] > 0)]
res = res.sort_values("decision_time").reset_index(drop=True)
sigma_tau = res["realized_vol_annual"] * np.sqrt(res["tau_minutes"] / 525600.0)
res["actual_z"] = np.log(res["spot_at_expiry"] / res["spot"]) / sigma_tau
z_short = res["actual_z"].rolling(10).mean()
z_long = res["actual_z"].rolling(30, min_periods=30).mean()
res["z_drift_raw"] = 0.6 * z_short + 0.4 * z_long
zseries = res.dropna(subset=["z_drift_raw"])[["decision_time", "z_drift_raw"]].drop_duplicates("decision_time")
print(f"z_drift_raw trajectory: {len(zseries)} points  {zseries['decision_time'].min()} -> {zseries['decision_time'].max()}")

# ---- 2. join to every candidate in the scan archive ----
arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "resolved_yes", "realized_vol_annual"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "tau_minutes",
                         "spot", "strike", "realized_vol_annual"])
arc = arc[arc["realized_vol_annual"] > 0].sort_values("logged_at")
arc = pd.merge_asof(arc, zseries.sort_values("decision_time"), left_on="logged_at",
                    right_on="decision_time", direction="backward").dropna(subset=["z_drift_raw"])
print(f"candidate rows with valid z_drift_raw: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")

# ---- 1m path data for touch reconstruction ----
p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
c1 = px["close"]

def build_book(cap):
    d = arc.copy()
    d["zd"] = d["z_drift_raw"].clip(-cap, cap)
    p_yes, p_no = [], []
    for row in d.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        py = R.compute_p_yes_zdrift_15m(row.spot, row.strike, row.tau_minutes, sig, row.zd, row.p_market)
        p_yes.append(py); p_no.append(1.0 - py)
    d["p_model_yes"], d["p_model_no"] = p_yes, p_no
    d["edge_yes"] = d["p_model_yes"] - d["p_market"]
    d["edge_no"] = d["p_model_no"] - (1.0 - d["p_market"])
    d["best_side"] = np.where(d["edge_yes"] >= d["edge_no"], "yes", "no")
    d["best_edge"] = np.where(d["edge_yes"] >= d["edge_no"], d["edge_yes"], d["edge_no"])
    qual = d[d["best_edge"] >= EDGE_THRESH].copy()
    per_ticker = qual.sort_values("best_edge", ascending=False).drop_duplicates("contract_ticker")
    per_ticker["win"] = np.where(per_ticker["best_side"] == "yes", per_ticker["resolved_yes"],
                                 1 - per_ticker["resolved_yes"])
    per_ticker["cost"] = np.where(per_ticker["best_side"] == "yes", per_ticker["p_market"],
                                  1 - per_ticker["p_market"])
    return per_ticker

print(f"\n{'cap':>5s} {'n':>4s} {'yes':>4s} {'no':>4s} {'WR':>7s} {'BE':>7s} {'edge':>8s} {'touched%':>9s}")
results = {}
for cap in CAPS:
    b = build_book(cap)
    yes_n = (b["best_side"] == "yes").sum()
    no_n = (b["best_side"] == "no").sum()
    wr, be = b["win"].mean(), b["cost"].mean()

    # touch reconstruction for this cap's selected trades
    touches = []
    for row in b.itertuples(index=False):
        tau = float(row.tau_minutes)
        close_t = row.logged_at + pd.Timedelta(minutes=tau)
        path = c1[(c1.index >= row.logged_at) & (c1.index <= close_t + pd.Timedelta(minutes=1))]
        if len(path) < 3:
            touches.append(np.nan); continue
        if row.best_side == "yes":
            touches.append(path.min() <= row.strike)
        else:
            touches.append(path.max() >= row.strike)
    b["touched"] = touches
    touched_pct = pd.Series(touches).mean()
    results[cap] = b
    print(f"{cap:5.2f} {len(b):4d} {yes_n:4d} {no_n:4d} {wr:7.1%} {be:7.1%} {wr-be:+8.4f} {touched_pct:9.1%}")

print("\n=== era split: pre-06-30 (reform) vs post ===")
for cap in CAPS:
    b = results[cap]
    b["era"] = np.where(b["logged_at"] < pd.Timestamp("2026-06-30", tz="UTC"), "pre", "post")
    for era, g in b.groupby("era"):
        if len(g) < 10:
            continue
        print(f"  cap={cap:.2f} {era:4s}: n={len(g):4d}  WR={g['win'].mean():.1%}  "
              f"edge={g['win'].mean()-g['cost'].mean():+.4f}  touched%={g['touched'].mean():.1%}")

for cap in CAPS:
    results[cap].to_csv(f"{OUT}/s33_cap_{str(cap).replace('.','p')}.csv", index=False)
print("\nDONE_S33")
