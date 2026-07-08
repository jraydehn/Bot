"""
S3 -- Decode the refreshed CG parquet (through 2026-07-08) with the ALREADY-
trained HMM (no retrain -- moving the model after validation would invalidate
s2's split-half), extend the taken-trade conditioning through today, and
write the state series + trade join used by the rescue search.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(31)
OUT = "reform_results/cg_hmm_20260708"

with open(f"{OUT}/hmm_cg_flow_btc_1h.pkl", "rb") as f:
    pkg = pickle.load(f)
model, scaler, FEAT_COLS = pkg["model"], pkg["scaler"], pkg["feat_cols"]

cg = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/cg_flow_btc_1h.parquet").sort_index()
fb, fs = cg["fut_buy_usd"], cg["fut_sell_usd"]
sb, ss = cg["spot_buy_usd"], cg["spot_sell_usd"]
oi = cg["oi_close"]
ll, ls = cg["liq_long_usd"], cg["liq_short_usd"]
feat = pd.DataFrame(index=cg.index)
feat["fut_ratio_1h"]  = fb / (fb + fs).replace(0, np.nan)
feat["fut_cvd_12h"]   = (fb - fs).rolling(12).sum() / (fb + fs).rolling(12).sum().replace(0, np.nan)
feat["spot_ratio_1h"] = sb / (sb + ss).replace(0, np.nan)
feat["oi_chg_4h"]     = oi.pct_change(4, fill_method=None)
feat["liq_imb_4h"]    = (ls.rolling(4).sum() - ll.rolling(4).sum()) / (ls.rolling(4).sum() + ll.rolling(4).sum() + 1.0)
lt = ll + ls
feat["liq_tot_z_10d"] = (lt - lt.rolling(240).mean()) / lt.rolling(240).std().replace(0, np.nan)
feat = feat.dropna()

X = scaler.transform(feat[FEAT_COLS].values)
gaps = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(gaps > 7200)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 5]
valid_idx = [i for s, e in zip(starts, ends) if e - s >= 5 for i in range(s, e)]
states = model.predict(X[valid_idx], lengths=lengths)
sv = pd.DataFrame({"ts": feat.index[valid_idx], "cg_state": states})
sv.to_csv(f"{OUT}/cg_states_1h.csv", index=False)
print(f"decoded {len(sv)} hours: {sv['ts'].min()} -> {sv['ts'].max()}")

# ── join to taken trades (full window, now through today) ────────────────
p = pd.read_csv("results/paper_trades.csv", low_memory=False)
p["logged_at_p"] = pd.to_datetime(p["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["bet_amount","resolved_yes","p_yes_model","p_market","would_pnl"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna(subset=["logged_at_p"]).sort_values("logged_at_p")
p = pd.merge_asof(p, sv.rename(columns={"ts": "cg_ts"}), left_on="logged_at_p",
                  right_on="cg_ts", direction="backward", tolerance=pd.Timedelta("2h"))
t = p[(p["bet_amount"] > 0) & p["resolved_yes"].notna() & p["p_yes_model"].notna()
      & p["cg_state"].notna()].copy()
t["pw"]  = np.where(t["side"]=="no", 1-t["p_yes_model"], t["p_yes_model"])
t["be"]  = np.where(t["side"]=="no", 1-t["p_market"], t["p_market"])
t["won"] = np.where(t["side"]=="no", t["resolved_yes"]==0, t["resolved_yes"]==1)
t["tedge"] = t["won"].astype(float) - t["be"]
t["week"] = t["logged_at_p"].dt.to_period("W-FRI")
print(f"taken trades with state: {len(t)} ({t['logged_at_p'].min().date()} -> {t['logged_at_p'].max().date()})")

LBL = {0:"shortliq-quiet",1:"neutral",2:"buy-flow",3:"sell-flow",
       4:"LONG-CASCADE",5:"SHORT-SQUEEZE",6:"longliq-quiet"}
def boot(sub, n_boot=4000):
    e = sub["tedge"].values; n = len(e)
    if n < 5: return (np.nan,)*4
    means = np.array([e[rng.integers(0,n,n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means,2.5), np.percentile(means,97.5), (means<=0).mean()

print(f"\n{'state':<16}{'n':>5}{'WR':>7}{'BE':>7}{'edge':>8}{'PnL':>10}{'P(e<=0)':>9}{'wks+':>6}")
for s in sorted(t["cg_state"].unique()):
    sub = t[t["cg_state"]==s]
    if len(sub) < 8:
        print(f"{LBL.get(int(s),s):<16}{len(sub):>5}  (thin)"); continue
    m, lo, hi, pb = boot(sub)
    wk = sub.groupby("week")["would_pnl"].sum()
    print(f"{LBL.get(int(s),s):<16}{len(sub):>5}{sub['won'].mean():>7.3f}{sub['be'].mean():>7.3f}"
          f"{m:>+8.3f}{sub['would_pnl'].sum():>+10.2f}{pb:>9.3f}{(wk>0).mean():>6.2f}")

# the proposed losing bucket: NO trades in states {1,2,5}
lose = t[(t["cg_state"].isin([1,2,5])) & (t["side"]=="no")]
keep = t[~(t["cg_state"].isin([1,2,5]) & (t["side"]=="no"))]
print(f"\nPROPOSED BLOCK BUCKET (NO in neutral/buy-flow/short-squeeze): "
      f"n={len(lose)} W={lose['won'].sum()} L={(~lose['won']).sum()} PnL=${lose['would_pnl'].sum():+.2f}")
print(f"REMAINDER: n={len(keep)} PnL=${keep['would_pnl'].sum():+.2f}")
t.to_csv(f"{OUT}/taken_with_cg_state.csv", index=False)
print("saved taken_with_cg_state.csv")
