"""
test_gbm_pup_pnl.py (scratch) — OOS PnL head-to-head: GBM p_up vs live lookup tables.

Train GBM on bars <= 2026-05-16 (CV frame). Test on scan archive 05-18..06-16 (OOS).
Both p_up models go through the SAME p_yes->edge->Kelly->per-expiry-cap pipeline.
Compares realized P&L (the metric that matters — IC already favored GBM +0.067).
"""
import json, math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2
FEATS=["ret_24h","ret_72h","rv24","sharpe_24h"]

# ---- 1. train GBM on bar frame (OOS: bars end 05-16) ----
frame=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
try:
    import lightgbm as lgb
    gbm=lgb.LGBMClassifier(n_estimators=250,max_depth=4,learning_rate=0.03,
        min_child_samples=80,subsample=0.8,colsample_bytree=0.8,verbose=-1)
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    gbm=H(max_depth=4,learning_rate=0.03,max_iter=300,min_samples_leaf=80)
Xcols=["trend","rev"]+FEATS
gbm.fit(frame[Xcols],frame["next_up"])
gbase=frame["next_up"].mean()

# ---- 2. live lookup tables ----
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json"))
     for r in ["Bull","Bear","Sideways"]}
def lookup_pup(tb,rb,regime):
    key=f"{int(np.clip(tb,-5,5))},{int(np.clip(rb,-11,11))}"
    t=TAB.get(regime,TAB["Sideways"])
    return t.get(key,t["__baseline__"])

# ---- 3. archive OOS, join regime feats ----
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes",
          "vol_eff","macro_regime_bull","macro_regime_sdwy","macro_regime_bear","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"])
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)]
a=a[a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC")]
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
idx=np.searchsorted(lab.index.values,a["dt"].values,side="right")-1; idx=np.clip(idx,0,len(lab)-1)
for f in FEATS: a[f]=lab[f].values[idx]
a["regime"]=lab["regime"].values[idx]   # archive macro_regime posteriors are unpopulated; use label
a=a.dropna(subset=FEATS)
a["ry"]=a["resolved_yes"].round().astype(int)
print(f"OOS archive contracts: {len(a)}  cycles: {a['logged_at'].nunique()}")

# ---- 4. p_up for each model ----
a["trend"]=a["composite_trend"]; a["rev"]=a["composite_rev"]
a["pup_gbm"]=gbm.predict_proba(a[Xcols])[:,1]
a["pup_lookup"]=[lookup_pup(t,r,reg) for t,r,reg in
    zip(a["composite_trend"],a["composite_rev"],a["regime"])]
a["pup_pooled"]=gbase

def pyes(pup,spot,strike,vol,tau,K):
    sig=vol*math.sqrt(tau)
    if sig<=0: return 0.5
    zk=math.log(strike/spot)/sig
    zd=norm.ppf(min(max(pup,0.01),0.99))*K*math.sqrt(tau/60.0)
    return float(norm.sf(zk-zd))
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm)
    return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))

def backtest(col,K):
    rows=a[["logged_at","contract_ticker","close_ts","spot","strike","p_market","vol_eff","tau_minutes","ry",col]].copy()
    rows["py"]=[pyes(p,s,k,v,t,K) for p,s,k,v,t in zip(rows[col],rows.spot,rows.strike,rows.vol_eff,rows.tau_minutes)]
    rows["e_yes"]=rows.py-rows.p_market-FEE*np.minimum(rows.p_market,1-rows.p_market)
    rows["e_no"] =rows.p_market-rows.py-FEE*np.minimum(rows.p_market,1-rows.p_market)
    rows["side"]=np.where(rows.e_yes>=rows.e_no,"yes","no")
    rows["edge"]=np.where(rows.side=="yes",rows.e_yes,rows.e_no)
    rows=rows[rows.edge>MIN_EDGE]
    traded=set(); expcount={}; pnl=0.0; n=0; wins=0
    for ts,grp in rows.sort_values("logged_at").groupby("logged_at"):
        grp=grp.sort_values("edge",ascending=False)
        for _,r in grp.iterrows():
            if r.contract_ticker in traded: continue
            ek=r.close_ts
            if expcount.get(ek,0)>=EXP_CAP: continue
            cost=r.p_market if r.side=="yes" else 1-r.p_market
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP)
            u=unit(r.side,r.p_market,r.ry)
            pnl+=(bf*BANKROLL/cost)*u; n+=1; wins+=u>0
            traded.add(r.contract_ticker); expcount[ek]=expcount.get(ek,0)+1
            break  # one best bet per cycle
    return n,wins/n if n else 0,pnl

print(f"\n{'model':<22}{'trades':>7}{'WR':>7}{'PnL$':>10}{'$/trade':>9}")
for K in [1.0,1.4]:
    print(f"-- DRIFT_K={K} --")
    for col,lbl in [("pup_pooled","pooled (no regime)"),("pup_lookup","lookup tables (LIVE)"),("pup_gbm","GBM (continuous)")]:
        n,wr,pnl=backtest(col,K)
        print(f"  {lbl:<20}{n:>7}{wr:>7.1%}{pnl:>+10.0f}{(pnl/n if n else 0):>+9.2f}")
