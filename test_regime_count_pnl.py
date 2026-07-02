"""
test_regime_count_pnl.py (scratch) — does MORE regimes help PnL (not IC)?

Same OOS setup as the GBM test: train lookup tables (humble/smoothed form that
works on PnL) with K regimes via KMeans, backtest on scan archive 05-18..06-16.
Isolates regime COUNT (cells fixed at live granularity).
"""
import json, math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; SMOOTH=30
FEATS=["ret_24h","ret_72h","rv24","sharpe_24h"]

frame=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
def cell(t,r): return (int(np.clip(t,-5,5)), int(np.clip(r,-11,11)))  # live granularity

# archive OOS
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"])
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)&(a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
idx=np.searchsorted(lab.index.values,a["dt"].values,side="right")-1; idx=np.clip(idx,0,len(lab)-1)
for f in FEATS: a[f]=lab[f].values[idx]
a=a.dropna(subset=FEATS); a["ry"]=a["resolved_yes"].round().astype(int)

def pyes(pup,spot,strike,vol,tau,K=1.0):
    sig=vol*math.sqrt(tau)
    if sig<=0: return 0.5
    return float(norm.sf(math.log(strike/spot)/sig - norm.ppf(min(max(pup,0.01),0.99))*K*math.sqrt(tau/60.0)))
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm)
    return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))

def build_and_score(Kreg, DRIFT_K=1.0):
    if Kreg==1:
        tr_reg=np.zeros(len(frame),int); te_reg=np.zeros(len(a),int)
    else:
        sc=StandardScaler().fit(frame[FEATS])
        km=KMeans(Kreg,n_init=4,random_state=0).fit(sc.transform(frame[FEATS]))
        tr_reg=km.labels_; te_reg=km.predict(sc.transform(a[FEATS]))
    base=frame["next_up"].mean()
    t2=frame.assign(reg=tr_reg, c=[cell(t,r) for t,r in zip(frame.trend,frame.rev)])
    rb=t2.groupby("reg")["next_up"].mean().to_dict()
    grp=t2.groupby(["reg","c"])["next_up"].agg(["mean","size"])
    tab={(rg,c):(row["size"]*row["mean"]+SMOOTH*rb.get(rg,base))/(row["size"]+SMOOTH) for (rg,c),row in grp.iterrows()}
    pup=np.array([tab.get((rg,cell(t,r)),rb.get(rg,base)) for t,r,rg in zip(a.composite_trend,a.composite_rev,te_reg)])
    # backtest
    py=np.array([pyes(p,s,k,v,t,DRIFT_K) for p,s,k,v,t in zip(pup,a.spot,a.strike,a.vol_eff,a.tau_minutes)])
    eyes=py-a.p_market.values-FEE*np.minimum(a.p_market.values,1-a.p_market.values)
    eno =a.p_market.values-py-FEE*np.minimum(a.p_market.values,1-a.p_market.values)
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"logged_at":a.logged_at.values,"ticker":a.contract_ticker.values,"close_ts":a.close_ts.values,
        "pm":a.p_market.values,"ry":a.ry.values,"side":side,"edge":edge})
    d=d[d.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; n=0; w=0
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP); u=unit(r.side,r.pm,r.ry)
            pnl+=(bf*BANKROLL/cost)*u; n+=1; w+=u>0; traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1
            break
    return n, (w/n if n else 0), pnl

print(f"OOS contracts: {len(a)}  cycles: {a['logged_at'].nunique()}")
print(f"\n{'K regimes':>10}{'trades':>8}{'WR':>7}{'PnL$ (K=1.0)':>14}{'PnL$ (K=1.4)':>14}")
for Kreg in [1,3,5,8,12]:
    n0,wr0,p0=build_and_score(Kreg,1.0)
    n1,wr1,p1=build_and_score(Kreg,1.4)
    print(f"{Kreg:>10}{n0:>8}{wr0:>7.1%}{p0:>+14.0f}{p1:>+14.0f}")
