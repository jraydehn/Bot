"""
test_regime_count_pnl_hmm.py (scratch) — does MORE *sticky HMM* regimes help PnL?
Gold-standard answer to the regime-count challenge: real GaussianHMM (Viterbi/sticky),
fit OOS on bar features (<=05-16), backtest on scan archive 05-18..06-16.
"""
import math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; SMOOTH=30
FEATS=["ret_24h","ret_72h","rv24","sharpe_24h"]
def cell(t,r): return (int(np.clip(t,-5,5)), int(np.clip(r,-11,11)))

frame=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
lab=lab.dropna(subset=FEATS).sort_index()
CUT=pd.Timestamp("2026-05-16 21:00",tz="UTC")

a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"])
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)&(a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
a["ry"]=a["resolved_yes"].round().astype(int)

def pyes(pup,spot,strike,vol,tau,K):
    sig=vol*math.sqrt(tau)
    if sig<=0: return 0.5
    return float(norm.sf(math.log(strike/spot)/sig - norm.ppf(min(max(pup,0.01),0.99))*K*math.sqrt(tau/60.0)))
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))

def run(Kstates, DRIFT_K):
    sc=StandardScaler().fit(lab[lab.index<=CUT][FEATS])
    if Kstates==1:
        lab_state=pd.Series(0,index=lab.index)
    else:
        hmm=GaussianHMM(n_components=Kstates,covariance_type="full",n_iter=80,random_state=0)
        hmm.fit(sc.transform(lab[lab.index<=CUT][FEATS]))
        lab_state=pd.Series(hmm.predict(sc.transform(lab[FEATS])),index=lab.index)  # Viterbi over timeline
    # map states by nearest-past hour (np.searchsorted on raw values to avoid tz issues)
    lab_idx=lab_state.index.values
    fr=frame.copy()
    fpos=np.clip(np.searchsorted(lab_idx, fr.index.values, side="right")-1, 0, len(lab_state)-1)
    fr["st"]=lab_state.values[fpos]
    base=fr["next_up"].mean(); rb=fr.groupby("st")["next_up"].mean().to_dict()
    fr["c"]=[cell(t,r) for t,r in zip(fr.trend,fr.rev)]
    grp=fr.groupby(["st","c"])["next_up"].agg(["mean","size"])
    tab={(s,c):(row["size"]*row["mean"]+SMOOTH*rb.get(s,base))/(row["size"]+SMOOTH) for (s,c),row in grp.iterrows()}
    idx=np.clip(np.searchsorted(lab_idx, a["dt"].values, side="right")-1, 0, len(lab_state)-1)
    ast=lab_state.values[idx]
    pup=np.array([tab.get((s,cell(t,r)),rb.get(s,base)) for t,r,s in zip(a.composite_trend,a.composite_rev,ast)])
    py=np.array([pyes(p,s,k,v,t,DRIFT_K) for p,s,k,v,t in zip(pup,a.spot,a.strike,a.vol_eff,a.tau_minutes)])
    eyes=py-a.p_market.values-FEE*np.minimum(a.p_market.values,1-a.p_market.values)
    eno =a.p_market.values-py-FEE*np.minimum(a.p_market.values,1-a.p_market.values)
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"logged_at":a.logged_at.values,"ticker":a.contract_ticker.values,"close_ts":a.close_ts.values,
        "pm":a.p_market.values,"ry":a.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; n=0; w=0
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP); u=unit(r.side,r.pm,r.ry)
            pnl+=(bf*BANKROLL/cost)*u; n+=1; w+=u>0; traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1
            break
    return n,(w/n if n else 0),pnl

print(f"OOS contracts: {len(a)}")
print(f"\n{'HMM states':>11}{'trades':>8}{'WR':>7}{'PnL$ (K=1.0)':>14}{'PnL$ (K=1.4)':>14}")
for Ks in [1,3,5,8]:
    n0,wr0,p0=run(Ks,1.0); n1,wr1,p1=run(Ks,1.4)
    print(f"{Ks:>11}{n0:>8}{wr0:>7.1%}{p0:>+14.0f}{p1:>+14.0f}")
