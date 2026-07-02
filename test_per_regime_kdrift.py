"""
test_per_regime_kdrift.py (scratch) — do different regimes want different DRIFT_K?
Restrict the backtest to each production regime's cycles and sweep k independently.
If the PnL(k) peaks differ across regimes -> per-regime k has merit. Same peak -> no.
"""
import json, math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
def lookup(t,r,reg):
    key=f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"; tb=TAB.get(reg,TAB["Sideways"]); return tb.get(key,tb["__baseline__"])
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"])
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)&(a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
idx=np.clip(np.searchsorted(lab.index.values,a["dt"].values,side="right")-1,0,len(lab)-1)
a["regime"]=lab["regime"].values[idx]; a["ry"]=a["resolved_yes"].round().astype(int)
a["pup"]=[lookup(t,r,reg) for t,r,reg in zip(a.composite_trend,a.composite_rev,a.regime)]

def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def backtest(df,K):
    sig=df.vol_eff.values*np.sqrt(df.tau_minutes.values)
    zk=np.log(df.strike.values/df.spot.values)/sig
    zd=norm.ppf(np.clip(df.pup.values,0.01,0.99))*K*np.sqrt(df.tau_minutes.values/60.0)
    py=norm.sf(zk-zd)
    pm=df.p_market.values; fee=FEE*np.minimum(pm,1-pm)
    eyes=py-pm-fee; eno=pm-py-fee; side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"logged_at":df.logged_at.values,"ticker":df.contract_ticker.values,"close_ts":df.close_ts.values,
        "pm":pm,"ry":df.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; n=0; w=0
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP); u=unit(r.side,r.pm,r.ry)
            pnl+=(bf*BANKROLL/cost)*u; n+=1; w+=u>0; traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1
            break
    return n,pnl
KGRID=[-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0]
print("PnL by DRIFT_K, per regime (restricted to that regime's cycles):")
print(f"{'regime':>10}{'n_contracts':>12}  " + "".join(f"k={k:>4}" for k in KGRID))
for reg in ["Bull","Bear","Sideways","ALL"]:
    sub=a if reg=="ALL" else a[a.regime==reg]
    pnls=[backtest(sub,k)[1] for k in KGRID]
    star=KGRID[int(np.argmax(pnls))]
    print(f"{reg:>10}{len(sub):>12}  " + "".join(f"{p:>6.0f}" for p in pnls) + f"   best k={star:+.1f}")
