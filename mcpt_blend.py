"""mcpt_blend.py — MCPT the slow+fast BLENDS (50/50, 60/40, 40/60).
Null: keep production p_up fixed, circular-shift the FAST component, re-blend, measure OOS PnL.
Tests whether the fast component's ALIGNMENT adds real signal to the blend."""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
exec(open("build_fast_regime.py").read().split('print(f"\\n{')[0])
rng=np.random.default_rng(7)
feats=["ret2","ret6","ret12","rv6"]; lab=fit_regime(feats,3); lvals=lab.values; Ln=len(lab)
li=lab.index.values

# precompute production p_up aligned to full archive `a`
PROD=prod_pup(a)

def fast_pup_array(label_values):
    ls=pd.Series(label_values,index=lab.index); lix=ls.index.values
    f2=fr.assign(reg=ls.values[np.clip(np.searchsorted(lix,fr.index.values,side="right")-1,0,len(ls)-1)])
    f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
    rb=f2.groupby("reg").next_up.mean().to_dict()
    g=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
    tab={(rg,cl):((row["size"]*row["mean"]+SMOOTH_K*rb.get(rg,base_all))/(row["size"]+SMOOTH_K) if row["size"]>=MIN_N else rb.get(rg,base_all)) for (rg,cl),row in g.iterrows()}
    reg=ls.values[np.clip(np.searchsorted(lix,a.dt.values,side="right")-1,0,len(ls)-1)]
    return np.array([tab.get((rg,cell(t,r)),rb.get(rg,base_all)) for t,r,rg in zip(a.composite_trend,a.composite_rev,reg)])

def pnl_array(pup):  # pup aligned to full a; FULL OOS PnL
    aa=a
    sig=aa.vol_eff.values*np.sqrt(aa.tau_minutes.values); zk=np.log(aa.strike.values/aa.spot.values)/sig
    zf=norm.ppf(np.clip(pup,0.01,0.99))*np.sqrt(aa.tau_minutes.values/60.0); pm=aa.p_market.values; fee=FEE*np.minimum(pm,1-pm)
    eyes=norm.sf(zk-zf*KDRIFT)-pm-fee; eno=pm-norm.sf(zk-zf*KNO)-fee
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"la":aa.logged_at.values,"tk":aa.contract_ticker.values,"ct":aa.close_ts.values,"pm":pm,"ry":aa.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    tr_=set();ec={};pnl=0.0
    for ts,gg in d.sort_values("la").groupby("la"):
        for _,r in gg.sort_values("edge",ascending=False).iterrows():
            if r.tk in tr_ or ec.get(r.ct,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            pnl+=(min(r.edge/cost*KMULT,CAP)*BANKROLL/cost)*(((1-r.pm-FEE*min(r.pm,1-r.pm)) if r.ry==1 else -(r.pm+FEE*min(r.pm,1-r.pm))) if r.side=="yes" else ((r.pm-FEE*min(r.pm,1-r.pm)) if r.ry==0 else -(1-r.pm+FEE*min(r.pm,1-r.pm))))
            tr_.add(r.tk); ec[r.ct]=ec.get(r.ct,0)+1; break
    return pnl

FAST=fast_pup_array(lvals)
WS={"50/50":0.5,"60/40":0.6,"40/60":0.4}
real={k:pnl_array(w*PROD+(1-w)*FAST) for k,w in WS.items()}
prod_only=pnl_array(PROD)
print(f"production (w=1.0): {prod_only:+.0f}")
for k in WS: print(f"REAL blend {k} (slow/fast): {real[k]:+.0f}")
NPERM=200
print(f"\nrunning {NPERM} shifts (production fixed, fast component circularly shifted)...")
null={k:[] for k in WS}
for i in range(NPERM):
    sf=fast_pup_array(np.roll(lvals,int(rng.integers(200,Ln-200))))
    for k,w in WS.items(): null[k].append(pnl_array(w*PROD+(1-w)*sf))
    if (i+1)%50==0: print(f"  ...{i+1}/{NPERM}")
print(f"\n=== BLEND MCPT (vs shifted-fast null; production fixed) ===")
print(f"  production baseline: {prod_only:+.0f}")
for k in WS:
    arr=np.array(null[k]); p=(np.sum(arr>=real[k])+1)/(NPERM+1)
    pbeat_prod=(np.sum(arr>=prod_only)+1)/(NPERM+1)
    print(f"  {k}: real={real[k]:+.0f}  null_mean={arr.mean():+.0f}(±{arr.std():.0f})  null_95%={np.percentile(arr,95):+.0f}  p(shift>=real)={p:.3f}  | P(shift beats production)={pbeat_prod:.3f}")
print(f"\n  Interp: p<0.05 = the fast component's ALIGNMENT adds real signal to the blend.")
print(f"          If real blend ~ null mean and null mean ~ production, the blend is NOT a real improvement.")
