"""
test_vwap_damp.py (scratch) — net-PnL study: dampen/block NO when vwap_stretch>=1.
Same OOS pipeline as k_yes test (lookup p_up, k_yes=0.90, k_no=0.30, per-expiry cap 2, Kelly).
Charges the rule for winners it blocks. Two-half stability.
"""
import json, math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; K_YES=0.90; K_NO=0.30
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
def lookup(t,r,reg):
    key=f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"; tb=TAB.get(reg,TAB["Sideways"]); return tb.get(key,tb["__baseline__"])
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","vwap_stretch_score"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)&(a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
a["vw"]=a["vwap_stretch_score"].fillna(0)
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
idx=np.clip(np.searchsorted(lab.index.values,a["dt"].values,side="right")-1,0,len(lab)-1)
a["regime"]=lab["regime"].values[idx]; a["ry"]=a["resolved_yes"].round().astype(int)
a["pup"]=[lookup(t,r,reg) for t,r,reg in zip(a.composite_trend,a.composite_rev,a.regime)]
sig=a.vol_eff.values*np.sqrt(a.tau_minutes.values); zk=np.log(a.strike.values/a.spot.values)/sig
zf=norm.ppf(np.clip(a.pup.values,0.01,0.99))*np.sqrt(a.tau_minutes.values/60.0)
pm=a.p_market.values; fee=FEE*np.minimum(pm,1-pm)
a["eyes"]=norm.sf(zk-zf*K_YES)-pm-fee; a["eno"]=pm-norm.sf(zk-zf*K_NO)-fee
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def bt(df,damp):  # damp = factor for NO bets with vw>=1 (0=block, 1=baseline)
    side=np.where(df.eyes.values>=df.eno.values,"yes","no"); edge=np.where(side=="yes",df.eyes.values,df.eno.values)
    d=pd.DataFrame({"logged_at":df.logged_at.values,"ticker":df.contract_ticker.values,"close_ts":df.close_ts.values,
        "pm":df.p_market.values,"ry":df.ry.values,"side":side,"edge":edge,"vw":df.vw.values})
    # block: drop NO candidates with vw>=1 before selection
    if damp==0.0:
        d=d[~((d.side=="no")&(d.vw>=1))]
    d=d[d.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; n=0; aff_pnl=0.0; aff_n=0
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP)
            aff = (r.side=="no" and r.vw>=1)
            if aff and damp not in (0.0,1.0): bf*=damp
            u=unit(r.side,r.pm,r.ry); pv=(bf*BANKROLL/cost)*u
            pnl+=pv; n+=1
            if aff: aff_pnl+=pv; aff_n+=1
            traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1; break
    return n,pnl,aff_n,aff_pnl
mid=a["dt"].quantile(0.5); H1=a[a.dt<=mid]; H2=a[a.dt>mid]
print("vwap_stretch>=1 NO dampener — NET PnL (charged for blocked winners)")
print(f"{'variant':<22}{'FULL_n':>7}{'FULL_$':>9}{'H1_$':>8}{'H2_$':>8}{'vw>=1_NO_n':>11}{'vw>=1_NO_$':>11}")
for lbl,damp in [("baseline (1.0)",1.0),("dampen 0.5",0.5),("dampen 0.25",0.25),("block (0.0)",0.0)]:
    n,pnl,an,ap=bt(a,damp); _,p1,_,_=bt(H1,damp); _,p2,_,_=bt(H2,damp)
    print(f"{lbl:<22}{n:>7}{pnl:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}{an:>11}{ap:>+11.0f}")
