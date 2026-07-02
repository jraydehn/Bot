"""
test_kyes_sweep.py (scratch) — is a LOWER k_yes better?
Dual drift: YES side uses k_yes, NO side uses k_no=0.30 (live). Sweep k_yes.
Report total PnL + YES-side vs NO-side breakdown on OOS archive.
"""
import json, math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; K_NO=0.30
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
sig=a.vol_eff.values*np.sqrt(a.tau_minutes.values); zk=np.log(a.strike.values/a.spot.values)/sig
zfac=norm.ppf(np.clip(a.pup.values,0.01,0.99))*np.sqrt(a.tau_minutes.values/60.0)
pm=a.p_market.values; fee=FEE*np.minimum(pm,1-pm)
py_no=norm.sf(zk - zfac*K_NO); eno=pm-py_no-fee
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def run(k_yes):
    py_yes=norm.sf(zk - zfac*k_yes); eyes=py_yes-pm-fee
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"logged_at":a.logged_at.values,"ticker":a.contract_ticker.values,"close_ts":a.close_ts.values,
        "pm":pm,"ry":a.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; ny=nn=0; py_pnl=no_pnl=0.0; yw=0
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP); u=unit(r.side,r.pm,r.ry); pv=(bf*BANKROLL/cost)*u
            pnl+=pv
            if r.side=="yes": ny+=1; py_pnl+=pv; yw+=u>0
            else: nn+=1; no_pnl+=pv
            traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1
            break
    return ny,nn,py_pnl,no_pnl,pnl,(yw/ny if ny else 0)
print(f"k_no fixed = {K_NO}.  Sweep k_yes:")
print(f"{'k_yes':>7}{'YES_n':>7}{'YES_WR':>8}{'YES_PnL':>9}{'NO_n':>7}{'NO_PnL':>9}{'TOTAL_PnL':>11}")
for ky in [0.0,0.2,0.3,0.5,0.8,1.0,1.4,2.0]:
    ny,nn,yp,np_,tot,ywr=run(ky)
    tag="  <-- LIVE" if abs(ky-1.4)<1e-9 else ""
    print(f"{ky:>7.1f}{ny:>7}{ywr:>8.1%}{yp:>+9.0f}{nn:>7}{np_:>+9.0f}{tot:>+11.0f}{tag}")
