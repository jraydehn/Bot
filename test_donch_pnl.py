"""test_donch_pnl.py (scratch) — net-PnL: block/dampen NO when 1h Donchian position high.
Same OOS pipeline (lookup p_up, k_yes=0.90, k_no=0.30, per-expiry cap 2, Kelly). Two-half."""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07; BANKROLL=2000.0; KELLY_MULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; K_YES=0.90; K_NO=0.30
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
def lookup(t,r,reg):
    key=f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"; tb=TAB.get(reg,TAB["Sideways"]); return tb.get(key,tb["__baseline__"])
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k: break
        out+=k; cur=k[-1][0]+1
        if len(k)<1000: break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]); df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["h","l","c"]: df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-19 06:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e); donch=(h["c"]-h["l"].rolling(20).min())/(h["h"].rolling(20).max()-h["l"].rolling(20).min())
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a["p_market"]>0)&(a["p_market"]<1)&(a["vol_eff"]>0)&(a["dt"]<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
pos=np.clip(np.searchsorted(donch.index.values,a["dt"].values,side="right")-1,0,len(donch)-1)
a["donch1h"]=donch.values[pos]; a=a.dropna(subset=["donch1h"])
lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
idx=np.clip(np.searchsorted(lab.index.values,a["dt"].values,side="right")-1,0,len(lab)-1)
a["regime"]=lab["regime"].values[idx]; a["ry"]=a["resolved_yes"].round().astype(int)
a["pup"]=[lookup(t,r,reg) for t,r,reg in zip(a.composite_trend,a.composite_rev,a.regime)]
sig=a.vol_eff.values*np.sqrt(a.tau_minutes.values); zk=np.log(a.strike.values/a.spot.values)/sig
zf=norm.ppf(np.clip(a.pup.values,0.01,0.99))*np.sqrt(a.tau_minutes.values/60.0); pm=a.p_market.values; fee=FEE*np.minimum(pm,1-pm)
a["eyes"]=norm.sf(zk-zf*K_YES)-pm-fee; a["eno"]=pm-norm.sf(zk-zf*K_NO)-fee
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def bt(df,thr,damp):  # NO bets with donch1h>thr: damp=0 block, else scale
    side=np.where(df.eyes.values>=df.eno.values,"yes","no"); edge=np.where(side=="yes",df.eyes.values,df.eno.values)
    g=pd.DataFrame({"logged_at":df.logged_at.values,"ticker":df.contract_ticker.values,"close_ts":df.close_ts.values,
        "pm":df.p_market.values,"ry":df.ry.values,"side":side,"edge":edge,"dn":df.donch1h.values})
    if damp==0.0: g=g[~((g.side=="no")&(g.dn>thr))]
    g=g[g.edge>MIN_EDGE]; traded=set(); ec={}; pnl=0.0; n=0; an=0; ap=0.0
    for ts,gg in g.sort_values("logged_at").groupby("logged_at"):
        for _,r in gg.sort_values("edge",ascending=False).iterrows():
            if r.ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KELLY_MULT,CAP); aff=(r.side=="no" and r.dn>thr)
            if aff and damp not in (0.0,1.0): bf*=damp
            u=unit(r.side,r.pm,r.ry); pv=(bf*BANKROLL/cost)*u; pnl+=pv; n+=1
            if aff: an+=1; ap+=pv
            traded.add(r.ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1; break
    return n,pnl,an,ap
mid=a["dt"].quantile(0.5); H1=a[a.dt<=mid]; H2=a[a.dt>mid]
print("NO dampener on high 1h-Donchian — NET PnL (charged for blocked winners)")
print(f"{'variant':<26}{'FULL_$':>9}{'H1_$':>8}{'H2_$':>8}{'affNO_n':>9}{'affNO_$':>9}")
for lbl,thr,damp in [("baseline",1.1,1.0),("block donch>0.8",0.8,0.0),("block donch>0.7",0.7,0.0),
                     ("dampen0.5 donch>0.7",0.7,0.5),("dampen0.25 donch>0.7",0.7,0.25),("block donch>0.6",0.6,0.0)]:
    n,pnl,an,ap=bt(a,thr,damp); _,p1,_,_=bt(H1,thr,damp); _,p2,_,_=bt(H2,thr,damp)
    print(f"{lbl:<26}{pnl:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}{an:>9}{ap:>+9.0f}")
