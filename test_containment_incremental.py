"""test_containment_incremental.py — clean 2-state containment HMM + does it add value BEYOND donch?
1) fit 2-state contained/expansion HMM  2) NO-EV by state  3) OVERLAP with donch>0.80
4) incremental NO-EV within the donch-NOT-blocked band  5) net-PnL: donch vs donch+containment."""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KMULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; KDRIFT=0.90; KNO=0.30; TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC")
def cell(t,r): return f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
prodlab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); prodlab.index=pd.to_datetime(prodlab.index,utc=True)
def fetch(s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":"1h","startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    d=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);d["ts"]=pd.to_datetime(d["t"],unit="ms",utc=True)
    for x in ["h","l","c"]: d[x]=d[x].astype(float)
    return d.set_index("ts").sort_index()
h=fetch(int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000),int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000))
c,hi,lo=h.c,h.h,h.l; lr=np.log(c/c.shift(1)); W=48
r6=lr.rolling(6).sum()
feat=pd.DataFrame({"var_ratio":r6.rolling(W).var()/(6*lr.rolling(W).var()),
    "autocorr":lr.rolling(W).apply(lambda x: pd.Series(x).autocorr(lag=1),raw=False),
    "rv_ratio":lr.rolling(6).std()/lr.rolling(24).std(),
    "donch_width":(hi.rolling(20).max()-lo.rolling(20).min())/c}).dropna()
COLS=list(feat.columns)
tr=feat[feat.index<=TRAIN_END]; sc=StandardScaler().fit(tr[COLS])
m=GaussianHMM(n_components=2,covariance_type="full",n_iter=500,random_state=42).fit(sc.transform(tr[COLS]))
lab=pd.Series(m.predict(sc.transform(feat[COLS])),index=feat.index)
# identify expansion state = higher var_ratio
vr=[feat[lab.values==s].var_ratio.mean() for s in range(2)]; EXP=int(np.argmax(vr))
print(f"2-state containment HMM: flip rate {(lab.diff()!=0).mean():.1%} (residence ~{1/(lab.diff()!=0).mean():.0f}h), self-trans {[round(m.transmat_[i,i],2) for i in range(2)]}")
print(f"  EXPANSION=state{EXP} (var_ratio {max(vr):.2f}), CONTAINED=state{1-EXP} (var_ratio {min(vr):.2f})")
donch=(c-lo.rolling(20).min())/(hi.rolling(20).max()-lo.rolling(20).min())
# archive
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)&(a.dt<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]; a["ry"]=a["resolved_yes"].round().astype(int)
li=lab.index.values; a["exp"]=(lab.values[np.clip(np.searchsorted(li,a.dt.values,side="right")-1,0,len(lab)-1)]==EXP)
a["donch"]=donch.values[np.clip(np.searchsorted(donch.index.values,a.dt.values,side="right")-1,0,len(donch)-1)]
a["preg"]=prodlab["regime"].values[np.clip(np.searchsorted(prodlab.index.values,a.dt.values,side="right")-1,0,len(prodlab)-1)]
# OTM-NO franchise
f=a[(a.p_market>=0.10)&(a.p_market<0.40)].copy()
d=f.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("resolved_yes","first"),exp=("exp","first"),dn=("donch","median")).reset_index()
d["ry"]=d.ry.round().astype(int); d["ev"]=d.apply(lambda r:(r.pm-r.ry)-FEE*min(r.pm,1-r.pm),axis=1)
print(f"\n=== NO-EV by containment state (n={len(d)}, baseline {d.ev.mean():+.4f}) ===")
for nm,mk in [("CONTAINED",~d.exp),("EXPANSION",d.exp)]:
    g=d[mk]; print(f"  {nm:<10} n={len(g):>4} NO_WR={(g.ry==0).mean():.0%} NO_EV={g.ev.mean():+.4f}")
print(f"\n=== OVERLAP with donch>0.80 ===")
print(f"  expansion contracts that are ALSO donch>0.80: {(d[d.exp].dn>0.8).mean():.0%}")
print(f"\n=== INCREMENTAL: NO-EV by state, ONLY among donch<=0.80 (not already blocked) ===")
nb=d[d.dn<=0.80]
for nm,mk in [("CONTAINED",~nb.exp),("EXPANSION",nb.exp)]:
    g=nb[mk]; print(f"  {nm:<10} n={len(g):>4} NO_WR={(g.ry==0).mean():.0%} NO_EV={g.ev.mean():+.4f}")
# ---- net-PnL: donch-only vs donch + expansion-NO block ----
def unit(side,pm,ry):
    ff=FEE*min(pm,1-pm); return ((1-pm-ff) if ry==1 else -(pm+ff)) if side=="yes" else ((pm-ff) if ry==0 else -(1-pm+ff))
def pup_prod(t,r,reg): tb=TAB.get(reg,TAB["Sideways"]); return tb.get(cell(t,r),tb["__baseline__"])
def bt(block_exp,lo_=None,hi_=None):
    aa=a if lo_ is None else a[(a.dt>lo_)&(a.dt<=hi_)]
    pup=np.array([pup_prod(t,r,rg) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,aa.preg)])
    sig=aa.vol_eff.values*np.sqrt(aa.tau_minutes.values); zk=np.log(aa.strike.values/aa.spot.values)/sig
    zf=norm.ppf(np.clip(pup,0.01,0.99))*np.sqrt(aa.tau_minutes.values/60.0); pm=aa.p_market.values; fee=FEE*np.minimum(pm,1-pm)
    eyes=norm.sf(zk-zf*KDRIFT)-pm-fee; eno=pm-norm.sf(zk-zf*KNO)-fee
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    g=pd.DataFrame({"la":aa.logged_at.values,"tk":aa.contract_ticker.values,"ct":aa.close_ts.values,"pm":pm,"ry":aa.ry.values,"side":side,"edge":edge,"dn":aa.donch.values,"exp":aa.exp.values})
    g=g[g.edge>MIN_EDGE]; trd=set();ec={};pnl=0.0
    for ts,gg in g.sort_values("la").groupby("la"):
        for _,r in gg.sort_values("edge",ascending=False).iterrows():
            if r.tk in trd or ec.get(r.ct,0)>=EXP_CAP: continue
            if r.side=="no" and r.dn>0.80: continue                 # live donch block
            if block_exp and r.side=="no" and r.exp and r.dn>0.50: continue  # NEW: expansion NO block (mid/high range)
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            bf=min(r.edge/cost*KMULT,CAP)
            if r.side=="no" and r.dn<0.20: bf=min(bf*1.5,0.075)      # live donch boost
            pnl+=(bf*BANKROLL/cost)*unit(r.side,r.pm,r.ry); trd.add(r.tk); ec[r.ct]=ec.get(r.ct,0)+1; break
    return pnl
MID=a.dt.quantile(0.5); LO=a.dt.min()-pd.Timedelta(hours=1); HI=a.dt.max()
print(f"\n=== NET-PnL: live donch gate vs + containment-expansion NO block ===")
print(f"  {'variant':<34}{'FULL':>9}{'H1':>8}{'H2':>8}")
print(f"  {'donch only (current LIVE)':<32}{bt(False):>+9.0f}{bt(False,LO,MID):>+8.0f}{bt(False,MID,HI):>+8.0f}")
print(f"  {'donch + expansion-NO block':<32}{bt(True):>+9.0f}{bt(True,LO,MID):>+8.0f}{bt(True,MID,HI):>+8.0f}")
