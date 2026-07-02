"""test_fast_vs_prod.py (scratch) — FAIR comparison: does the fast regime beat the
ACTUAL production p_up tables (composite_calibration_regime JSONs), not a from-scratch rebuild?
Two-half OOS on the scan archive."""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KMULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; SMOOTH=30; KDRIFT=0.90; KNO=0.30
TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC")
def cell(t,r): return f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"
# ---- ACTUAL production tables ----
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    df["c"]=df["c"].astype(float); return df.set_index("ts").sort_index()
# fast regime
s=int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e); c=h.c; lr=np.log(c/c.shift(1))
feat=pd.DataFrame(index=h.index)
feat["ret2"]=c/c.shift(2)-1; feat["ret6"]=c/c.shift(6)-1; feat["ret12"]=c/c.shift(12)-1; feat["rv6"]=lr.rolling(6).std()
feat=feat.dropna(); FAST=["ret2","ret6","ret12","rv6"]
tr=feat[feat.index<=TRAIN_END]; sc=StandardScaler().fit(tr[FAST])
hmm=GaussianHMM(n_components=3,covariance_type="full",n_iter=80,random_state=0).fit(sc.transform(tr[FAST]))
fast_lab=pd.Series(hmm.predict(sc.transform(feat[FAST])),index=feat.index)
# fast p_up tables from bar frame (same SMOOTH method)
fr=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna(); fr=fr[fr.index<=TRAIN_END]
li=fast_lab.index.values
frp=np.clip(np.searchsorted(li,fr.index.values,side="right")-1,0,len(fast_lab)-1)
f2=fr.assign(reg=fast_lab.values[frp]); f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
base=fr.next_up.mean(); rb=f2.groupby("reg").next_up.mean().to_dict()
grp=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
FASTTAB={(rg,cl):(row["size"]*row["mean"]+SMOOTH*rb.get(rg,base))/(row["size"]+SMOOTH) for (rg,cl),row in grp.iterrows()}
# archive + prod regime label
prod=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); prod.index=pd.to_datetime(prod.index,utc=True)
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)&(a.dt<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]; a["ry"]=a["resolved_yes"].round().astype(int)
pl=prod.index.values; ap=np.clip(np.searchsorted(pl,a.dt.values,side="right")-1,0,len(prod)-1); a["preg"]=prod["regime"].values[ap]
fp=np.clip(np.searchsorted(li,a.dt.values,side="right")-1,0,len(fast_lab)-1); a["freg"]=fast_lab.values[fp]
def pup_prod(t,r,reg):
    tb=TAB.get(reg,TAB["Sideways"]); return tb.get(cell(t,r),tb["__baseline__"])
def pup_fast(t,r,reg): return FASTTAB.get((reg,cell(t,r)),rb.get(reg,base))
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def run(which,dt_lo=None,dt_hi=None):
    aa=a if dt_lo is None else a[(a.dt>dt_lo)&(a.dt<=dt_hi)]
    if which=="prod": pup=np.array([pup_prod(t,r,rg) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,aa.preg)])
    else: pup=np.array([pup_fast(t,r,rg) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,aa.freg)])
    sig=aa.vol_eff.values*np.sqrt(aa.tau_minutes.values); zk=np.log(aa.strike.values/aa.spot.values)/sig
    zf=norm.ppf(np.clip(pup,0.01,0.99))*np.sqrt(aa.tau_minutes.values/60.0); pm=aa.p_market.values; fee=FEE*np.minimum(pm,1-pm)
    eyes=norm.sf(zk-zf*KDRIFT)-pm-fee; eno=pm-norm.sf(zk-zf*KNO)-fee
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"la":aa.logged_at.values,"tk":aa.contract_ticker.values,"ct":aa.close_ts.values,"pm":pm,"ry":aa.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    tr_=set(); ec={}; pnl=0.0
    for ts,g in d.sort_values("la").groupby("la"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.tk in tr_ or ec.get(r.ct,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            pnl+=(min(r.edge/cost*KMULT,CAP)*BANKROLL/cost)*unit(r.side,r.pm,r.ry); tr_.add(r.tk); ec[r.ct]=ec.get(r.ct,0)+1; break
    return pnl
mid=a.dt.quantile(0.5); lo=a.dt.min()-pd.Timedelta(hours=1); hi=a.dt.max()
print("FAIR comparison — actual PRODUCTION tables vs from-scratch FAST regime (two-half OOS):")
print(f"{'model':<26}{'FULL':>9}{'H1':>9}{'H2':>9}")
for nm,w in [("PROD tables (current LIVE)","prod"),("FAST regime (from-scratch)","fast")]:
    print(f"  {nm:<24}{run(w):>+9.0f}{run(w,lo,mid):>+9.0f}{run(w,mid,hi):>+9.0f}")
