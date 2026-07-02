"""test_fast_regime_pup.py (scratch) — does a FASTER regime improve p_up conditioning?
Build 3-state HMM on SLOW features (ret24/72h, rv24, sharpe) vs FAST features
(ret2/6/12h, rv6h). Condition p_up on each, OOS net-PnL on scan archive 05-18..06-16.
Train tables on bars <= 05-16; test on archive. Compare: pooled / slow / fast."""
import math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KMULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; SMOOTH=30; KDRIFT=0.90
TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC")
def cell(t,r): return (int(np.clip(t,-5,5)), int(np.clip(r,-11,11)))
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["c"]:df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
# 1h candles 2024->06-16, build slow + fast features
s=int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e); c=h.c; lr=np.log(c/c.shift(1))
feat=pd.DataFrame(index=h.index)
feat["ret24"]=c/c.shift(24)-1; feat["ret72"]=c/c.shift(72)-1
feat["rv24"]=lr.rolling(24).std(); feat["sharpe24"]=feat["ret24"]/(feat["rv24"]+1e-9)
feat["ret2"]=c/c.shift(2)-1; feat["ret6"]=c/c.shift(6)-1; feat["ret12"]=c/c.shift(12)-1
feat["rv6"]=lr.rolling(6).std()
feat=feat.dropna()
SLOW=["ret24","ret72","rv24","sharpe24"]; FAST=["ret2","ret6","ret12","rv6"]
def regime_labels(cols):
    tr=feat[feat.index<=TRAIN_END]
    sc=StandardScaler().fit(tr[cols])
    hmm=GaussianHMM(n_components=3,covariance_type="full",n_iter=80,random_state=0).fit(sc.transform(tr[cols]))
    lab=pd.Series(hmm.predict(sc.transform(feat[cols])),index=feat.index)
    # flip-rate diagnostic: how often does the regime change hour-to-hour?
    flip=(lab.diff()!=0).mean()
    return lab,flip
# SLOW = the ACTUAL production sticky regime (Bull/Bear/Sideways), not a from-scratch refit
prod=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); prod.index=pd.to_datetime(prod.index,utc=True)
slow_lab=prod["regime"].astype("category").cat.codes; slow_lab.index=prod.index
slow_lab=slow_lab[~slow_lab.index.duplicated()].sort_index()
slow_flip=(slow_lab.diff()!=0).mean()
fast_lab,fast_flip=regime_labels(FAST)
print(f"regime FLIP RATE (hour-to-hour change): slow/prod={slow_flip:.1%} (residence~{1/max(slow_flip,1e-9):.0f}h)  fast={fast_flip:.1%} (~{1/max(fast_flip,1e-9):.1f}h)")
# bar frame for tables (trend,rev,next_up)
fr=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
fr=fr[fr.index<=TRAIN_END]
# archive for OOS PnL test
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)&(a.dt<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]
a["ry"]=a["resolved_yes"].round().astype(int)
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def run(lab,dt_lo=None,dt_hi=None):
    li=lab.index.values
    base=fr.next_up.mean()
    frp=np.clip(np.searchsorted(li,fr.index.values,side="right")-1,0,len(lab)-1)
    f2=fr.assign(reg=lab.values[frp]); f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
    rb=f2.groupby("reg").next_up.mean().to_dict()
    grp=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
    tab={(rg,cl):(row["size"]*row["mean"]+SMOOTH*rb.get(rg,base))/(row["size"]+SMOOTH) for (rg,cl),row in grp.iterrows()}
    aa=a
    if dt_lo is not None: aa=aa[(aa.dt>dt_lo)&(aa.dt<=dt_hi)]
    ap=np.clip(np.searchsorted(li,aa.dt.values,side="right")-1,0,len(lab)-1); areg=lab.values[ap]
    pup=np.array([tab.get((rg,cell(t,r)),rb.get(rg,base)) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,areg)])
    sig=aa.vol_eff.values*np.sqrt(aa.tau_minutes.values); zk=np.log(aa.strike.values/aa.spot.values)/sig
    zf=norm.ppf(np.clip(pup,0.01,0.99))*np.sqrt(aa.tau_minutes.values/60.0); pm=aa.p_market.values; fee=FEE*np.minimum(pm,1-pm)
    eyes=norm.sf(zk-zf*KDRIFT)-pm-fee; eno=pm-norm.sf(zk-zf*0.30)-fee
    side=np.where(eyes>=eno,"yes","no"); edge=np.where(side=="yes",eyes,eno)
    d=pd.DataFrame({"la":aa.logged_at.values,"tk":aa.contract_ticker.values,"ct":aa.close_ts.values,"pm":pm,"ry":aa.ry.values,"side":side,"edge":edge}); d=d[d.edge>MIN_EDGE]
    tr=set(); ec={}; pnl=0.0
    for ts,g in d.sort_values("la").groupby("la"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.tk in tr or ec.get(r.ct,0)>=EXP_CAP: continue
            cost=r.pm if r.side=="yes" else 1-r.pm
            if cost<=0: continue
            pnl+=(min(r.edge/cost*KMULT,CAP)*BANKROLL/cost)*unit(r.side,r.pm,r.ry); tr.add(r.tk); ec[r.ct]=ec.get(r.ct,0)+1; break
    return pnl
# pooled = single regime
pool=pd.Series(0,index=feat.index)
mid=a.dt.quantile(0.5); lo=a.dt.min()-pd.Timedelta(hours=1); hi=a.dt.max()
print(f"\nOOS scan archive PnL (train tables<=05-16):  FULL | H1 | H2  (H1/H2 split {mid})")
for nm,lab in [("pooled",pool),("SLOW (prod, current)",slow_lab),("FAST (intraday)",fast_lab)]:
    full=run(lab); h1=run(lab,lo,mid); h2=run(lab,mid,hi)
    print(f"  {nm:<22} ${full:>+8.0f} | ${h1:>+7.0f} | ${h2:>+7.0f}")
