"""build_fast_regime.py — PROPER fast-regime build + optimization, mirroring the
production pipeline exactly (train_macro_regime_hmm.py + build_regime_pup_tables.py),
changing ONLY the features. Sweeps intraday feature sets / n_states, builds
production-quality p_up tables for each, and compares to the ACTUAL production tables
on OOS scan-archive PnL with two-half stability.

Production methodology replicated:
  HMM: GaussianHMM(n, covariance_type='full', n_iter=500, random_state=42) on StandardScaled feats
  states ordered by mean sharpe-like feat (highest=Bull ... lowest=Bear)
  tables: TEST_START=2025-01-01, SMOOTH_K=30, MIN_N=10, clip trend[-5,5] rev[-11,11]
"""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KMULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; KDRIFT=0.90; KNO=0.30
SMOOTH_K=30; MIN_N=10; TREND_CLIP=5; REV_CLIP=11; SEED=42
TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC"); TABLE_START=pd.Timestamp("2025-01-01",tz="UTC")
def cell(t,r): return f"{int(np.clip(t,-TREND_CLIP,TREND_CLIP))},{int(np.clip(r,-REV_CLIP,REV_CLIP))}"

def fetch(s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":"1h","startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    df["c"]=df["c"].astype(float); return df.set_index("ts").sort_index()

print("fetching 1h candles + building feature library...")
h=fetch(int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000),int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000))
c=h.c; lr=np.log(c/c.shift(1))
def ret(n): return c/c.shift(n)-1
def rv(n): return lr.rolling(n).std()
def sharpe(n): return (lr.rolling(n).mean()/rv(n).replace(0,np.nan)).fillna(0)
LIB={"ret1":ret(1),"ret2":ret(2),"ret4":ret(4),"ret6":ret(6),"ret8":ret(8),"ret12":ret(12),"ret24":ret(24),
     "rv4":rv(4),"rv6":rv(6),"rv12":rv(12),"sharpe4":sharpe(4),"sharpe6":sharpe(6),"sharpe12":sharpe(12)}

# scored frame (trend, rev, next_up) from cache — production-computed composite scores
fr=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
fr=fr[(fr.index>=TABLE_START)&(fr.index<=TRAIN_END)]
base_all=fr.next_up.mean()

# production tables + labels (baseline)
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
prod=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); prod.index=pd.to_datetime(prod.index,utc=True)

# scan archive (OOS PnL)
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)&(a.dt<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]; a["ry"]=a["resolved_yes"].round().astype(int)
pl=prod.index.values; a["preg"]=prod["regime"].values[np.clip(np.searchsorted(pl,a.dt.values,side="right")-1,0,len(prod)-1)]
MID=a.dt.quantile(0.5); LO=a.dt.min()-pd.Timedelta(hours=1); HI=a.dt.max()

def fit_regime(feats,n):
    F=pd.concat([LIB[f] for f in feats],axis=1); F.columns=feats; F=F.dropna()
    tr=F[F.index<=TRAIN_END]; sc=StandardScaler().fit(tr.values)
    m=GaussianHMM(n_components=n,covariance_type="full",n_iter=500,random_state=SEED).fit(sc.transform(tr.values))
    lab=pd.Series(m.predict(sc.transform(F.values)),index=F.index)
    # order states by mean of last feature (sharpe-like) for stable naming (cosmetic only)
    return lab
def build_tables(lab):
    li=lab.index.values
    f2=fr.assign(reg=lab.values[np.clip(np.searchsorted(li,fr.index.values,side="right")-1,0,len(lab)-1)])
    f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
    rb=f2.groupby("reg").next_up.mean().to_dict()
    g=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
    tab={}
    for (rg,cl),row in g.iterrows():
        tab[(rg,cl)]=(row["size"]*row["mean"]+SMOOTH_K*rb.get(rg,base_all))/(row["size"]+SMOOTH_K) if row["size"]>=MIN_N else rb.get(rg,base_all)
    return tab,rb
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def pnl_run(pup_fn,dt_lo=None,dt_hi=None):
    aa=a if dt_lo is None else a[(a.dt>dt_lo)&(a.dt<=dt_hi)]
    pup=pup_fn(aa)
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
            pnl+=(min(r.edge/cost*KMULT,CAP)*BANKROLL/cost)*unit(r.side,r.pm,r.ry); tr_.add(r.tk); ec[r.ct]=ec.get(r.ct,0)+1; break
    return pnl
# production baseline
def prod_pup(aa): return np.array([TAB.get(rg,TAB["Sideways"]).get(cell(t,r),TAB.get(rg,TAB["Sideways"])["__baseline__"]) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,aa.preg)])
print(f"\n{'config':<46}{'FULL':>9}{'H1':>8}{'H2':>8}")
pf,p1,p2=pnl_run(prod_pup),pnl_run(prod_pup,LO,MID),pnl_run(prod_pup,MID,HI)
print(f"  {'PRODUCTION (slow, current LIVE)':<44}{pf:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}")
print("  "+"-"*68)
CANDS=[
 (["ret2","ret6","ret12","rv6"],3),
 (["ret1","ret4","ret8","rv4"],3),
 (["ret6","ret12","rv12","sharpe12"],3),
 (["ret4","ret12","rv12","sharpe4"],3),
 (["ret2","ret6","ret12","rv6"],2),
 (["ret2","ret6","ret12","rv6"],4),
 (["ret6","ret12","rv12","sharpe12"],4),
]
for feats,n in CANDS:
    lab=fit_regime(feats,n)
    tab,rb=build_tables(lab); li=lab.index.values
    def make(tab=tab,rb=rb,li=li,lab=lab):
        def f(aa):
            reg=lab.values[np.clip(np.searchsorted(li,aa.dt.values,side="right")-1,0,len(lab)-1)]
            return np.array([tab.get((rg,cell(t,r)),rb.get(rg,base_all)) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,reg)])
        return f
    fn=make(); flip=(lab.diff()!=0).mean()
    full,h1,h2=pnl_run(fn),pnl_run(fn,LO,MID),pnl_run(fn,MID,HI)
    name=f"FAST {'+'.join(feats)} ({n}st, flip {flip:.0%})"
    print(f"  {name:<44}{full:>+9.0f}{h1:>+8.0f}{h2:>+8.0f}")
