"""test_joint_regime_pup.py — JOINT (slow_macro × fast_intraday) regime p_up calibration.
Mirrors build_fast_regime.py / production pipeline EXACTLY; only change = condition p_up tables
on the JOINT regime label instead of slow-only. Reports:
  1) sparsity diagnosis (how much of the archive hits a real (trend,rev) cell vs pooled fallback)
  2) OOS scan-archive PnL: production(slow) vs fast-only vs JOINT, FULL + two halves
  3) MCPT: permute the FAST overlay (keep slow fixed) — does the REAL joint beat shuffled joints?
     (controls for the structure artifact: if a random fast split conditions just as well, it's noise)
"""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; BANKROLL=2000.0; KMULT=0.10; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; KDRIFT=0.90; KNO=0.30
SMOOTH_K=30; MIN_N=10; TREND_CLIP=5; REV_CLIP=11; SEED=42
TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC"); TABLE_START=pd.Timestamp("2025-01-01",tz="UTC")
FAST_FEATS=["ret2","ret6","ret12","rv6"]; FAST_N=3   # canonical intraday config (no sweep = no cherry-pick)
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

print("fetching 1h candles + feature library...")
h=fetch(int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000),int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000))
c=h.c; lr=np.log(c/c.shift(1))
def ret(n): return c/c.shift(n)-1
def rv(n): return lr.rolling(n).std()
LIB={"ret2":ret(2),"ret6":ret(6),"ret12":ret(12),"rv6":rv(6)}

fr=pd.read_parquet("reform_results/_pup_cv_frame.parquet").dropna()
fr=fr[(fr.index>=TABLE_START)&(fr.index<=TRAIN_END)]
base_all=fr.next_up.mean()
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
prod=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); prod.index=pd.to_datetime(prod.index,utc=True)

a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)&(a.dt<=pd.Timestamp("2026-06-16 21:00",tz="UTC"))]; a["ry"]=a["resolved_yes"].round().astype(int)
pl=prod.index.values; a["preg"]=prod["regime"].values[np.clip(np.searchsorted(pl,a.dt.values,side="right")-1,0,len(prod)-1)]
MID=a.dt.quantile(0.5); LO=a.dt.min()-pd.Timedelta(hours=1); HI=a.dt.max()

def fit_fast(feats,n):
    F=pd.concat([LIB[f] for f in feats],axis=1); F.columns=feats; F=F.dropna()
    tr=F[F.index<=TRAIN_END]; sc=StandardScaler().fit(tr.values)
    m=GaussianHMM(n_components=n,covariance_type="full",n_iter=500,random_state=SEED).fit(sc.transform(tr.values))
    return pd.Series(m.predict(sc.transform(F.values)),index=F.index)

def build_tables(joint_lab):
    """joint_lab: Series of joint-regime strings. Hierarchical fallback: cell -> joint base -> slow base -> global."""
    li=joint_lab.index.values
    jr=joint_lab.values[np.clip(np.searchsorted(li,fr.index.values,side="right")-1,0,len(joint_lab)-1)]
    f2=fr.assign(reg=jr); f2["slow"]=[s.split("|")[0] for s in jr]
    f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
    jbase=f2.groupby("reg").next_up.mean().to_dict()
    sbase=f2.groupby("slow").next_up.mean().to_dict()
    g=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
    tab={}; real=0; fb=0
    for (rg,cl),row in g.iterrows():
        if row["size"]>=MIN_N:
            tab[(rg,cl)]=(row["size"]*row["mean"]+SMOOTH_K*jbase.get(rg,base_all))/(row["size"]+SMOOTH_K); real+=1
        else: fb+=1
    return tab,jbase,sbase,real,fb

def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))

def pnl_from_pup(pup,aa):
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

def slice_(lo,hi): return a if lo is None else a[(a.dt>lo)&(a.dt<=hi)]
def prod_pup(aa): return np.array([TAB.get(rg,TAB["Sideways"]).get(cell(t,r),TAB.get(rg,TAB["Sideways"])["__baseline__"]) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,aa.preg)])

# ---- build joint regime ----
fast=fit_fast(FAST_FEATS,FAST_N); fi=fast.index.values
def joint_label_for(dts, fast_lab):
    fl=fast_lab.values[np.clip(np.searchsorted(fast_lab.index.values,dts,side="right")-1,0,len(fast_lab)-1)]
    sl=prod["regime"].values[np.clip(np.searchsorted(pl,dts,side="right")-1,0,len(prod)-1)]
    return np.array([f"{s}|F{int(f)}" for s,f in zip(sl,fl)])
# joint label series aligned to fast index (for table build): need slow at fast timestamps
jl_vals=joint_label_for(fast.index.values,fast)
joint_lab=pd.Series(jl_vals,index=fast.index)

tab,jbase,sbase,real,fb=build_tables(joint_lab)
def joint_pup_fn(fast_lab,tab,jbase):
    def f(aa):
        jr=joint_label_for(aa.dt.values,fast_lab)
        return np.array([tab.get((rg,cell(t,r)),jbase.get(rg,base_all)) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,jr)])
    return f

# sparsity
arch_jr=joint_label_for(a.dt.values,fast)
hit=np.array([(rg,cell(t,r)) in tab for t,r,rg in zip(a.composite_trend,a.composite_rev,arch_jr)])
print(f"\nfast: {FAST_FEATS} {FAST_N}st, flip {(fast.diff()!=0).mean():.0%}")
print(f"joint regimes present (calib frame): {joint_lab.nunique()} combos  |  archive: {pd.Series(arch_jr).nunique()} combos")
print(f"SPARSITY: real (trend,rev) cells built={real}, pooled-fallback cells={fb}")
print(f"  archive rows hitting a REAL cell: {hit.mean():.0%}  (rest fall back to joint-regime baseline)")
print(f"  combo frequency in archive:");
for k,v in pd.Series(arch_jr).value_counts().items(): print(f"    {k:<14} n={v}")

print(f"\n{'config':<40}{'FULL':>9}{'H1':>8}{'H2':>8}")
for nm,fn in [("PRODUCTION (slow, LIVE)",prod_pup),("JOINT (slow×fast)",joint_pup_fn(fast,tab,jbase))]:
    full=pnl_from_pup(fn(a),a); h1=pnl_from_pup(fn(slice_(LO,MID)),slice_(LO,MID)); h2=pnl_from_pup(fn(slice_(MID,HI)),slice_(MID,HI))
    print(f"  {nm:<38}{full:>+9.0f}{h1:>+8.0f}{h2:>+8.0f}")
prod_full=pnl_from_pup(prod_pup(a),a); joint_full=pnl_from_pup(joint_pup_fn(fast,tab,jbase)(a),a)

# ---- MCPT: permute the fast overlay (keep slow fixed), rebuild joint tables, measure PnL ----
print(f"\nMCPT (200 perms): shuffle fast labels (slow fixed), rebuild joint, measure FULL PnL")
print(f"  H0: a RANDOM fast overlay conditions p_up just as well (structure artifact)")
rng=np.random.default_rng(SEED); perm_pnls=[]
fast_arr=fast.values
for i in range(200):
    pf=rng.permutation(fast_arr)
    plab=pd.Series(pf,index=fast.index)
    jlv=np.array([f"{s}|F{int(x)}" for s,x in zip([v.split('|')[0] for v in jl_vals],pf)])
    jlab=pd.Series(jlv,index=fast.index)
    pt,pjb,_,_,_=build_tables(jlab)
    pnl=pnl_from_pup(joint_pup_fn(plab,pt,pjb)(a),a); perm_pnls.append(pnl)
perm=np.array(perm_pnls)
pval=(np.sum(perm>=joint_full)+1)/(len(perm)+1)
print(f"  REAL joint FULL PnL={joint_full:+.0f}  vs  production={prod_full:+.0f}  (delta {joint_full-prod_full:+.0f})")
print(f"  permuted-fast PnL: mean={perm.mean():+.0f} p50={np.percentile(perm,50):+.0f} p90={np.percentile(perm,90):+.0f} max={perm.max():+.0f}")
print(f"  MCPT p (real joint beats shuffled fast overlay) = {pval:.3f}")
print(f"  => {'REAL signal: fast dimension adds orthogonal value' if pval<0.05 else 'STRUCTURE ARTIFACT: random fast split conditions as well (reject)'}")
