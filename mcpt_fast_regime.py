"""mcpt_fast_regime.py — MCPT: is the fast regime's PnL edge real signal or regime-structure artifact?
Circular-shift the fast labels (preserves stickiness/run-lengths/state freqs, breaks market alignment),
rebuild production-quality p_up tables, measure OOS PnL. p = P(shifted >= real)."""
import json, math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
# reuse setup + helpers (fetch, LIB, fr, TAB, prod, a, pnl_run, build_tables, fit_regime, prod_pup, cell, base_all, LO/MID/HI)
exec(open("build_fast_regime.py").read().split('print(f"\\n{')[0])

rng=np.random.default_rng(7)
feats=["ret2","ret6","ret12","rv6"]
lab=fit_regime(feats,3)
lvals=lab.values; lidx=lab.index.values; N=len(lab)

def pnl_for_labels(label_values):
    ls=pd.Series(label_values,index=lab.index)
    li=ls.index.values
    f2=fr.assign(reg=ls.values[np.clip(np.searchsorted(li,fr.index.values,side="right")-1,0,len(ls)-1)])
    f2["cl"]=[cell(t,r) for t,r in zip(f2.trend,f2.rev)]
    rb=f2.groupby("reg").next_up.mean().to_dict()
    g=f2.groupby(["reg","cl"]).next_up.agg(["mean","size"])
    tab={(rg,cl):((row["size"]*row["mean"]+SMOOTH_K*rb.get(rg,base_all))/(row["size"]+SMOOTH_K) if row["size"]>=MIN_N else rb.get(rg,base_all)) for (rg,cl),row in g.iterrows()}
    def fn(aa):
        reg=ls.values[np.clip(np.searchsorted(li,aa.dt.values,side="right")-1,0,len(ls)-1)]
        return np.array([tab.get((rg,cell(t,r)),rb.get(rg,base_all)) for t,r,rg in zip(aa.composite_trend,aa.composite_rev,reg)])
    return pnl_run(fn)

real=pnl_for_labels(lvals)
realh1=None
print(f"REAL fast regime OOS PnL (FULL): {real:+.0f}")
print(f"production baseline: {pnl_run(prod_pup):+.0f}   pooled baseline: ~+11323")
NPERM=200
print(f"\nrunning {NPERM} circular-shift permutations (preserves regime structure, breaks market alignment)...")
perms=[]
for i in range(NPERM):
    k=int(rng.integers(200,N-200))
    perms.append(pnl_for_labels(np.roll(lvals,k)))
    if (i+1)%50==0: print(f"  ...{i+1}/{NPERM}")
perms=np.array(perms)
p=(np.sum(perms>=real)+1)/(NPERM+1)
print(f"\n=== MCPT RESULT ===")
print(f"  real fast PnL:        {real:+.0f}")
print(f"  shifted-null mean:    {perms.mean():+.0f}   (std {perms.std():.0f})")
print(f"  shifted-null pctiles: 5%={np.percentile(perms,5):+.0f}  50%={np.percentile(perms,50):+.0f}  95%={np.percentile(perms,95):+.0f}  max={perms.max():+.0f}")
print(f"  p-value P(shifted>=real) = {p:.4f}")
print(f"  => {'SIGNIFICANT — fast features carry real signal' if p<0.05 else 'NOT significant — edge is regime-structure artifact, not the fast features'}")
