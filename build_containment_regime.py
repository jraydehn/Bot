"""build_containment_regime.py — define a CONTAINMENT/EXPANSION HMM on historical BTC 1h data.
Features capture range DYNAMICS (trending vs mean-reverting, vol/range expansion), NOT direction.
Inspect: state characteristics + forward move size + (the money metric) OTM-NO EV per state from archive.
Production HMM methodology: GaussianHMM(full cov, n_iter=500, seed 42), StandardScaler."""
import requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
FEE=0.07; TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC")
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
c,hi,lo=h.c,h.h,h.l; lr=np.log(c/c.shift(1))
W=48
# CONTAINMENT/EXPANSION features (range dynamics, direction-agnostic):
r6=lr.rolling(6).sum()
var_ratio=r6.rolling(W).var()/(6*lr.rolling(W).var())          # >1 trending/expansive, <1 mean-reverting/contained
autocorr=lr.rolling(W).apply(lambda x: pd.Series(x).autocorr(lag=1),raw=False)  # +trending, -mean-reverting
rv_ratio=lr.rolling(6).std()/lr.rolling(24).std()              # >1 vol expanding
donch_width=(hi.rolling(20).max()-lo.rolling(20).min())/c      # range tightness (wide=expansion)
feat=pd.DataFrame({"var_ratio":var_ratio,"autocorr":autocorr,"rv_ratio":rv_ratio,"donch_width":donch_width}).dropna()
COLS=list(feat.columns)
print("CONTAINMENT/EXPANSION features:", COLS)
print("  var_ratio  : variance ratio (>1 trending/expansive, <1 mean-reverting/contained)")
print("  autocorr   : lag-1 return autocorr (+ trending, - mean-reverting)")
print("  rv_ratio   : 6h vol / 24h vol (>1 vol expanding)")
print("  donch_width: 20h range width / price (wide = expansion)")
tr=feat[feat.index<=TRAIN_END]; sc=StandardScaler().fit(tr[COLS])
m=GaussianHMM(n_components=3,covariance_type="full",n_iter=500,random_state=42).fit(sc.transform(tr[COLS]))
lab=pd.Series(m.predict(sc.transform(feat[COLS])),index=feat.index)
# forward move size (what NO cares about): next-1h abs return + next-3h max excursion
fwd1=np.abs(lr.shift(-1)).reindex(feat.index)
fwd3=(np.log(hi.rolling(3).max().shift(-3)/c)).reindex(feat.index)  # next-3h up-excursion (hurts OTM-above NO)
print("\n=== STATE CHARACTERISTICS ===")
print(f"  {'st':>3}{'n':>7}{'%':>6}  {'var_ratio':>10}{'autocorr':>10}{'rv_ratio':>10}{'donch_w':>9}  {'fwd1h|ret|':>11}{'fwd3h_up':>10}")
for s in range(3):
    g=feat[lab.values==s]
    print(f"  {s:>3}{len(g):>7}{len(g)/len(feat):>6.0%}  {g.var_ratio.mean():>10.3f}{g.autocorr.mean():>+10.3f}{g.rv_ratio.mean():>10.3f}{g.donch_width.mean():>9.4f}  {fwd1[lab.values==s].mean():>11.5f}{fwd3[lab.values==s].mean():>+10.4f}")
flip=(lab.diff()!=0).mean(); print(f"\n  flip rate {flip:.1%} (residence ~{1/flip:.0f}h);  self-trans diag: {[round(m.transmat_[i,i],2) for i in range(3)]}")

# === THE MONEY METRIC: OTM-NO EV per state (join archive) ===
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for col in ["p_market","resolved_yes"]: a[col]=pd.to_numeric(a[col],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["p_market","resolved_yes","dt"]); a=a[(a.p_market>=0.10)&(a.p_market<0.40)]
li=lab.index.values; a["st"]=lab.values[np.clip(np.searchsorted(li,a.dt.values,side="right")-1,0,len(lab)-1)]
d=a.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("resolved_yes","first"),st=("st","first")).reset_index()
d["ry"]=d["ry"].round().astype(int); d["ev"]=d.apply(lambda r:(r.pm-r.ry)-FEE*min(r.pm,1-r.pm),axis=1)
print(f"\n=== OTM-NO franchise EV by containment state (the metric that matters) ===")
print(f"  baseline NO EV={d.ev.mean():+.4f}  (n={len(d)})")
for s in range(3):
    g=d[d.st==s]
    if len(g)>20: print(f"  state {s}: n={len(g):>4} NO_WR={(g.ry==0).mean():.0%} NO_EV/ct={g.ev.mean():+.4f}")
