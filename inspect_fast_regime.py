"""inspect_fast_regime.py — dump the fast HMM regime's parameters + actual intraday behavior."""
import requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
TRAIN_END=pd.Timestamp("2026-05-16 21:00",tz="UTC")
def fetch(s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":"1h","startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    d=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);d["ts"]=pd.to_datetime(d["t"],unit="ms",utc=True)
    d["c"]=d["c"].astype(float); return d.set_index("ts").sort_index()
h=fetch(int(pd.Timestamp("2024-06-01",tz="UTC").timestamp()*1000),int(pd.Timestamp("2026-06-16 21:00",tz="UTC").timestamp()*1000))
c=h.c; lr=np.log(c/c.shift(1))
feat=pd.DataFrame({"ret2":c/c.shift(2)-1,"ret6":c/c.shift(6)-1,"ret12":c/c.shift(12)-1,"rv6":lr.rolling(6).std()}).dropna()
COLS=["ret2","ret6","ret12","rv6"]
print("="*72)
print("FAST REGIME — PARAMETERS")
print("="*72)
print("  Features (all backward-looking, computed on 1h bars):")
print("    ret2  = 2-hour return    (close/close[-2] - 1)")
print("    ret6  = 6-hour return")
print("    ret12 = 12-hour return")
print("    rv6   = 6-hour realized vol (std of 1h log-returns over 6 bars)")
print("  HMM: GaussianHMM(n_components=3, covariance_type='full', n_iter=500, random_state=42)")
print("  Scaling: StandardScaler  |  Labels: Viterbi (model.predict)")
print("  Trained on bars <= 2026-05-16; tables on 2025-01-01..05-16 (production methodology)")
tr=feat[feat.index<=TRAIN_END]; sc=StandardScaler().fit(tr[COLS])
m=GaussianHMM(n_components=3,covariance_type="full",n_iter=500,random_state=42).fit(sc.transform(tr[COLS]))
lab=pd.Series(m.predict(sc.transform(feat[COLS])),index=feat.index)
nxt=(np.log(c/c.shift(1)).shift(-1)>0).astype(float).reindex(feat.index)
print("\n"+"="*72); print("  STATE CHARACTERISTICS (raw feature means per state)"); print("="*72)
print(f"  {'state':>5}{'n':>7}{'%hrs':>7}{'next_up%':>9}{'ret2':>9}{'ret6':>9}{'ret12':>9}{'rv6':>9}")
for s in range(3):
    g=feat[lab.values==s]
    print(f"  {s:>5}{len(g):>7}{len(g)/len(feat):>7.0%}{nxt[lab.values==s].mean():>9.1%}"
          f"{g.ret2.mean():>+9.4f}{g.ret6.mean():>+9.4f}{g.ret12.mean():>+9.4f}{g.rv6.mean():>9.5f}")
print("\n  Transition matrix (rows=from, cols=to) — diagonal = stickiness:")
for i in range(3):
    print(f"    from {i}: "+"  ".join(f"{m.transmat_[i,j]:.3f}" for j in range(3)))
print(f"\n  mean self-transition (diag): {np.mean(np.diag(m.transmat_)):.3f}")
flip=(lab.diff()!=0).mean(); print(f"  hour-to-hour FLIP rate: {flip:.1%}  → mean residence ≈ {1/flip:.0f} hours")
# run-length distribution
runs=[]; cur=lab.iloc[0]; ln=1
for v in lab.iloc[1:]:
    if v==cur: ln+=1
    else: runs.append(ln); cur=v; ln=1
runs.append(ln); runs=np.array(runs)
print(f"  regime-run lengths (hours): median={np.median(runs):.0f}  mean={runs.mean():.1f}  p10={np.percentile(runs,10):.0f}  p90={np.percentile(runs,90):.0f}")
print(f"  # regime changes total: {len(runs)-1} over {len(lab)} hours")
print("\n"+"="*72); print("  RECENT INTRADAY BEHAVIOR — last 5 days, hourly (regime + what drove it)"); print("="*72)
recent=lab[lab.index>=lab.index.max()-pd.Timedelta(days=5)]
rf=feat.reindex(recent.index)
print(f"  {'time (UTC)':<17}{'regime':>7}{'ret2':>9}{'ret6':>9}{'rv6':>9}   <- = regime CHANGE")
prev=None
for ts,s in recent.items():
    chg=" <- CHANGE" if (prev is not None and s!=prev) else ""
    print(f"  {ts.strftime('%m-%d %H:%M'):<17}{s:>7}{rf.loc[ts,'ret2']:>+9.4f}{rf.loc[ts,'ret6']:>+9.4f}{rf.loc[ts,'rv6']:>9.5f}{chg}")
    prev=s
