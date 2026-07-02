"""test_jesse_signal_kalshi.py (scratch) — does the Jesse MTF-VWMA trend signal add
value to the Kalshi BTC NO franchise? Compute 1h VWMA(5/8/13/51) stack + use logged adx_1h.
Check NO-EV when 'strong uptrend' fires, and INCREMENTAL value vs the existing donch>0.80 signal."""
import requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
FEE=0.07
def ev_no(pm,ry): return (pm-ry)-FEE*min(pm,1-pm)
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["h","l","c","v"]:df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
def vwma(c,vol,n): return (c*vol).rolling(n).sum()/vol.rolling(n).sum()
s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-21 03:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e)
f5,f8,f13,f51=[vwma(h.c,h.v,n) for n in (5,8,13,51)]
donch=(h.c-h.l.rolling(20).min())/(h.h.rolling(20).max()-h.l.rolling(20).min())
sig=pd.DataFrame({"vwma_up":(f5>f8)&(f8>f13)&(h.c>f51),"vwma_dn":(f5<f8)&(f8<f13)&(h.c<f51),"donch":donch},index=h.index)
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["p_market","resolved_yes","adx_1h"]: a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["p_market","resolved_yes","dt"]); a=a[(a.p_market>=0.10)&(a.p_market<0.40)]
pos=np.clip(np.searchsorted(sig.index.values,a["dt"].values,side="right")-1,0,len(sig)-1)
for col in ["vwma_up","vwma_dn","donch"]: a[col]=sig[col].values[pos]
d=a.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("resolved_yes","first"),
    up=("vwma_up","first"),dn=("vwma_dn","first"),dc=("donch","median"),adx=("adx_1h","median")).reset_index()
d["ry"]=d["ry"].round().astype(int); d["ev"]=d.apply(lambda r:ev_no(r.pm,r.ry),axis=1)
d=d.dropna(subset=["up","dc"])
base=d.ev.mean()
print(f"OTM-NO franchise n={len(d)}  baseline NO EV={base:+.4f}")
print("\n=== NO EV by Jesse 1h-VWMA trend signal ===")
print(f"  vwma strong UP (5>8>13 & c>vwma51):  n={d.up.sum():>4} NO_WR={(d[d.up].ry==0).mean():.0%} EV={d[d.up].ev.mean():+.4f}")
print(f"  vwma strong UP + adx>28:             n={((d.up)&(d.adx>28)).sum():>4} NO_WR={(d[(d.up)&(d.adx>28)].ry==0).mean():.0%} EV={d[(d.up)&(d.adx>28)].ev.mean():+.4f}")
print(f"  neither (no strong trend):           n={(~d.up&~d.dn).sum():>4} EV={d[~d.up&~d.dn].ev.mean():+.4f}")
print(f"  vwma strong DOWN:                    n={d.dn.sum():>4} EV={d[d.dn].ev.mean():+.4f}")
print("\n=== INCREMENTAL vs existing donch>0.80 block ===")
print(f"  donch>0.80 (already blocked):        n={(d.dc>0.8).sum():>4} EV={d[d.dc>0.8].ev.mean():+.4f}")
print(f"  vwma UP+adx>28 AND donch<=0.80 (NOT yet blocked): n={((d.up)&(d.adx>28)&(d.dc<=0.8)).sum():>4} EV={d[(d.up)&(d.adx>28)&(d.dc<=0.8)].ev.mean():+.4f}")
print(f"     ^ if this is clearly negative, the Jesse signal adds value beyond donch.")
# overlap
overlap=((d.up)&(d.adx>28)&(d.dc>0.8)).sum()/max((d.up&(d.adx>28)).sum(),1)
print(f"  overlap: {overlap:.0%} of (vwma UP+adx>28) contracts are ALSO donch>0.80")
