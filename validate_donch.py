"""validate_donch.py (scratch) — does 1h/15m Donchian/BB position predict NO losses on the ARCHIVE?
Fetch 1h+15m candles, compute indicators, join to archive OTM-NO, EV by bucket + net-PnL backtest."""
import json, math, time, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
FEE=0.07
def fetch(symbol,interval,start_ms,end_ms):
    out=[];cur=start_ms
    while cur<end_ms:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":symbol,"interval":interval,
            "startTime":cur,"endTime":end_ms,"limit":1000},timeout=20).json()
        if not k: break
        out+=k; cur=k[-1][0]+1;
        if len(k)<1000: break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"])
    df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["o","h","l","c"]: df[x]=df[x].astype(float)
    return df.set_index("ts")[["o","h","l","c"]].sort_index()
def rsi(c,n=14):
    d=c.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+up/dn)
def indi(df,n=20):
    sr=rsi(df["c"]);lo=sr.rolling(14).min();hi=sr.rolling(14).max();srk=((sr-lo)/(hi-lo)).clip(0,1).rolling(3).mean()*100
    m=df["c"].rolling(n).mean();sd=df["c"].rolling(n).std();pb=(df["c"]-(m-2*sd))/(4*sd)
    dc=(df["c"]-df["l"].rolling(n).min())/(df["h"].rolling(n).max()-df["l"].rolling(n).min())
    return pd.DataFrame({"donch":dc,"bbB":pb,"srsi":srk},index=df.index)
s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-19 06:00",tz="UTC").timestamp()*1000)
h1=indi(fetch("BTCUSDT","1h",s,e)); m15=indi(fetch("BTCUSDT","15m",s,e))
print(f"1h bars {len(h1)}, 15m bars {len(m15)}")
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["p_market","resolved_yes"]: a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["p_market","resolved_yes","dt"]); a=a[(a["p_market"]>=0.10)&(a["p_market"]<0.40)]
d=a.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("resolved_yes","first"),dt=("dt","first")).reset_index()
d["ry"]=d["ry"].round().astype(int); d=d.sort_values("dt")
for nm,ind in [("1h",h1),("15m",m15)]:
    pos=np.clip(np.searchsorted(ind.index.values,d["dt"].values,side="right")-1,0,len(ind)-1)
    for col in ["donch","bbB","srsi"]: d[f"{col}_{nm}"]=ind[col].values[pos]
d["ev"]=d.apply(lambda r:(r.pm-r.ry)-FEE*min(r.pm,1-r.pm),axis=1); d["wk"]=d.dt.dt.isocalendar().week
d=d.dropna(subset=["donch_1h","bbB_1h"])
base=d.ev.mean(); pool=d.ev.values; rng=np.random.default_rng(1)
print(f"\nOTM-NO franchise n={len(d)} baseline EV={base:+.4f}")
print(f"\n=== NO EV by 1h Donchian position (high = price near recent highs) ===")
d["db"]=pd.cut(d["donch_1h"],[0,.2,.4,.6,.8,1.01])
for ix,g in d.groupby("db"):
    if len(g)>30:
        wk=g.groupby("wk").ev.mean(); sg=''.join('-' if v<0 else '+' for _,v in wk.items())
        print(f"  donch_1h{str(ix):<11} n={len(g):>4} NO_WR={(g.ry==0).mean():.0%} EV={g.ev.mean():+.4f} wk={sg}")
print(f"\n=== candidate AVOID conditions (high channel position) — MCPT ===")
for nm,m in {"donch_1h>0.7":d.donch_1h>0.7,"donch_1h>0.8":d.donch_1h>0.8,"bbB_1h>0.8":d.bbB_1h>0.8,
    "donch_15m>0.8":d.donch_15m>0.8,"donch_1h>0.7 & bbB_1h>0.7":(d.donch_1h>0.7)&(d.bbB_1h>0.7),
    "donch_1h>0.6 & donch_15m>0.6":(d.donch_1h>0.6)&(d.donch_15m>0.6)}.items():
    sub=d[m]
    if len(sub)<40: print(f"  {nm:<30} n={len(sub)} (thin)"); continue
    ev=sub.ev.mean(); wk=sub.groupby("wk").ev.mean(); sg=''.join('-' if v<0 else '+' for _,v in wk.items())
    p=(sum(rng.choice(pool,len(sub),replace=False).mean()<=ev for _ in range(2000))+1)/2001
    print(f"  {nm:<30} n={len(sub):>4} NO_WR={(sub.ry==0).mean():.0%} EV={ev:+.4f} vs_base={ev-base:+.4f} wk={sg} MCPTp={p:.3f}")
d.to_parquet("reform_results/_donch_archive.parquet")
print("\nsaved joined frame -> _donch_archive.parquet (for net-PnL backtest next)")
