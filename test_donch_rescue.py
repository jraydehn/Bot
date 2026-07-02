"""test_donch_rescue.py (scratch) — rescue search within the donch>0.80 NO block.
Among the NO bets the gate blocks, split winners/losers by z_strike, offset, pm, tau.
Goal: find a sub-bucket where NO is still +EV (rescue) vs the core losing region."""
import requests, numpy as np, pandas as pd, warnings, math
warnings.filterwarnings("ignore")
FEE=0.07; rng=np.random.default_rng(5)
def ev_no(pm,ry): return (pm-ry)-FEE*min(pm,1-pm)
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"BTCUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["h","l","c"]:df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-21 03:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e); donch=(h.c-h.l.rolling(20).min())/(h.h.rolling(20).max()-h.l.rolling(20).min())
a=pd.read_csv("results/btc_scan_archive.csv",low_memory=False)
for c in ["p_market","resolved_yes","spot","strike","tau_minutes","vol_eff","offset_pct","adx_1h","composite_trend","composite_rev","stoch_k"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["p_market","resolved_yes","spot","strike","tau_minutes","vol_eff","dt"])
a=a[(a.vol_eff>0)&(a.tau_minutes>0)]
pos=np.clip(np.searchsorted(donch.index.values,a["dt"].values,side="right")-1,0,len(donch)-1)
a["donch"]=donch.values[pos]
# z_strike = log(K/S)/(vol*sqrt(tau))  — standardized distance to strike (higher = safer NO)
a["zstrike"]=np.log(a.strike/a.spot)/(a.vol_eff*np.sqrt(a.tau_minutes))
# NO franchise candidates blocked by gate: donch>0.80, NO-bet pm range
b=a[(a.donch>0.80)&(a.p_market>=0.05)&(a.p_market<0.55)].copy()
d=b.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("resolved_yes","first"),
    z=("zstrike","median"),off=("offset_pct","median"),tau=("tau_minutes","median"),
    adx=("adx_1h","median"),ct=("composite_trend","median"),sk=("stoch_k","median"),
    dt=("dt","first")).reset_index()
d["ry"]=d["ry"].round().astype(int); d["ev"]=d.apply(lambda r:ev_no(r.pm,r.ry),axis=1)
d["bewr"]=1-d.pm  # breakeven NO_WR
print(f"=== donch>0.80 blocked NO bets: n={len(d)}  NO_WR={(d.ry==0).mean():.0%}  EV={d.ev.mean():+.4f} ===")
print(f"   (rescue = find a sub-bucket where NO_WR >= breakeven (1-pm) i.e. EV>0)\n")
def split(col,bins,name):
    d["bk"]=pd.cut(d[col],bins)
    print(f"-- by {name} --")
    for ix,g in d.groupby("bk"):
        if len(g)<15: continue
        wr=(g.ry==0).mean(); be=(1-g.pm).mean()
        flag=" <-- RESCUE?" if g.ev.mean()>0 else ""
        print(f"   {str(ix):<14} n={len(g):>4} NO_WR={wr:.0%} BE={be:.0%} EV={g.ev.mean():+.4f}{flag}")
split("pm",[0,.12,.18,.25,.35,.55],"pm (lower=deeper OTM=safer NO)")
split("z",[-5,0,1.0,1.5,2.0,2.5,10],"z_strike (higher=further OTM in sigma)")
split("off",[-0.1,0,0.003,0.006,0.01,0.02,0.1],"offset_pct (higher=strike further above spot)")
split("tau",[0,20,35,50,70,200],"tau_minutes (lower=less time to reach strike)")
# combo: deepest OTM (high z) + short tau
for nm,m in {"zstrike>2.0":d.z>2.0,"zstrike>1.5 & tau<35":(d.z>1.5)&(d.tau<35),
             "pm<0.15":d.pm<0.15,"pm<0.15 & zstrike>2.0":(d.pm<0.15)&(d.z>2.0)}.items():
    g=d[m]
    if len(g)<15: print(f"   {nm}: n={len(g)} (thin)"); continue
    p=(sum(rng.choice(d.ev.values,len(g),replace=False).mean()>=g.ev.mean() for _ in range(2000))+1)/2001
    print(f"\n   [{nm}] n={len(g)} NO_WR={(g.ry==0).mean():.0%} EV={g.ev.mean():+.4f} MCPTp(vs blocked)={p:.3f}")
