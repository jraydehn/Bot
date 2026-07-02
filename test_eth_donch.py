"""test_eth_donch.py (scratch) — does 1h-Donchian NO-avoid work for ETH too?
EV-by-bucket + net-PnL block test (ETH pipeline: score_to_p_model ETH, mult 0.15, per-expiry cap)."""
import math, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
import composite_scorer as cs
FEE=0.07; BANKROLL=2000.0; KMULT=0.15; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2
def fetch(interval,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":"ETHUSDT","interval":interval,"startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["h","l","c"]:df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-19 22:00",tz="UTC").timestamp()*1000)
h=fetch("1h",s,e); donch=(h["c"]-h["l"].rolling(20).min())/(h["h"].rolling(20).max()-h["l"].rolling(20).min())
print(f"ETH 1h bars {len(h)}")
a=pd.read_csv("results/eth_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)]
pos=np.clip(np.searchsorted(donch.index.values,a["dt"].values,side="right")-1,0,len(donch)-1)
a["donch1h"]=donch.values[pos]; a=a.dropna(subset=["donch1h"]); a["ry"]=a["resolved_yes"].round().astype(int)
# ---- EV by donch bucket (franchise OTM-NO, deduped) ----
o=a[(a.p_market>=0.10)&(a.p_market<0.40)].copy()
d=o.groupby("contract_ticker").agg(pm=("p_market","median"),ry=("ry","first"),dn=("donch1h","median"),dt=("dt","first")).reset_index()
d["ev"]=d.apply(lambda r:(r.pm-r.ry)-FEE*min(r.pm,1-r.pm),axis=1); d["wk"]=d.dt.dt.isocalendar().week
base=d.ev.mean(); pool=d.ev.values; rng=np.random.default_rng(1)
print(f"\nETH OTM-NO franchise n={len(d)} baseline EV={base:+.4f}")
print("NO EV by 1h Donchian position:")
d["db"]=pd.cut(d.dn,[0,.2,.4,.6,.8,1.01])
for ix,g in d.groupby("db"):
    if len(g)>20: wk=g.groupby("wk").ev.mean(); sg=''.join('-' if v<0 else '+' for _,v in wk.items()); print(f"  {str(ix):<11} n={len(g):>4} NO_WR={(g.ry==0).mean():.0%} EV={g.ev.mean():+.4f} wk={sg}")
for nm,m in {"donch>0.8":d.dn>0.8,"donch>0.7":d.dn>0.7}.items():
    sub=d[m]
    if len(sub)>=30:
        ev=sub.ev.mean(); p=(sum(rng.choice(pool,len(sub),replace=False).mean()<=ev for _ in range(2000))+1)/2001
        print(f"  {nm}: n={len(sub)} NO_WR={(sub.ry==0).mean():.0%} EV={ev:+.4f} vs_base={ev-base:+.4f} MCPTp={p:.3f}")
# ---- net-PnL block test ----
def pyes(row):
    sig=row.vol_eff*math.sqrt(row.tau_minutes)
    try: return cs.score_to_p_model(int(row.composite_trend),int(row.composite_rev),row.spot,row.strike,sig,asset="ETH",p_up_override=row.composite_p_up)
    except: return np.nan
a["py"]=a.apply(pyes,axis=1); a=a.dropna(subset=["py"])
fee=FEE*np.minimum(a.p_market,1-a.p_market); a["eyes"]=a.py-a.p_market-fee; a["eno"]=a.p_market-a.py-fee
a["side"]=np.where(a.eyes>=a.eno,"yes","no"); a["edge"]=np.where(a.side=="yes",a.eyes,a.eno)
a["rawk"]=a.edge/np.where(a.side=="yes",a.p_market,1-a.p_market)
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def bt(df,thr,block):
    g=df[df.edge>MIN_EDGE]; tr=set(); ec={}; pnl=0.0; an=0; ap=0.0
    for ts,gg in g.sort_values("logged_at").groupby("logged_at"):
        for _,r in gg.sort_values("edge",ascending=False).iterrows():
            if r.contract_ticker in tr or ec.get(r.close_ts,0)>=EXP_CAP: continue
            if block and r.side=="no" and r.donch1h>thr: continue
            cost=r.p_market if r.side=="yes" else 1-r.p_market
            if cost<=0: continue
            bf=min(r.rawk*KMULT,CAP); pv=(bf*BANKROLL/cost)*unit(r.side,r.p_market,r.ry); pnl+=pv
            if r.side=="no" and r.donch1h>thr: an+=1; ap+=pv
            tr.add(r.contract_ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1; break
    return pnl,an,ap
mid=a.dt.quantile(0.5); H1=a[a.dt<=mid]; H2=a[a.dt>mid]
print(f"\nnet-PnL: block ETH NO when donch_1h>thr (mult 0.15)")
print(f"{'variant':<22}{'FULL_$':>9}{'H1_$':>8}{'H2_$':>8}{'affNO_n':>9}{'affNO_$':>9}")
for lbl,thr,bl in [("baseline",1.1,False),("block donch>0.8",0.8,True),("block donch>0.7",0.7,True)]:
    p,an,ap=bt(a,thr,bl); p1,_,_=bt(H1,thr,bl); p2,_,_=bt(H2,thr,bl)
    print(f"{lbl:<22}{p:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}{an:>9}{ap:>+9.0f}")
