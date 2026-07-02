"""test_eth_kelly_mult.py (scratch) — ETH hourly Kelly multiplier sweep (net PnL).
Current decision.py uses 0.50 flat for ETH -> 87% of bets hit the 5% cap (flat-max).
Reconstruct p_yes via real score_to_p_model(asset='ETH'); sweep multiplier; per-expiry cap 2."""
import math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import composite_scorer as cs
FEE=0.07; BANKROLL=2000.0; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2
a=pd.read_csv("results/eth_scan_archive.csv",low_memory=False)
for c in ["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]:
    a[c]=pd.to_numeric(a[c],errors="coerce")
a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
a=a.dropna(subset=["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)]
a["ry"]=a["resolved_yes"].round().astype(int)
def pyes(row):
    sig=row.vol_eff*math.sqrt(row.tau_minutes)
    try: return cs.score_to_p_model(int(row.composite_trend),int(row.composite_rev),row.spot,row.strike,sig,asset="ETH",p_up_override=row.composite_p_up)
    except: return np.nan
a["py"]=a.apply(pyes,axis=1); a=a.dropna(subset=["py"])
fee=FEE*np.minimum(a.p_market,1-a.p_market)
a["eyes"]=a.py-a.p_market-fee; a["eno"]=a.p_market-a.py-fee
a["side"]=np.where(a.eyes>=a.eno,"yes","no"); a["edge"]=np.where(a.side=="yes",a.eyes,a.eno)
a["rawk"]=a.edge/np.where(a.side=="yes",a.p_market,1-a.p_market)  # full Kelly fraction proxy
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))
def bt(df,mult):
    d=df[df.edge>MIN_EDGE]
    traded=set(); ec={}; pnl=0.0; n=0; capped=0
    pe=[]
    for ts,g in d.sort_values("logged_at").groupby("logged_at"):
        for _,r in g.sort_values("edge",ascending=False).iterrows():
            if r.contract_ticker in traded or ec.get(r.close_ts,0)>=EXP_CAP: continue
            cost=r.p_market if r.side=="yes" else 1-r.p_market
            if cost<=0: continue
            bf=min(r.rawk*mult,CAP); capped+= bf>=CAP-1e-9
            pv=(bf*BANKROLL/cost)*unit(r.side,r.p_market,r.ry); pnl+=pv; n+=1; pe.append(pv)
            traded.add(r.contract_ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1; break
    pe=np.array(pe); sh=pe.mean()/pe.std() if len(pe)>1 and pe.std()>0 else 0
    return n,pnl,capped/max(n,1),sh
mid=a.dt.quantile(0.5); H1=a[a.dt<=mid]; H2=a[a.dt>mid]
print(f"ETH archive contracts: {len(a)}  side mix: {a.side.value_counts().to_dict()}")
print(f"\n{'mult':>6}{'trades':>7}{'%capped':>9}{'FULL_$':>9}{'H1_$':>8}{'H2_$':>8}{'Sharpe':>8}")
for m in [0.50,0.30,0.20,0.15,0.10,0.075,0.05]:
    n,pnl,cap,sh=bt(a,m); _,p1,_,_=bt(H1,m); _,p2,_,_=bt(H2,m)
    tag="  <-LIVE" if m==0.50 else ""
    print(f"{m:>6.3f}{n:>7}{cap*100:>8.0f}%{pnl:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}{sh:>8.3f}{tag}")
