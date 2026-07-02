"""test_donch_boost.py (scratch) — does LOW 1h-Donchian qualify as a NO Kelly booster?
On top of the live high-donch block (>0.80), boost NO size when donch_1h<thr. Net PnL, both halves.
Tests BTC and ETH."""
import math, requests, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
import composite_scorer as cs
FEE=0.07; BANKROLL=2000.0; CAP=0.05; MIN_EDGE=0.02; EXP_CAP=2; BLOCK=0.80
TAB={r:json.load(open(f"reform_results/composite_calibration_regime_{r}.json")) for r in ["Bull","Bear","Sideways"]}
def lookup(t,r,reg):
    key=f"{int(np.clip(t,-5,5))},{int(np.clip(r,-11,11))}"; tb=TAB.get(reg,TAB["Sideways"]); return tb.get(key,tb["__baseline__"])
def fetch(sym,s,e):
    out=[];cur=s
    while cur<e:
        k=requests.get("https://api.binance.us/api/v3/klines",params={"symbol":sym,"interval":"1h","startTime":cur,"endTime":e,"limit":1000},timeout=20).json()
        if not k:break
        out+=k;cur=k[-1][0]+1
        if len(k)<1000:break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]);df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for x in ["h","l","c"]:df[x]=df[x].astype(float)
    return df.set_index("ts").sort_index()
def unit(side,pm,ry):
    f=FEE*min(pm,1-pm); return ((1-pm-f) if ry==1 else -(pm+f)) if side=="yes" else ((pm-f) if ry==0 else -(1-pm+f))

def run(asset):
    sym="BTCUSDT" if asset=="BTC" else "ETHUSDT"
    arc=f"results/{asset.lower()}_scan_archive.csv"
    KMULT=(0.10 if asset=="BTC" else 0.15)  # NO multiplier (BTC NO=0.10, ETH=0.15)
    KYES=(0.90 if asset=="BTC" else None)
    s=int(pd.Timestamp("2026-05-01",tz="UTC").timestamp()*1000); e=int(pd.Timestamp("2026-06-19 22:00",tz="UTC").timestamp()*1000)
    h=fetch(sym,s,e); donch=(h["c"]-h["l"].rolling(20).min())/(h["h"].rolling(20).max()-h["l"].rolling(20).min())
    a=pd.read_csv(arc,low_memory=False)
    for c in ["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes"]: a[c]=pd.to_numeric(a[c],errors="coerce")
    a["dt"]=pd.to_datetime(a["logged_at"],errors="coerce",utc=True,format="mixed")
    a=a.dropna(subset=["composite_trend","composite_rev","composite_p_up","spot","strike","p_market","tau_minutes","vol_eff","resolved_yes","dt"]).sort_values("dt")
    a=a[(a.p_market>0)&(a.p_market<1)&(a.vol_eff>0)]
    pos=np.clip(np.searchsorted(donch.index.values,a["dt"].values,side="right")-1,0,len(donch)-1)
    a["dn"]=donch.values[pos]; a=a.dropna(subset=["dn"]); a["ry"]=a["resolved_yes"].round().astype(int)
    lab=pd.read_parquet("reform_results/hmm_macro_labels_btc.parquet"); lab.index=pd.to_datetime(lab.index,utc=True)
    if asset=="BTC":
        idx=np.clip(np.searchsorted(lab.index.values,a["dt"].values,side="right")-1,0,len(lab)-1); reg=lab["regime"].values[idx]
        a["pup"]=[lookup(t,r,rg) for t,r,rg in zip(a.composite_trend,a.composite_rev,reg)]
        zf=norm.ppf(np.clip(a.pup,0.01,0.99))*np.sqrt(a.tau_minutes/60.0); zk=np.log(a.strike/a.spot)/(a.vol_eff*np.sqrt(a.tau_minutes))
        fee=FEE*np.minimum(a.p_market,1-a.p_market)
        a["eyes"]=norm.sf(zk-zf*0.90)-a.p_market-fee; a["eno"]=a.p_market-norm.sf(zk-zf*0.30)-fee
    else:
        a["py"]=a.apply(lambda r:(cs.score_to_p_model(int(r.composite_trend),int(r.composite_rev),r.spot,r.strike,r.vol_eff*math.sqrt(r.tau_minutes),asset="ETH",p_up_override=r.composite_p_up) if r.vol_eff>0 else np.nan),axis=1)
        a=a.dropna(subset=["py"]); fee=FEE*np.minimum(a.p_market,1-a.p_market); a["eyes"]=a.py-a.p_market-fee; a["eno"]=a.p_market-a.py-fee
    a["side"]=np.where(a.eyes>=a.eno,"yes","no"); a["edge"]=np.where(a.side=="yes",a.eyes,a.eno)
    a["rawk"]=a.edge/np.where(a.side=="yes",a.p_market,1-a.p_market)
    def bt(df,boost_thr,boost_mult,boost_ceil):
        g=df[df.edge>MIN_EDGE]; tr=set(); ec={}; pnl=0.0; bn=0; bp=0.0
        for ts,gg in g.sort_values("logged_at").groupby("logged_at"):
            for _,r in gg.sort_values("edge",ascending=False).iterrows():
                if r.contract_ticker in tr or ec.get(r.close_ts,0)>=EXP_CAP: continue
                if r.side=="no" and r.dn>BLOCK: continue   # live high-donch block
                cost=r.p_market if r.side=="yes" else 1-r.p_market
                if cost<=0: continue
                m=KMULT if (r.side=="no" or asset=="ETH") else 0.05
                boosted=(r.side=="no" and r.dn<boost_thr)
                if boosted: bf=min(r.rawk*m*boost_mult,boost_ceil)
                else: bf=min(r.rawk*m,CAP)
                pv=(bf*BANKROLL/cost)*unit(r.side,r.p_market,r.ry); pnl+=pv
                if boosted: bn+=1; bp+=pv
                tr.add(r.contract_ticker); ec[r.close_ts]=ec.get(r.close_ts,0)+1; break
        return pnl,bn,bp
    mid=a.dt.quantile(0.5); H1=a[a.dt<=mid]; H2=a[a.dt>mid]
    print(f"\n=== {asset} — low-donch NO booster (on top of live >0.80 block) ===")
    print(f"{'variant':<30}{'FULL_$':>9}{'H1_$':>8}{'H2_$':>8}{'boostN':>8}{'boost_$':>9}")
    for lbl,bt_,bm,bc in [("baseline (block only)",0.0,1.0,CAP),("boost donch<0.2 x1.5 ceil.075",0.20,1.5,0.075),
                          ("boost donch<0.2 x2.0 ceil.10",0.20,2.0,0.10),("boost donch<0.3 x1.5 ceil.075",0.30,1.5,0.075)]:
        p,bn,bp=bt(a,bt_,bm,bc); p1,_,_=bt(H1,bt_,bm,bc); p2,_,_=bt(H2,bt_,bm,bc)
        print(f"{lbl:<30}{p:>+9.0f}{p1:>+8.0f}{p2:>+8.0f}{bn:>8}{bp:>+9.0f}")
run("BTC"); run("ETH")
