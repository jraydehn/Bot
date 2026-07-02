"""deep_dive_losers.py (scratch) — StochRSI / Bollinger / Donchian on 5m,15m,1h
at each of today's BTC NO trade times. Losers vs winners. Fetches Binance.US 1m."""
import time, requests, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

def fetch_1m(symbol, start_ms, end_ms):
    out=[]; cur=start_ms
    while cur < end_ms:
        r=requests.get("https://api.binance.us/api/v3/klines",
            params={"symbol":symbol,"interval":"1m","startTime":cur,"endTime":end_ms,"limit":1000},timeout=20)
        k=r.json()
        if not k: break
        out+=k; cur=k[-1][0]+60000
        if len(k)<1000: break
    df=pd.DataFrame(out,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"])
    df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True)
    for col in ["o","h","l","c","v"]: df[col]=df[col].astype(float)
    return df.set_index("ts")[["o","h","l","c","v"]].sort_index()

# today's BTC NO trades
p=pd.read_csv("results/paper_trades.csv",low_memory=False)
p["logged_at"]=pd.to_datetime(p["logged_at"],errors="coerce",utc=True,format="mixed")
for c in ["bet_amount","resolved_yes","p_market"]: p[c]=pd.to_numeric(p[c],errors="coerce")
r=p[(p["logged_at"]>=pd.Timestamp("2026-06-19 00:00",tz="UTC"))&(p["bet_amount"]>0)&(p["resolved_yes"].notna())].copy().sort_values("logged_at")
r["won"]=(r["side"]=="no")&(r["resolved_yes"]==0)
t0=int(pd.Timestamp("2026-06-16 00:00",tz="UTC").timestamp()*1000)
t1=int((r["logged_at"].max()+pd.Timedelta(minutes=5)).timestamp()*1000)
m1=fetch_1m("BTCUSDT",t0,t1)
print(f"fetched 1m: {len(m1)} bars {m1.index.min()} -> {m1.index.max()}")

def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn)
def stochrsi_k(c,n=14,k=3):
    rs=rsi(c,n); lo=rs.rolling(n).min(); hi=rs.rolling(n).max()
    sr=((rs-lo)/(hi-lo)).clip(0,1); return sr.rolling(k).mean()*100
def bb(c,n=20,s=2):
    m=c.rolling(n).mean(); sd=c.rolling(n).std(); up=m+s*sd; lo=m-s*sd
    return (c-lo)/(up-lo), (up-lo)/m   # %B, bandwidth
def donch(h,l,c,n=20):
    up=h.rolling(n).max(); lo=l.rolling(n).min(); return (c-lo)/(up-lo)  # 0..1 position

def feats_at(ts, tf):
    o=m1.resample(tf).agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()
    o=o[o.index<=ts]
    if len(o)<25: return {}
    srk=stochrsi_k(o["c"]).iloc[-1]; pb,bw=bb(o["c"]); dc=donch(o["h"],o["l"],o["c"])
    return {f"stochRSI_{tf}":srk, f"bb%B_{tf}":pb.iloc[-1], f"bbWidth_{tf}":bw.iloc[-1], f"donch_{tf}":dc.iloc[-1]}

rows=[]
for _,tr in r.iterrows():
    d={"t":tr["logged_at"].strftime("%H:%M"),"won":tr["won"],"pm":tr["p_market"]}
    for tf in ["5min","15min","1h"]: d.update(feats_at(tr["logged_at"],tf))
    rows.append(d)
f=pd.DataFrame(rows)
pd.set_option("display.width",220)
print("\n=== per-trade indicators (W=win, L=loss) ===")
cols=["t","won","pm"]+[c for c in f.columns if c not in ("t","won","pm")]
print(f[cols].round(2).to_string(index=False))
print("\n=== WINNERS vs LOSERS mean ===")
for c in [c for c in f.columns if c not in ("t","won")]:
    W=f[f.won][c].mean(); L=f[~f.won][c].mean()
    print(f"  {c:<14} WIN={W:>7.2f}  LOSS={L:>7.2f}  diff={L-W:+.2f}")
