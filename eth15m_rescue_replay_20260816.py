"""ETH 15m rescue referee — frozen 2026-08-16.

From the regime x signal grid (1,080 combos over the combo book + 5
rejected populations). GATE search on the NOtrio+knife combo: ZERO
survivors (book is gate-saturated). ONE rescue survivor, the third
independent asset-expression of the dislocation-buy class (BTC RESCUE-D,
SOL RESC-1):

  RESC-E: production edge in [0.00, 0.02) & z_spot_6h < -1 &
          kalman_residual < 0   (discovery: n=24, 58% WR, +$703,
          p=0.013, 3/3 wks)

Forward rows (dt >= FREEZE) are decision evidence; evaluate ~08-25.
"""
import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
FREEZE = pd.Timestamp("2026-08-16 09:00", tz="UTC")
df = pd.read_csv("results/paper_trades_eth15m.csv", low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True, format="mixed")
df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
for c in ["p_market", "p_model_15m", "resolved_yes", "kalman_residual", "spot"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
ok = df["spot"].notna()
ss = pd.Series(df.loc[ok, "spot"].values, index=pd.DatetimeIndex(df.loc[ok, "dt"]))
rl = ss.rolling("6h")
df["z_spot_6h"] = np.nan
df.loc[ok, "z_spot_6h"] = ((ss - rl.mean()) / rl.std()).values
start = FREEZE if "--forward" in sys.argv else pd.Timestamp("2026-07-29", tz="UTC")
ab = df[(df["dt"] >= start) & df["resolved_yes"].notna()
        & df["p_market"].between(0.03, 0.97)].dropna(subset=["p_model_15m"]).copy()
fee = 0.07 * ab["p_market"] * (1 - ab["p_market"])
ey = ab["p_model_15m"] - ab["p_market"] - fee
en = ab["p_market"] - ab["p_model_15m"] - fee
ab["side"] = np.where(ey >= en, "yes", "no")
ab["edge"] = np.maximum(ey, en)
q = ab.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
m = ((q["edge"] >= 0.00) & (q["edge"] < 0.02)
     & (q["z_spot_6h"] < -1).fillna(False)
     & (q["kalman_residual"] < 0).fillna(False))
s = q[m]
if not len(s):
    print("RESC-E: no rows yet")
else:
    w = np.where(s["side"] == "yes", s["resolved_yes"] == 1, s["resolved_yes"] == 0)
    c = np.where(s["side"] == "yes", s["p_market"], 1 - s["p_market"])
    f2 = 0.07 * s["p_market"] * (1 - s["p_market"])
    pnl = np.where(w, 100 * (1 - c) / c, -100) - (100 / c) * f2
    wk = pd.Series(pnl, index=s.index).groupby(s["dt"].dt.isocalendar().week).sum()
    print(f"RESC-E: n={len(s)} WR={w.mean():.0%} vs BE={c.mean():.0%} net=${pnl.sum():+,.0f} "
          f"weekly={{{', '.join(f'{int(k)}: {v:+.0f}' for k, v in wk.items())}}}")
