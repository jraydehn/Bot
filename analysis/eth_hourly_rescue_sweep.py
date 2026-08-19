"""Phase 2 — conditional rescue sweep for ETH hourly pm bands outside the
fav book's range. One-condition rules over MECHANICAL features only
(model-derived signals excluded per feedback_derived_signal_needs_model_oos).

Selection: May-Jun AND July both positive with n>=60 each. Aug-fwd is the
untouched confirm and is NOT used for selection. Accounting identical to
eth_hourly_fav_runner (flat $100, first qualifying scan per contract,
fee = (stake/cost)*0.07*pm*(1-pm)).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
import sys as _sys
ASSET = _sys.argv[1].lower() if len(_sys.argv) > 1 else "eth"
ARCH = BASE / "results" / f"{ASSET}_scan_archive.csv"

FEATS = ["tau_minutes", "offset_pct", "chg_30m", "chg_10m", "chg_5m", "bp_5m",
         "stoch_k", "vol_score", "vpin_score", "obi_score", "no_score",
         "funding_bias", "vol_eff", "pm_drift_5m", "adx_1h", "rvol_1h",
         "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
         "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
         "vwap_distance_pct", "body_15m", "dir_15m", "ichi_bear",
         "confirmation_score", "squeeze_1h", "sigma_dist_high_1pct"]
CATS = ["markov_daily_regime"]

USE = ["logged_at", "contract_ticker", "close_ts", "p_market", "tau_minutes",
       "resolved_yes"] + [f for f in FEATS if f != "tau_minutes"] + CATS
df = pd.read_csv(ARCH, usecols=USE, low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
for c in ["p_market", "resolved_yes"] + FEATS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["dt", "p_market"])
df = df[df["tau_minutes"] > 0]

res = (df.dropna(subset=["resolved_yes"])
         .drop_duplicates("contract_ticker", keep="last")
         .set_index("contract_ticker")["resolved_yes"])
df["outcome"] = df["contract_ticker"].map(res)

SEL = [("May-Jun", "2026-05-01", "2026-07-01"), ("July", "2026-07-01", "2026-08-01")]
CONF = ("Aug-fwd", "2026-08-01", "2026-12-31")
TARGETS = [("yes", 0.50, 0.70), ("yes", 0.70, 0.80),
           ("no", 0.03, 0.20), ("no", 0.20, 0.40)]

def pnl_of(t, side):
    pm = t["p_market"]
    win = (t["outcome"] == 1) if side == "yes" else (t["outcome"] == 0)
    cost = pm if side == "yes" else 1 - pm
    gross = np.where(win, 100 * (1 - cost) / cost, -100.0)
    fee = (100 / cost) * 0.07 * pm * (1 - pm)
    return pd.Series(gross - fee, index=t.index), win

results = []
for side, lo, hi in TARGETS:
    band = df[(df["p_market"] >= lo) & (df["p_market"] < hi)].copy()
    band = band.sort_values("dt")
    # per-rule: first scan per contract where condition holds
    for feat in FEATS:
        col = band[feat]
        if col.notna().sum() < 5000:
            continue
        qs = col.quantile([0.2, 0.35, 0.5, 0.65, 0.8]).unique()
        for q in qs:
            for op in (">=", "<="):
                cond = col >= q if op == ">=" else col <= q
                t = band[cond & band["outcome"].notna()]
                t = t.drop_duplicates("contract_ticker", keep="first")
                pnl, win = pnl_of(t, side)
                ok, wpnls, wns = True, [], []
                for _, ws, we in SEL:
                    m = (t["dt"] >= ws) & (t["dt"] < we)
                    wp, wn = pnl[m].sum(), int(m.sum())
                    wpnls.append(wp); wns.append(wn)
                    if wp <= 0 or wn < 60:
                        ok = False
                if not ok:
                    continue
                mc = (t["dt"] >= CONF[1]) & (t["dt"] < CONF[2])
                results.append({
                    "side": side, "band": f"{lo:.2f}-{hi:.2f}",
                    "rule": f"{feat} {op} {q:.4g}", "n": len(t),
                    "wr": round(win.mean() * 100, 1),
                    "MayJun": round(wpnls[0]), "July": round(wpnls[1]),
                    "Aug": round(pnl[mc].sum()), "Aug_n": int(mc.sum()),
                    "total": round(pnl.sum()),
                    "selmin": round(min(wpnls))})
    # categorical regime rule
    for feat in CATS:
        for val in band[feat].dropna().unique():
            cond = band[feat] == val
            t = band[cond & band["outcome"].notna()].drop_duplicates("contract_ticker", keep="first")
            if len(t) < 200:
                continue
            pnl, win = pnl_of(t, side)
            ok, wpnls = True, []
            for _, ws, we in SEL:
                m = (t["dt"] >= ws) & (t["dt"] < we)
                wp = pnl[m].sum(); wpnls.append(wp)
                if wp <= 0 or m.sum() < 60:
                    ok = False
            if not ok:
                continue
            mc = (t["dt"] >= CONF[1]) & (t["dt"] < CONF[2])
            results.append({"side": side, "band": f"{lo:.2f}-{hi:.2f}",
                            "rule": f"{feat} == {val}", "n": len(t),
                            "wr": round(win.mean() * 100, 1),
                            "MayJun": round(wpnls[0]), "July": round(wpnls[1]),
                            "Aug": round(pnl[mc].sum()), "Aug_n": int(mc.sum()),
                            "total": round(pnl.sum()), "selmin": round(min(wpnls))})

out = pd.DataFrame(results)
if out.empty:
    print("NO rules passed selection (both windows positive, n>=60 each).")
else:
    out = out.sort_values(["side", "band", "selmin"], ascending=[True, True, False])
    for (s, b), g in out.groupby(["side", "band"]):
        print(f"\n=== {s.upper()} {b} — {len(g)} rules passed selection ===")
        print(g.drop(columns=["side", "band"]).head(12).to_string(index=False))
