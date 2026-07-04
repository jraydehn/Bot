#!/usr/bin/env python3
"""S10 — Does honest v3 add PnL to the BTC HOURLY runner?
Uses STRICTLY walk-forward OOS scores (wf_preds_FINAL.parquet) — never the
final artifact scored in-sample. v3 score for a decision at wall time L is
the WF prediction at bar T = floor(L,'h') - 1h (the forecast of the hour
containing L, exactly what live inference provides at L).

Test 1  Agreement filter on hourly taken trades (both books).
Test 2  Direction-gate rebuild candidate on the deduped scan archive
        (one row per contract, tau nearest 50, resolved_yes labels),
        vs the existing btc_pup_direction_gate blocked population.
Test 3  corr(v3, composite_p_up) at matched hours (informational).

Conventions: $100 flat stakes; entry price = p_market +/- spread/2 (books)
or p_market (archive — no spread logged; stated assumption); Kalshi fee
0.07*price*(1-price) per contract; be_WR = dollar-weighted breakeven =
sum(cost incl fees)/sum(gross payout). One bet per contract_ticker.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
rng = np.random.default_rng(13)

wf = pd.read_parquet(HERE / "wf_preds_FINAL.parquet")["p"]

def join_v3(logged_at: pd.Series) -> pd.Series:
    bars = logged_at.dt.floor("h") - pd.Timedelta(hours=1)
    return pd.Series(wf.reindex(pd.DatetimeIndex(bars, tz="UTC")).values,
                     index=logged_at.index)

def pnl_cols(df, price):
    ct = 100.0 / price
    fee = 0.07 * price * (1 - price) * ct
    won = df["win"].values == 1
    return np.where(won, ct * (1 - price) - fee, -100.0 - fee), ct, fee

def stats(d, tag):
    if not len(d):
        return f"  {tag:<28} n=0"
    be = (100.0 + d['fee']).sum() / (d['ct'].sum())  # cost / gross payout ($1*ct)
    return (f"  {tag:<28} n={len(d):>4}  WR={d['win'].mean():.3f}  "
            f"PnL=${d['pnl'].sum():>8,.0f}  per-bet=${d['pnl'].mean():>7.2f}  be_WR={be:.3f}")

# ════ TEST 1: agreement filter on hourly taken trades ═════════════════════
books = []
for f in ("paper_trades.csv", "paper_trades_pre_regime_pup_20260616.csv"):
    b = pd.read_csv(PROJ / "results" / f, low_memory=False)
    b["book"] = f
    books.append(b)
bk = pd.concat(books, ignore_index=True)
bk = bk[bk["decision"] == "trade"].copy()
bk["res"] = bk["resolved_yes"].astype(str).map({"True": 1, "False": 0})
bk["logged_at"] = pd.to_datetime(bk["logged_at"], utc=True, errors="coerce", format="mixed")
bk = bk[bk["res"].notna() & bk["logged_at"].notna() & bk["p_market"].notna()]
bk["spread"] = pd.to_numeric(bk["spread"], errors="coerce").fillna(0.01).clip(0, 0.10)
bk = bk.sort_values("logged_at").drop_duplicates("contract_ticker", keep="first")
n0 = len(bk)
bk["v3"] = join_v3(bk["logged_at"])
drop_rate = bk["v3"].isna().mean()
bk = bk[bk["v3"].notna()].copy()
print(f"TEST 1 — hourly agreement filter | trades={n0}, v3-join dropped "
      f"{drop_rate:.1%} (WF preds end 2026-07-04 15:00), n={len(bk)}")
price = np.clip(np.where(bk["side"] == "no",
                         1 - bk["p_market"] + bk["spread"] / 2,
                         bk["p_market"] + bk["spread"] / 2), 0.03, 0.99)
bk["win"] = np.where(bk["side"] == "no", 1 - bk["res"], bk["res"])
bk["pnl"], bk["ct"], bk["fee"] = pnl_cols(bk, price)
print(stats(bk, "ALL trades (baseline)"))
print(f"  side mix: {bk['side'].value_counts().to_dict()}")
no_disc = (bk.loc[bk.side == 'no', 'v3'] >= 0.50).mean()
yes_disc = (bk.loc[bk.side == 'yes', 'v3'] < 0.50).mean()
print(f"  filter discrimination: {no_disc:.1%} of NO trades have v3>=0.50 (would drop); "
      f"{yes_disc:.1%} of YES trades have v3<0.50 (would drop)")
for lo, hi, tag in [(0.50, 0.50, "agree@0.50/0.50"),
                    (0.46, 0.54, "strong zone 0.46/0.54"),
                    (0.50, 0.54, "asym NO<=0.50 YES>=0.54"),
                    (0.46, 0.50, "asym NO<=0.46 YES>=0.50"),
                    (0.52, 0.48, "loose NO<=0.52 YES>=0.48")]:
    keep = np.where(bk["side"] == "no", bk["v3"] <= lo, bk["v3"] >= hi)
    print(f" -- {tag}")
    print(stats(bk[keep], "KEPT"))
    print(stats(bk[~keep], "DROPPED"))
    for s in ("no", "yes"):
        m = bk["side"] == s
        print(stats(bk[keep & m], f"  kept {s.upper()}"))
        print(stats(bk[~keep & m], f"  dropped {s.upper()}"))
# weekly consistency @0.50
keep = np.where(bk["side"] == "no", bk["v3"] <= 0.50, bk["v3"] >= 0.50)
bk["wk"] = bk["logged_at"].dt.to_period("W-WED").astype(str)
wkt = bk.assign(keep=keep).groupby(["wk", "keep"])["pnl"].agg(["sum", "count"]).unstack("keep")
print("\n  weekly PnL kept(True) vs dropped(False) @0.50:")
print(wkt.round(0).to_string())
g = bk.assign(keep=keep).groupby("wk").apply(
    lambda x: x[x.keep].pnl.mean() - x[~x.keep].pnl.mean())
print(f"  weeks favoring kept (per-bet): {(g > 0).sum()}/{g.notna().sum()}")

# ════ TEST 2: direction-gate candidate on deduped archive ═════════════════
print("\nTEST 2 — direction-gate rebuild on scan archive (dedup tau~50)")
ar = pd.read_csv(PROJ / "results" / "btc_scan_archive.csv",
                 usecols=["logged_at", "contract_ticker", "tau_minutes",
                          "p_market", "resolved_yes"], low_memory=False)
ar["res"] = pd.to_numeric(ar["resolved_yes"], errors="coerce")
ar["logged_at"] = pd.to_datetime(ar["logged_at"], utc=True, errors="coerce", format="mixed")
ar = ar[ar["res"].notna() & ar["logged_at"].notna() & ar["p_market"].between(0.03, 0.97)]
ar["dtau"] = (ar["tau_minutes"] - 50).abs()
ar = ar.sort_values("dtau").drop_duplicates("contract_ticker", keep="first")
ar["v3"] = join_v3(ar["logged_at"])
print(f"  deduped contracts={len(ar)}, v3 join missing {ar['v3'].isna().mean():.1%}")
# archive coverage holes
days = ar["logged_at"].dt.date.value_counts().sort_index()
full = pd.date_range(days.index.min(), days.index.max(), freq="D").date
missing_days = [d for d in full if d not in days.index]
print(f"  archive day holes: {missing_days}")
ar = ar[ar["v3"].notna()].copy()
ar["wk"] = ar["logged_at"].dt.to_period("W-WED").astype(str)

def side_frame(sub, side):
    d = sub.copy()
    p = np.clip(np.where(side == "no", 1 - d["p_market"], d["p_market"]), 0.03, 0.99)
    d["win"] = np.where(side == "no", 1 - d["res"], d["res"])
    d["pnl"], d["ct"], d["fee"] = pnl_cols(d, p)
    return d

def mcpt_pm_strat(d_all, blocked_mask, n_perm=2000):
    """Permute v3-block flag within pm-decile bins; p = P(perm blocked mean pnl
    <= observed blocked mean pnl). Low p ⇒ blocked pop genuinely worse."""
    obs = d_all.loc[blocked_mask, "pnl"].mean()
    bins = pd.qcut(d_all["p_market"], 10, duplicates="drop")
    k = blocked_mask.values.copy()
    means = np.empty(n_perm)
    grp_idx = [np.where(bins == b)[0] for b in bins.unique()]
    for i in range(n_perm):
        kk = k.copy()
        for gi in grp_idx:
            kk[gi] = rng.permutation(kk[gi])
        means[i] = d_all["pnl"].values[kk].mean()
    return obs, float((means <= obs).mean())

for side, ths, cmp_ in (("no", [0.52, 0.54, 0.56], ">="), ("yes", [0.48, 0.46, 0.44], "<=")):
    d = side_frame(ar, side)
    print(f"\n  hypothetical {side.upper()} book: " + stats(d, "all").strip())
    for th in ths:
        for pmco, ctag in ((None, ""), (0.75, " & pm>=0.75")):
            m = (d["v3"] >= th) if cmp_ == ">=" else (d["v3"] <= th)
            if pmco:
                m &= (d["p_market"] >= pmco) if side == "no" else (d["p_market"] <= 1 - pmco)
            b = d[m]
            if len(b) < 25:
                print(f"    block {side.upper()} v3{cmp_}{th}{ctag}: n={len(b)} (thin)")
                continue
            obs, p = mcpt_pm_strat(d, m)
            wkneg = b.groupby("wk")["pnl"].sum()
            print(f"    block {side.upper()} v3{cmp_}{th}{ctag}: " + stats(b, "").strip()
                  + f"  MCPT(pm-strat) p={p:.4f}  wks_neg={(wkneg<0).sum()}/{len(wkneg)}")

# existing gate blocked population for comparison
bl = pd.read_csv(PROJ / "results" / "blocked_trades.csv",
                 usecols=["logged_at", "gate_name", "contract_ticker", "asset",
                          "side", "pm", "resolved_yes"], low_memory=False)
bl = bl[(bl["gate_name"] == "btc_pup_direction_gate") & (bl["asset"] == "BTC")].copy()
bl["res"] = bl["resolved_yes"].astype(str).map({"True": 1, "False": 0, "1": 1, "0": 0})
bl["pm"] = pd.to_numeric(bl["pm"], errors="coerce")
bl = bl[bl["res"].notna() & bl["pm"].between(0.03, 0.97)]
bl = bl.drop_duplicates("contract_ticker", keep="first")
print("\n  EXISTING btc_pup_direction_gate blocked population (deduped):")
for s in ("no", "yes"):
    d = bl[bl["side"] == s].copy()
    if not len(d):
        continue
    p = np.clip(np.where(s == "no", 1 - d["pm"], d["pm"]), 0.03, 0.99)
    d["win"] = np.where(s == "no", 1 - d["res"], d["res"])
    d["pnl"], d["ct"], d["fee"] = pnl_cols(d, p)
    print(stats(d, f"blocked {s.upper()}"))

# ════ TEST 3: overlap with the current drift inputs (live-logged values) ══
# NOTE: the dataset's composite_p_up column is CONSTANT 0.504 — the
# composite_calibration.json on disk uses "t,r": float keys while the
# training lut expects "t_r": {"p_yes","n"} dicts, so the lookup never fires
# (same dead behavior in training and in btc_p_up_v3_model — parity-safe,
# LGBM ignores constants). Overlap is therefore measured against the
# LIVE-LOGGED composite_p_up / p_up_v2 in the hourly books, one obs per hour.
from scipy.stats import spearmanr
rows3 = []
for f in ("paper_trades.csv", "paper_trades_pre_regime_pup_20260616.csv"):
    rows3.append(pd.read_csv(PROJ / "results" / f,
                             usecols=["logged_at", "composite_p_up", "p_up_v2"],
                             low_memory=False))
b3 = pd.concat(rows3)
b3["logged_at"] = pd.to_datetime(b3["logged_at"], utc=True, errors="coerce", format="mixed")
b3["cp"] = pd.to_numeric(b3["composite_p_up"], errors="coerce")
b3["v2"] = pd.to_numeric(b3["p_up_v2"], errors="coerce")
b3["bar"] = b3["logged_at"].dt.floor("h") - pd.Timedelta(hours=1)
b3["v3"] = wf.reindex(pd.DatetimeIndex(b3["bar"], tz="UTC")).values
h = b3.dropna(subset=["v3", "cp"]).groupby("bar").first()
print(f"\nTEST 3 — matched hours n={len(h)}: corr(v3, live composite_p_up) "
      f"pearson={np.corrcoef(h.v3, h.cp)[0,1]:.3f} "
      f"spearman={spearmanr(h.v3, h.cp).statistic:.3f}")
h2 = b3.dropna(subset=["v3", "v2"]).groupby("bar").first()
print(f"  corr(v3, old leaky p_up_v2): pearson={np.corrcoef(h2.v3, h2.v2)[0,1]:.3f} "
      f"spearman={spearmanr(h2.v3, h2.v2).statistic:.3f} (n={len(h2)})")
print("\nS10 DONE")
