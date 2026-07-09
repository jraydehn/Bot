"""
S1 -- SOL 15m losing-streak analysis (KXSOL15M-26JUL090030-30 .. 090145-45,
4 consecutive NO losses, -$300.24 would_pnl).

Goal: find conditions that (a) would have avoided the streak trades, (b) hold
a genuinely NEGATIVE edge across the FULL history of taken NO trades (n=1,572,
05-11 -> 07-09), (c) survive episode-clustered bootstrap + weekly stability.

Discipline (per feedback_zero_lookahead_reconstruction + this session):
- Zero-lookahead reconstruction: every indicator uses only bars whose CLOSE
  <= trade logged_at (cutoff = ts - frame_minutes before searchsorted).
- Episode clustering: consecutive taken NO trades <=45min apart = one episode;
  bootstrap resamples EPISODES, not rows (the 4-loss streak is exactly one
  episode -- treating its trades as independent would fake significance).
- Flat metric: per-trade edge = won - breakeven; $ uses logged would_pnl
  (the model's own intended sizing).
- Streak-coverage tracked per split: how many of the 4 streak trades the
  condition would have blocked.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1009)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

# ── load taken trades ──────────────────────────────────────────────────────
pt = pd.read_csv("results/paper_trades_sol15m.csv", low_memory=False)
pt["logged_at_p"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
for c in pt.columns:
    if c in ("logged_at", "logged_at_p", "decision_time", "asset", "contract_ticker",
             "close_time", "side", "decision") or pt[c].dtype != object:
        continue
    conv = pd.to_numeric(pt[c], errors="coerce")
    if conv.notna().sum() > 0.5 * pt[c].notna().sum():
        pt[c] = conv

taken = pt[pt["decision"] == "trade"].dropna(subset=["resolved_yes", "logged_at_p"]).copy()
taken["side"] = taken["side"].str.lower()
taken["won"] = np.where(taken["side"] == "yes", taken["resolved_yes"] == 1, taken["resolved_yes"] == 0)
taken["be"] = np.where(taken["side"] == "yes", taken["p_market"], 1 - taken["p_market"])
taken["tedge"] = taken["won"].astype(float) - taken["be"]
no = taken[taken["side"] == "no"].sort_values("logged_at_p").reset_index(drop=True).copy()
print(f"NO book: {len(no)} trades  {no['logged_at_p'].min()} -> {no['logged_at_p'].max()}")
print(f"NO book baseline: WR={no['won'].mean():.3f} BE={no['be'].mean():.3f} "
      f"edge={no['tedge'].mean():+.4f}  $={no['would_pnl'].sum():+.2f}")

# episode ids: gap > 45 min -> new episode
gaps = no["logged_at_p"].diff().dt.total_seconds() / 60
no["episode"] = (gaps > 45).cumsum()
n_ep = no["episode"].nunique()
streak_eps = no[no["contract_ticker"].isin(STREAK)]["episode"].unique()
print(f"episodes: {n_ep}; streak trades span episode(s): {streak_eps}")

# ── zero-lookahead multi-TF reconstruction ────────────────────────────────
p1m = sorted(pathlib.Path("data").glob("binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2026-04-15"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
FRAMES = {"5m": (df1m.resample("5min").agg(AGG).dropna(), 5),
          "15m": (df1m.resample("15min").agg(AGG).dropna(), 15),
          "1h": (df1m.resample("1h").agg(AGG).dropna(), 60),
          "4h": (df1m.resample("4h").agg(AGG).dropna(), 240),
          "1d": (df1m.resample("1D").agg(AGG).dropna(), 1440)}
print(f"1m source: {p1m.name} ends {df1m.index.max()}")


def bars_before(df, ts, fm, n):
    cutoff = ts - pd.Timedelta(minutes=fm)
    i = df.index.searchsorted(cutoff, side="right") - 1
    if i < 25:
        return None
    return df.iloc[max(0, i - n):i + 1]


def compute_all(df, ts, fm):
    out = {}
    b = bars_before(df, ts, fm, 60)
    if b is None or len(b) < 25:
        return {k: np.nan for k in ["chg", "bbp", "bbw", "rsi", "sto", "don", "kcp"]}
    c, h, l = b["close"], b["high"], b["low"]
    out["chg"] = float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    up, lo = ma + 2 * sd, ma - 2 * sd
    w = float((up - lo).iloc[-1]); mid = float(ma.iloc[-1])
    out["bbp"] = (float(c.iloc[-1]) - float(lo.iloc[-1])) / w if w > 0 else np.nan
    out["bbw"] = w / mid if mid else np.nan
    d = c.diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls_ = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["rsi"] = float((100 - 100 / (1 + g / ls_.replace(0, np.nan))).iloc[-1])
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    out["sto"] = float((((c - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100).iloc[-1])
    dh, dl = h.rolling(20).max().iloc[-1], l.rolling(20).min().iloc[-1]
    out["don"] = (float(c.iloc[-1]) - dl) / (dh - dl) if dh > dl else np.nan
    e10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(span=14, adjust=False).mean()
    ku, kl = e10 + 1.5 * a14, e10 - 1.5 * a14
    kw = float((ku - kl).iloc[-1])
    out["kcp"] = (float(c.iloc[-1]) - float(kl.iloc[-1])) / kw if kw > 0 else np.nan
    return out


print("reconstructing 7 families x 5 TFs for all NO trades (zero-lookahead)...")
NAME = {"chg": "chg", "bbp": "bb_pctb", "bbw": "bb_width", "rsi": "rsi",
        "sto": "stoch", "don": "donch", "kcp": "kc_pctb"}
for tf, (df, fm) in FRAMES.items():
    res = no["logged_at_p"].apply(lambda ts: compute_all(df, ts, fm))
    for k, nm in NAME.items():
        no[f"r_{nm}_{tf}"] = res.apply(lambda d: d[k])
    print(f"  {tf} done ({no[f'r_stoch_{tf}'].notna().sum()}/{len(no)} coverage)")

no.to_csv(f"{OUT}/no_book_reconstructed.csv", index=False)

# ── sweep: logged numerics + reconstructed + categoricals ─────────────────
EXCLUDE = {"logged_at", "logged_at_p", "decision_time", "asset", "contract_ticker",
           "close_time", "side", "decision", "resolved_yes", "would_win", "would_pnl",
           "spot_at_expiry", "price_move_pct", "miss_pct", "won", "be", "tedge",
           "episode", "kelly_fraction", "bet_fraction", "bet_amount", "bankroll",
           "spot", "floor_strike"}
num_cands, cat_cands = [], []
for c in no.columns:
    if c in EXCLUDE:
        continue
    if no[c].dtype == object:
        if no[c].dropna().nunique() <= 8 and no[c].notna().sum() >= 300:
            cat_cands.append(c)
        continue
    if no[c].notna().sum() >= 300 and no[c].dropna().nunique() > 8:
        num_cands.append(c)
    elif no[c].notna().sum() >= 300:
        cat_cands.append(c)
print(f"\nsweep candidates: {len(num_cands)} numeric, {len(cat_cands)} categorical")

streak_mask = no["contract_ticker"].isin(STREAK)


def ep_boot(sub_idx, n_boot=3000):
    d = no.loc[sub_idx]
    eps = d.groupby("episode")["tedge"].mean()
    e = eps.values
    if len(e) < 8:
        return len(e), np.nan, np.nan
    means = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)])
    return len(e), means.mean(), (means >= 0).mean()   # P(edge>=0): candidate should be NEGATIVE


rows = []
def test_split(feat, label, mask):
    n = int(mask.sum())
    if n < 40 or (len(no) - n) < 100:
        return
    d = no[mask]
    edge = d["tedge"].mean()
    if edge > -0.005:      # only care about meaningfully NEGATIVE buckets
        return
    n_eps, ep_edge, p_pos = ep_boot(no.index[mask])
    wk = d.copy(); wk["week"] = wk["logged_at_p"].dt.to_period("W-FRI")
    wkstats = wk.groupby("week")["tedge"].mean()
    wk_neg_frac = float((wkstats < 0).mean())
    rows.append({"feature": feat, "split": label, "n": n, "episodes": n_eps,
                 "edge": edge, "ep_edge": ep_edge, "P_pos": p_pos,
                 "wk_neg_frac": wk_neg_frac, "n_weeks": len(wkstats),
                 "streak_hits": int((mask & streak_mask).sum()),
                 "pnl": d["would_pnl"].sum()})


n_tests = 0
for feat in num_cands:
    col = no[feat]
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for dlab, mask in [(f">= {th:.4g}", col >= th), (f"< {th:.4g}", col < th)]:
            n_tests += 1
            test_split(feat, dlab, mask.fillna(False))
for feat in cat_cands:
    for val in no[feat].dropna().unique():
        n_tests += 1
        test_split(feat, f"== {val}", (no[feat] == val).fillna(False))

fd = pd.DataFrame(rows)
print(f"{n_tests} splits tested; {len(fd)} negative-edge buckets found")
fd.to_csv(f"{OUT}/sweep_results.csv", index=False)

# survivors: significant negative (P_pos<=0.05), catches >=3 streak trades, weekly-consistent
surv = fd[(fd["P_pos"] <= 0.05) & (fd["streak_hits"] >= 3) & (fd["wk_neg_frac"] >= 0.5)]
print(f"\nSURVIVORS (P(edge>=0)<=0.05, streak_hits>=3, >=50% negative weeks): {len(surv)}")
print(surv.sort_values("ep_edge").head(30).round(4).to_string(index=False))

# also the best significant negatives regardless of streak coverage, for context
sig = fd[(fd["P_pos"] <= 0.02)].sort_values("ep_edge")
print(f"\nall strongly significant negative buckets (P<=0.02), top 20 by edge:")
print(sig.head(20).round(4).to_string(index=False))
print("DONE_S1")
