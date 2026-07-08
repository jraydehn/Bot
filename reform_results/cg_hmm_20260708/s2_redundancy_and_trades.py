"""
S2+S3 -- (a) Redundancy: is cg_state predictable from signals the system
already consumes (of_hmm-adjacent flow columns, vol/macro regime)? If yes,
it's HMM#14 redundancy and we stop. (b) Taken-trade conditioning: per-state
edge / overconfidence gap / PnL on the resolved taken BTC hourly book, with
split-half + bootstrap discipline.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(29)
OUT = "reform_results/cg_hmm_20260708"

st = pd.read_csv(f"{OUT}/cg_states_1h.csv")
st.columns = ["ts", "cg_state"]
st["ts"] = pd.to_datetime(st["ts"], utc=True)
st = st.sort_values("ts")
print(f"cg_state series: {len(st)} hours, {st['ts'].min()} -> {st['ts'].max()}")

p = pd.read_csv("results/paper_trades.csv", low_memory=False)
p["logged_at_p"] = pd.to_datetime(p["logged_at"], format="mixed", utc=True, errors="coerce")
num_cols = ["bet_amount","resolved_yes","p_yes_model","p_market","would_pnl","vpin_score",
            "obi_score","funding_bias","liq_score","liq_bias","ls_long_pct","oi_chg_pct",
            "vol_score","vol_ratio","composite_trend","stoch_k","hour_utc",
            "hmm_ms_state","hmm_vd_state","hmm_of_state"]
for c in num_cols:
    if c in p.columns: p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna(subset=["logged_at_p"]).sort_values("logged_at_p")

# asof-join cg_state (state of the last completed CG hour at/before log time)
p = pd.merge_asof(p, st.rename(columns={"ts": "cg_ts"}), left_on="logged_at_p",
                  right_on="cg_ts", direction="backward", tolerance=pd.Timedelta("2h"))
print(f"paper_trades rows with cg_state: {p['cg_state'].notna().sum()}/{len(p)}")

# ── (a) redundancy: predict cg_state from existing signals ────────────────
red = p.dropna(subset=["cg_state"]).copy()
PRED = ["vpin_score","obi_score","funding_bias","liq_score","liq_bias","ls_long_pct",
        "oi_chg_pct","vol_score","vol_ratio","composite_trend","stoch_k"]
red = red.dropna(subset=PRED)
# dedup to one row per hour (paper_trades logs many rows/hour)
red["hr"] = red["logged_at_p"].dt.floor("h")
red = red.drop_duplicates(subset="hr", keep="last")
print(f"\nredundancy sample (unique hours with all predictors): {len(red)}")
if len(red) >= 200:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    X = red[PRED].values
    y = red["cg_state"].astype(int).values
    base = np.bincount(y).max() / len(y)
    accs = []
    for tr, te in TimeSeriesSplit(n_splits=4).split(X):
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=4)
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    print(f"majority-class baseline: {base:.3f}")
    print(f"RF OOS accuracy from existing signals: {np.mean(accs):.3f} (folds: {[f'{a:.3f}' for a in accs]})")
    print("(near-baseline accuracy = orthogonal; >> baseline = redundant)")

# also crosstab vs ms/vd/of states over the thin post-07-03 overlap
for hcol in ["hmm_ms_state","hmm_vd_state","hmm_of_state"]:
    ov = p.dropna(subset=["cg_state", hcol])
    if len(ov) < 100:
        print(f"{hcol}: overlap n={len(ov)} too thin, skipped")
        continue
    ct = pd.crosstab(ov["cg_state"], ov[hcol], normalize="index")
    # Cramér's V
    from scipy.stats import chi2_contingency
    raw = pd.crosstab(ov["cg_state"], ov[hcol])
    chi2 = chi2_contingency(raw)[0]
    n = raw.values.sum()
    v = np.sqrt(chi2 / (n * (min(raw.shape) - 1)))
    print(f"{hcol}: overlap n={len(ov)}  Cramér's V vs cg_state = {v:.3f}")

# ── (b) taken-trade conditioning ─────────────────────────────────────────
t = p[(p["bet_amount"] > 0) & p["resolved_yes"].notna() & p["p_yes_model"].notna()
      & p["cg_state"].notna()].copy()
t["pw"]  = np.where(t["side"]=="no", 1-t["p_yes_model"], t["p_yes_model"])
t["be"]  = np.where(t["side"]=="no", 1-t["p_market"], t["p_market"])
t["won"] = np.where(t["side"]=="no", t["resolved_yes"]==0, t["resolved_yes"]==1)
t["tedge"] = t["won"].astype(float) - t["be"]
t["week"] = t["logged_at_p"].dt.to_period("W-FRI")
print(f"\ntaken trades with cg_state: {len(t)}  ({t['logged_at_p'].min().date()} -> {t['logged_at_p'].max().date()})")
print(f"baseline: WR={t['won'].mean():.3f} BE={t['be'].mean():.3f} edge={t['tedge'].mean():+.4f} "
      f"gap={(t['pw']-t['won']).mean():+.4f} PnL=${t['would_pnl'].sum():+.2f}\n")

def boot(sub, n_boot=4000):
    e = sub["tedge"].values; n = len(e)
    if n < 5: return (np.nan,)*4
    means = np.array([e[rng.integers(0,n,n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means,2.5), np.percentile(means,97.5), (means<=0).mean()

LBL = {0:"shortliq-quiet",1:"neutral",2:"buy-flow",3:"sell-flow",
       4:"LONG-CASCADE",5:"SHORT-SQUEEZE",6:"longliq-quiet"}
print(f"{'state':<16}{'n':>5}{'WR':>7}{'BE':>7}{'edge':>8}{'gap':>8}{'PnL':>10}{'P(e<=0)':>9}{'wks+':>6}")
for s in sorted(t["cg_state"].unique()):
    sub = t[t["cg_state"]==s]
    if len(sub) < 10:
        print(f"{LBL.get(int(s),s):<16}{len(sub):>5}   (thin)")
        continue
    m, lo, hi, pb = boot(sub)
    wk = sub.groupby("week")["would_pnl"].sum()
    gap = (sub["pw"]-sub["won"]).mean()
    print(f"{LBL.get(int(s),s):<16}{len(sub):>5}{sub['won'].mean():>7.3f}{sub['be'].mean():>7.3f}"
          f"{m:>+8.3f}{gap:>+8.3f}{sub['would_pnl'].sum():>+10.2f}{pb:>9.3f}{(wk>0).mean():>6.2f}")

# split-half: any state whose full-sample edge deviates — does it hold OOS?
split = t["logged_at_p"].quantile(0.5, interpolation="nearest")
tr, te = t[t["logged_at_p"]<=split], t[t["logged_at_p"]>split]
print(f"\nsplit-half (train n={len(tr)}, test n={len(te)}, split at {split.date()}):")
base_tr = tr["tedge"].mean()
for s in sorted(t["cg_state"].unique()):
    a, b = tr[tr["cg_state"]==s], te[te["cg_state"]==s]
    if len(a) < 15 or len(b) < 10: continue
    print(f"  {LBL.get(int(s),s):<16} train: n={len(a)} edge={a['tedge'].mean():+.3f} (Δbase={a['tedge'].mean()-base_tr:+.3f})"
          f"   test: n={len(b)} edge={b['tedge'].mean():+.3f}")
t.to_csv(f"{OUT}/taken_with_cg_state.csv", index=False)
print("\nsaved taken_with_cg_state.csv")
