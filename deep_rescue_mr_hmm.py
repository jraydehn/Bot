"""
deep_rescue_mr_hmm.py — Exhaustive rescue search for MR HMM gates G1 and G4

G1: mr_state==1 (Trending DOWN / falling knife) + stoch_k_1h < 30 → block YES
G4: mr_state==3 (Deeply oversold everywhere) + stoch_k_1h < 30 → block YES

For each gate's "danger zone", we exhaustively search ALL available signals —
regime HMMs, microstructure, order flow, momentum, market structure, GARCH/vol —
for subsets where YES WR is high enough that blocking would hurt.

Methodology:
  1. Merge all archive sources (main, HMM, HE, error-HMM, SMC)
  2. Build binary rescue-condition columns for every feature × threshold
  3. Single-feature pass: binomtest vs baseline, Bonferroni correction
  4. Pairwise pass: all pairs of top-K single conditions
  5. Triple pass: all triples of top-J pairwise conditions
  6. Walk-forward validation on survivors (train=first half, test=second half)
  7. Report: n, WR, BE, edge, p-value, walk-forward consistency, P&L saved
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import binomtest
from itertools import combinations

FLAT          = 100.0
MIN_N         = 30          # minimum rescue bucket size
MIN_EDGE      = 0.05        # rescue WR must beat BE by at least 5pp
TOP_SINGLE    = 40          # carry top-N into pairwise pass
TOP_PAIR      = 20          # carry top-N pairs into triple pass
SEED          = 42

print("=" * 70)
print("DEEP RESCUE SEARCH — MR HMM Gates G1 and G4")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and merge all archive sources
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading archives...")

KEY = ["logged_at", "contract_ticker"]

# Main archive
main = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)
main["logged_at"] = pd.to_datetime(main["logged_at"], format="mixed", utc=True, errors="coerce")
for c in main.select_dtypes("object").columns:
    if c not in KEY:
        main[c] = pd.to_numeric(main[c], errors="coerce")
print(f"  main:      {len(main):,} rows, {main.columns.tolist().count('logged_at')} time col")

# MR state
mr = pd.read_parquet("results/btc_scan_archive_mr.parquet")
mr["logged_at"] = pd.to_datetime(mr["logged_at"], format="mixed", utc=True, errors="coerce")

# HMM archive (adds stoch_k_1h, stoch_k_4h, stoch_k_15m, rsi_1h, bp_1h, chg_1h, macd_hist_1h, hmm_vol_state, etc.)
hmm_arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
hmm_arc["logged_at"] = pd.to_datetime(hmm_arc["logged_at"], format="mixed", utc=True, errors="coerce")
hmm_extra = [c for c in hmm_arc.columns if c not in main.columns and c not in KEY]
print(f"  hmm:       {len(hmm_arc):,} rows, extra cols: {hmm_extra}")

# HE archive (DC levels, swing highs/lows, L1/L2 hierarchical extremes)
he = pd.read_parquet("results/btc_scan_archive_he.parquet")
he["logged_at"] = pd.to_datetime(he["logged_at"], format="mixed", utc=True, errors="coerce")
he_extra = [c for c in he.columns if c not in main.columns and c not in KEY and c not in hmm_arc.columns]
print(f"  he:        {len(he):,} rows, extra cols: {he_extra[:10]}...")

# Error HMM archive (rolling_cal_err, state_2/3/4, model_disagree)
erh = pd.read_parquet("results/btc_scan_archive_error_hmm.parquet")
erh["logged_at"] = pd.to_datetime(erh["logged_at"], format="mixed", utc=True, errors="coerce")
erh_extra = [c for c in erh.columns if c not in main.columns and c not in KEY and c not in he.columns and c not in hmm_arc.columns]
erh_keep = ["logged_at", "contract_ticker", "rolling_cal_err", "model_disagree",
            "state_2", "state_3", "state_4"]
erh_keep = [c for c in erh_keep if c in erh.columns]
print(f"  error_hmm: {len(erh):,} rows, keeping: {erh_keep}")

# SMC archive
smc = pd.read_csv("results/btc_scan_archive_smc.csv", low_memory=False,
                  usecols=["logged_at","contract_ticker","smc_4h","smc_1h","choch_4h","in_supply"])
smc["logged_at"] = pd.to_datetime(smc["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["smc_4h","smc_1h","choch_4h","in_supply"]:
    smc[c] = pd.to_numeric(smc[c], errors="coerce")
print(f"  smc:       {len(smc):,} rows")

# ─── Merge ───────────────────────────────────────────────────────────────────
print("\n[2] Merging sources on logged_at + contract_ticker...")

df = main.copy()
df = df.merge(mr[KEY + ["mr_state"]], on=KEY, how="left")

# HMM extra cols
hmm_merge_cols = KEY + hmm_extra + ["stoch_k_1h","stoch_k_4h","stoch_k_15m","rsi_1h",
                                     "bp_1h","chg_1h","macd_hist_1h","hmm_vol_state",
                                     "rsi_4h","macd_hist_4h","adx_4h","stoch_k_5m"]
hmm_merge_cols = list(set(c for c in hmm_merge_cols if c in hmm_arc.columns))
df = df.merge(hmm_arc[hmm_merge_cols], on=KEY, how="left")

# HE extra cols
he_merge_cols = KEY + [c for c in he.columns if c in he_extra]
df = df.merge(he[he_merge_cols], on=KEY, how="left")

# Error HMM
df = df.merge(erh[erh_keep], on=KEY, how="left")

# SMC
df = df.merge(smc, on=KEY, how="left")

# Numeric enforcement
for c in df.select_dtypes("object").columns:
    if c not in KEY:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["mr_state","resolved_yes"]).reset_index(drop=True)
df["resolved_yes"] = df["resolved_yes"].astype(int)
df = df.sort_values("logged_at").reset_index(drop=True)

print(f"  Merged df: {len(df):,} rows × {df.shape[1]} cols")
print(f"  Date range: {df['logged_at'].min().date()} → {df['logged_at'].max().date()}")
print(f"  mr_state dist: {dict(df['mr_state'].value_counts().sort_index())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define danger zones
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building danger zones...")

# Need stoch_k_1h
df["stoch_k_1h"] = pd.to_numeric(df.get("stoch_k_1h", np.nan), errors="coerce")

g1_mask = (df["mr_state"] == 1) & (df["stoch_k_1h"] < 30)
g4_mask = (df["mr_state"] == 3) & (df["stoch_k_1h"] < 30)

for name, mask in [("G1 (State1+sk1h<30)", g1_mask), ("G4 (State3+sk1h<30)", g4_mask)]:
    sub = df[mask]
    wr  = sub["resolved_yes"].mean()
    be  = sub["p_market"].mean()
    n   = len(sub)
    pnl = (sub.apply(lambda r: FLAT*(1-r.p_market)/r.p_market if r.resolved_yes==1 else -FLAT, axis=1)).sum()
    print(f"  {name}: n={n:,}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  P&L=${pnl:+,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build rescue condition columns
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Building rescue condition dictionary...")

def safe(df, col):
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")

conditions = {}

# ── Market depth (p_market, offset_pct)
for thr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    conditions[f"pm_ge_{thr}"] = df["p_market"] >= thr
    conditions[f"pm_lt_{thr}"] = df["p_market"] < thr
for lo, hi in [(0.40,0.60),(0.50,0.70),(0.60,0.80),(0.30,0.50)]:
    conditions[f"pm_in_{lo}_{hi}"] = (df["p_market"] >= lo) & (df["p_market"] < hi)

off = safe(df, "offset_pct")
if off is not None:
    for thr in [-5, -3, -2, -1, 0, 1, 2, 3, 5]:
        conditions[f"off_ge_{thr}"] = off >= thr
        conditions[f"off_lt_{thr}"] = off < thr
    conditions["off_otm"] = off < 0
    conditions["off_itm"] = off >= 0

# ── Stochastic oscillators (15m, 4h, raw)
for col, short in [("stoch_k_15m","sk15"), ("stoch_k_4h","sk4h"),
                   ("stoch_k","sk"), ("stoch_k_5m","sk5m")]:
    v = safe(df, col)
    if v is None:
        continue
    for thr in [20, 25, 30, 40, 50, 60, 70, 80]:
        conditions[f"{short}_lt_{thr}"] = v < thr
        conditions[f"{short}_ge_{thr}"] = v >= thr
    conditions[f"{short}_ob"] = v >= 70
    conditions[f"{short}_os"] = v <= 30

# ── RSI
for col, short in [("rsi_1h","rsi1h"), ("rsi_4h","rsi4h")]:
    v = safe(df, col)
    if v is None:
        continue
    for thr in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        conditions[f"{short}_lt_{thr}"] = v < thr
        conditions[f"{short}_ge_{thr}"] = v >= thr

# ── MACD histogram
for col, short in [("macd_hist_1h","mcd1h"), ("macd_hist_4h","mcd4h")]:
    v = safe(df, col)
    if v is None:
        continue
    conditions[f"{short}_pos"] = v > 0
    conditions[f"{short}_neg"] = v < 0
    for thr in [-100, -50, -20, 0, 20, 50, 100]:
        conditions[f"{short}_ge_{thr}"] = v >= thr
        conditions[f"{short}_lt_{thr}"] = v < thr

# ── EMA stack
v = safe(df, "ema_stack_bias")
if v is not None:
    conditions["ema_bull"] = v == 1
    conditions["ema_bear"] = v == -1
    conditions["ema_neut"] = v == 0
    conditions["ema_ge0"]  = v >= 0
    conditions["ema_le0"]  = v <= 0

# ── Composite trend / rev
v = safe(df, "composite_trend")
if v is not None:
    for thr in [-2,-1,0,1,2,3,4,5]:
        conditions[f"ct_ge_{thr}"] = v >= thr
        conditions[f"ct_le_{thr}"] = v <= thr
    conditions["ct_pos"] = v > 0
    conditions["ct_neg"] = v < 0

v = safe(df, "composite_rev")
if v is not None:
    for thr in [0,1,2,3,4]:
        conditions[f"cr_ge_{thr}"] = v >= thr
        conditions[f"cr_le_{thr}"] = v <= thr

# ── p_gbdt, p_up_v2, composite_p_up
for col, short in [("p_gbdt","pgbdt"),("p_up_v2","pv2"),("composite_p_up","cpu")]:
    v = safe(df, col)
    if v is None:
        continue
    for thr in [0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        conditions[f"{short}_ge_{thr}"] = v >= thr
        conditions[f"{short}_lt_{thr}"] = v < thr

# ── Vol / RVOL
v = safe(df, "rvol_1h")
if v is not None:
    for thr in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        conditions[f"rvol_ge_{thr}"] = v >= thr
        conditions[f"rvol_lt_{thr}"] = v < thr

v = safe(df, "vol_score")
if v is not None:
    conditions["vs_pos"] = v > 0
    conditions["vs_neg"] = v < 0
    conditions["vs_ge1"] = v >= 1
    conditions["vs_le-1"] = v <= -1

v = safe(df, "vol_eff")
if v is not None:
    for thr in [0.5, 0.7, 1.0, 1.3, 1.5]:
        conditions[f"veff_ge_{thr}"] = v >= thr
        conditions[f"veff_lt_{thr}"] = v < thr

# ── bp_1h, chg_1h
v = safe(df, "bp_1h")
if v is not None:
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        conditions[f"bp1h_ge_{thr}"] = v >= thr
        conditions[f"bp1h_lt_{thr}"] = v < thr

v = safe(df, "chg_1h")
if v is not None:
    for thr in [-0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        conditions[f"chg1h_ge_{thr}"] = v >= thr
        conditions[f"chg1h_lt_{thr}"] = v < thr

# ── ADX
v = safe(df, "adx_1h")
if v is not None:
    for thr in [15, 20, 25, 30]:
        conditions[f"adx_ge_{thr}"] = v >= thr
        conditions[f"adx_lt_{thr}"] = v < thr

v = safe(df, "adx_4h")
if v is not None:
    for thr in [15, 20, 25, 30]:
        conditions[f"adx4h_ge_{thr}"] = v >= thr
        conditions[f"adx4h_lt_{thr}"] = v < thr

# ── Order flow / market microstructure
for col, short in [("vpin_score","vpin"),("obi_score","obi"),("liq_score","liq")]:
    v = safe(df, col)
    if v is None:
        continue
    conditions[f"{short}_pos"] = v > 0
    conditions[f"{short}_neg"] = v < 0
    for thr in [-1, 0, 1]:
        conditions[f"{short}_ge_{thr}"] = v >= thr
        conditions[f"{short}_le_{thr}"] = v <= thr

v = safe(df, "funding_bias")
if v is not None:
    conditions["fund_bull"] = v == 1
    conditions["fund_bear"] = v == -1
    conditions["fund_neut"] = v == 0
    conditions["fund_ge0"]  = v >= 0
    conditions["fund_le0"]  = v <= 0

v = safe(df, "ls_long_pct")
if v is not None:
    for thr in [45, 50, 55, 60, 65, 70]:
        conditions[f"lsl_ge_{thr}"] = v >= thr
        conditions[f"lsl_lt_{thr}"] = v < thr

v = safe(df, "oi_chg_pct")
if v is not None:
    conditions["oi_pos"] = v > 0
    conditions["oi_neg"] = v < 0
    for thr in [-5, -2, 0, 2, 5]:
        conditions[f"oi_ge_{thr}"] = v >= thr

# ── HMM vol state
v = safe(df, "hmm_vol_state")
if v is not None:
    conditions["hvs_r0"] = v == 0
    conditions["hvs_r1"] = v == 1

# ── Error HMM: rolling_cal_err, model_disagree, state_3
v = safe(df, "rolling_cal_err")
if v is not None:
    for thr in [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        conditions[f"ce_ge_{thr}"] = v >= thr
        conditions[f"ce_lt_{thr}"] = v < thr

v = safe(df, "model_disagree")
if v is not None:
    for thr in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        conditions[f"md_ge_{thr}"] = v >= thr
        conditions[f"md_lt_{thr}"] = v < thr

for col in ["state_2","state_3","state_4"]:
    v = safe(df, col)
    if v is None:
        continue
    short = col.replace("state_","est")
    for s in range(int(v.max())+1 if v.notna().any() else 0):
        conditions[f"{short}_is_{s}"] = v == s

# ── DC direction / HE structure
for col, short in [("dc_direction_60","dc60"),("dc_direction_240","dc240"),
                   ("he_l1_direction","l1dir"),("he_l2_direction","l2dir")]:
    v = safe(df, col)
    if v is None:
        continue
    conditions[f"{short}_bull"] = v == 1
    conditions[f"{short}_bear"] = v == -1
    conditions[f"{short}_flat"] = v == 0

for col, short in [("dist_high_60","dh60"),("dist_high_240","dh240"),
                   ("dist_low_60","dl60"),("dist_low_240","dl240")]:
    v = safe(df, col)
    if v is None:
        continue
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0]:
        conditions[f"{short}_lt_{thr}"] = v < thr
        conditions[f"{short}_ge_{thr}"] = v >= thr

for col in ["strike_above_high_60","strike_above_high_240"]:
    v = safe(df, col)
    if v is not None:
        conditions[col] = v == 1
        conditions[f"not_{col}"] = v == 0

# ── SMC
for col in ["smc_4h","smc_1h"]:
    v = safe(df, col)
    if v is None:
        continue
    short = col.replace("smc_","smc")
    conditions[f"{short}_bull"] = v == 1
    conditions[f"{short}_bear"] = v == -1

for col in ["choch_4h","in_supply"]:
    v = safe(df, col)
    if v is not None:
        conditions[col] = v == 1
        conditions[f"no_{col}"] = v == 0

# ── Confirmation / structure
v = safe(df, "confirmation_score")
if v is not None:
    for thr in [-2, -1, 0, 1, 2]:
        conditions[f"conf_ge_{thr}"] = v >= thr
        conditions[f"conf_le_{thr}"] = v <= thr

v = safe(df, "no_score")
if v is not None:
    for thr in [-1, 0, 1, 2]:
        conditions[f"nos_ge_{thr}"] = v >= thr
        conditions[f"nos_le_{thr}"] = v <= thr

# ── Short-term price action
for col, short in [("chg_5m","c5m"),("chg_10m","c10m"),("chg_30m","c30m"),
                   ("bp_5m","bp5m"),("pm_drift_5m","pmd5m")]:
    v = safe(df, col)
    if v is None:
        continue
    conditions[f"{short}_pos"] = v > 0
    conditions[f"{short}_neg"] = v < 0
    if short in ["c5m","c10m","c30m"]:
        for thr in [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]:
            conditions[f"{short}_ge_{thr}"] = v >= thr

v = safe(df, "squeeze_1h")
if v is not None:
    conditions["squeeze_on"]  = v == 1
    conditions["squeeze_off"] = v == 0

# ── VWAP / EMA stretch
for col, short in [("vwap_stretch_score","vws"),("ema_stretch_score","ems")]:
    v = safe(df, col)
    if v is None:
        continue
    conditions[f"{short}_pos"] = v > 0
    conditions[f"{short}_neg"] = v < 0
    conditions[f"{short}_ge1"] = v >= 1
    conditions[f"{short}_le-1"] = v <= -1
    conditions[f"{short}_ge2"] = v >= 2

# ── Tau (time to expiry)
v = safe(df, "tau_minutes")
if v is not None:
    for thr in [30, 60, 90, 120, 180, 240]:
        conditions[f"tau_ge_{thr}"] = v >= thr
        conditions[f"tau_lt_{thr}"] = v < thr

# ── Hour of day (market session)
df["_hour"] = pd.to_datetime(df["logged_at"]).dt.hour
for h in range(0, 24, 4):
    conditions[f"hour_{h}to{h+4}"] = (df["_hour"] >= h) & (df["_hour"] < h+4)

print(f"  Built {len(conditions):,} candidate rescue conditions")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Rescue search function
# ─────────────────────────────────────────────────────────────────────────────
def rescue_search(gate_name, danger_mask, top_single=TOP_SINGLE, top_pair=TOP_PAIR):
    danger = df[danger_mask].copy()
    n_total = len(danger)
    wr_base  = danger["resolved_yes"].mean()
    be_base  = danger["p_market"].mean()
    edge_base = wr_base - be_base

    def pnl_flat(sub):
        return sum(
            FLAT*(1-r.p_market)/r.p_market if r.resolved_yes==1 else -FLAT
            for _, r in sub.iterrows()
        )

    # Walk-forward split
    midpoint = danger["logged_at"].median()
    train = danger[danger["logged_at"] < midpoint]
    test  = danger[danger["logged_at"] >= midpoint]

    print(f"\n{'='*70}")
    print(f"GATE {gate_name}  —  danger zone n={n_total:,}")
    print(f"  Baseline:  WR={wr_base:.3f}  BE={be_base:.3f}  edge={edge_base:+.3f}")
    print(f"  Train/Test split: {len(train)}/{len(test)} ({midpoint.date()})")
    print(f"{'='*70}")

    # ── Phase 1: single conditions ────────────────────────────────────────────
    single_results = []
    n_tested = 0

    for cname, cond in conditions.items():
        # Align condition to danger zone index
        if len(cond) != len(df):
            continue
        rescue_mask = cond.reindex(danger.index, fill_value=False).fillna(False).astype(bool)
        sub = danger[rescue_mask]
        n   = len(sub)
        if n < MIN_N:
            continue
        wr  = sub["resolved_yes"].mean()
        be  = sub["p_market"].mean()
        edge = wr - be
        if edge < MIN_EDGE:
            continue

        # Binomtest
        wins = int(sub["resolved_yes"].sum())
        bt   = binomtest(wins, n, be, alternative="greater")
        p    = bt.pvalue

        # Walk-forward
        sub_tr = train[cond.reindex(train.index, fill_value=False).fillna(False).astype(bool)]
        sub_te = test[cond.reindex(test.index, fill_value=False).fillna(False).astype(bool)]
        wr_tr  = sub_tr["resolved_yes"].mean() if len(sub_tr) >= 10 else np.nan
        wr_te  = sub_te["resolved_yes"].mean() if len(sub_te) >= 10 else np.nan
        be_tr  = sub_tr["p_market"].mean() if len(sub_tr) >= 1 else np.nan
        be_te  = sub_te["p_market"].mean() if len(sub_te) >= 1 else np.nan
        edge_tr = wr_tr - be_tr if not np.isnan(wr_tr) else np.nan
        edge_te = wr_te - be_te if not np.isnan(wr_te) else np.nan

        single_results.append({
            "name": cname, "n": n, "wr": wr, "be": be, "edge": edge,
            "p": p, "n_tr": len(sub_tr), "edge_tr": edge_tr,
            "n_te": len(sub_te), "edge_te": edge_te
        })
        n_tested += 1

    n_bonf = max(n_tested, 1)
    bonf_thresh = 0.05 / n_bonf

    print(f"\n  [Single-feature pass] {n_tested} conditions tested (Bonf. threshold p<{bonf_thresh:.2e})")
    single_results.sort(key=lambda x: x["p"])

    single_survivors = []
    for r in single_results:
        both_pos = (not np.isnan(r["edge_tr"]) and r["edge_tr"] > 0 and
                    not np.isnan(r["edge_te"]) and r["edge_te"] > 0)
        r["wf_ok"] = both_pos
        if r["p"] < bonf_thresh and both_pos:
            single_survivors.append(r)
            print(f"    ✓ {r['name']:40s} n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
                  f"edge={r['edge']:+.3f}  p={r['p']:.2e}  "
                  f"WF({r['n_tr']:3}/{r['n_te']:3}) tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    if not single_survivors:
        print("    (no Bonferroni-corrected survivors; top-5 by p-value:)")
        for r in single_results[:5]:
            print(f"    ~ {r['name']:40s} n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
                  f"edge={r['edge']:+.3f}  p={r['p']:.2e}  "
                  f"WF tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    # ── Phase 2: pairwise combinations ───────────────────────────────────────
    top_cands = single_results[:top_single]
    pair_results = []
    n_pair_tested = 0

    print(f"\n  [Pairwise pass] combining top-{len(top_cands)} single candidates...")
    for i, (r1, r2) in enumerate(combinations(top_cands, 2)):
        c1 = conditions[r1["name"]]
        c2 = conditions[r2["name"]]
        try:
            combined = (c1.reindex(danger.index, fill_value=False).fillna(False) &
                        c2.reindex(danger.index, fill_value=False).fillna(False))
        except Exception:
            continue
        sub = danger[combined.astype(bool)]
        n   = len(sub)
        if n < MIN_N:
            continue
        wr  = sub["resolved_yes"].mean()
        be  = sub["p_market"].mean()
        edge = wr - be
        if edge < MIN_EDGE:
            continue

        wins = int(sub["resolved_yes"].sum())
        bt   = binomtest(wins, n, be, alternative="greater")
        p    = bt.pvalue

        sub_tr = train[(c1.reindex(train.index, fill_value=False).fillna(False) &
                        c2.reindex(train.index, fill_value=False).fillna(False)).astype(bool)]
        sub_te = test[(c1.reindex(test.index, fill_value=False).fillna(False) &
                       c2.reindex(test.index, fill_value=False).fillna(False)).astype(bool)]
        wr_tr  = sub_tr["resolved_yes"].mean() if len(sub_tr) >= 10 else np.nan
        wr_te  = sub_te["resolved_yes"].mean() if len(sub_te) >= 10 else np.nan
        be_tr  = sub_tr["p_market"].mean() if len(sub_tr) >= 1 else np.nan
        be_te  = sub_te["p_market"].mean() if len(sub_te) >= 1 else np.nan
        edge_tr = wr_tr - be_tr if not np.isnan(wr_tr) else np.nan
        edge_te = wr_te - be_te if not np.isnan(wr_te) else np.nan

        pair_results.append({
            "name": f"{r1['name']} & {r2['name']}",
            "n": n, "wr": wr, "be": be, "edge": edge, "p": p,
            "n_tr": len(sub_tr), "edge_tr": edge_tr,
            "n_te": len(sub_te), "edge_te": edge_te
        })
        n_pair_tested += 1

    n_pair_bonf = max(n_pair_tested, 1)
    pair_thresh  = 0.05 / n_pair_bonf
    print(f"    {n_pair_tested} pairs tested (Bonf. threshold p<{pair_thresh:.2e})")

    pair_results.sort(key=lambda x: x["p"])
    pair_survivors = []
    for r in pair_results:
        both_pos = (not np.isnan(r["edge_tr"]) and r["edge_tr"] > 0 and
                    not np.isnan(r["edge_te"]) and r["edge_te"] > 0)
        r["wf_ok"] = both_pos
        if r["p"] < pair_thresh and both_pos:
            pair_survivors.append(r)
            print(f"    ✓ {r['name'][:80]:80s}\n"
                  f"      n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
                  f"p={r['p']:.2e}  WF({r['n_tr']:3}/{r['n_te']:3}) tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    if not pair_survivors:
        print("    (no Bonferroni-corrected survivors; top-5 pairs by p-value:)")
        for r in pair_results[:5]:
            both_pos = (not np.isnan(r["edge_tr"]) and r["edge_tr"] > 0 and
                        not np.isnan(r["edge_te"]) and r["edge_te"] > 0)
            print(f"    ~ {r['name'][:80]:80s}\n"
                  f"      n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
                  f"p={r['p']:.2e}  WF ok={both_pos} tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    # ── Phase 3: triple combinations from top pairs ────────────────────────
    top_pair_cands = pair_results[:top_pair]
    triple_results = []
    n_triple_tested = 0

    if len(top_pair_cands) >= 3:
        print(f"\n  [Triple pass] combining top-{len(top_pair_cands)} pairs with all single candidates...")
        # For each top pair, combine with each remaining single condition
        for pr in top_pair_cands:
            pc1_name, pc2_name = pr["name"].split(" & ", 1)
            c1 = conditions.get(pc1_name)
            c2 = conditions.get(pc2_name)
            if c1 is None or c2 is None:
                continue
            for cname, c3 in conditions.items():
                if cname in [pc1_name, pc2_name]:
                    continue
                try:
                    combined = (c1.reindex(danger.index, fill_value=False).fillna(False) &
                                c2.reindex(danger.index, fill_value=False).fillna(False) &
                                c3.reindex(danger.index, fill_value=False).fillna(False))
                except Exception:
                    continue
                sub = danger[combined.astype(bool)]
                n   = len(sub)
                if n < MIN_N:
                    continue
                wr  = sub["resolved_yes"].mean()
                be  = sub["p_market"].mean()
                edge = wr - be
                if edge < MIN_EDGE:
                    continue
                wins = int(sub["resolved_yes"].sum())
                bt   = binomtest(wins, n, be, alternative="greater")
                p    = bt.pvalue

                sub_tr = train[(c1.reindex(train.index, fill_value=False).fillna(False) &
                                c2.reindex(train.index, fill_value=False).fillna(False) &
                                c3.reindex(train.index, fill_value=False).fillna(False)).astype(bool)]
                sub_te = test[(c1.reindex(test.index, fill_value=False).fillna(False) &
                               c2.reindex(test.index, fill_value=False).fillna(False) &
                               c3.reindex(test.index, fill_value=False).fillna(False)).astype(bool)]
                wr_tr  = sub_tr["resolved_yes"].mean() if len(sub_tr) >= 10 else np.nan
                wr_te  = sub_te["resolved_yes"].mean() if len(sub_te) >= 10 else np.nan
                be_tr  = sub_tr["p_market"].mean() if len(sub_tr) >= 1 else np.nan
                be_te  = sub_te["p_market"].mean() if len(sub_te) >= 1 else np.nan
                edge_tr = wr_tr - be_tr if not np.isnan(wr_tr) else np.nan
                edge_te = wr_te - be_te if not np.isnan(wr_te) else np.nan

                triple_results.append({
                    "name": f"{pc1_name} & {pc2_name} & {cname}",
                    "n": n, "wr": wr, "be": be, "edge": edge, "p": p,
                    "n_tr": len(sub_tr), "edge_tr": edge_tr,
                    "n_te": len(sub_te), "edge_te": edge_te
                })
                n_triple_tested += 1

    n_triple_bonf = max(n_triple_tested, 1)
    triple_thresh  = 0.05 / n_triple_bonf
    print(f"    {n_triple_tested} triples tested (Bonf. threshold p<{triple_thresh:.2e})")

    triple_results.sort(key=lambda x: x["p"])
    triple_survivors = []
    for r in triple_results:
        both_pos = (not np.isnan(r["edge_tr"]) and r["edge_tr"] > 0 and
                    not np.isnan(r["edge_te"]) and r["edge_te"] > 0)
        r["wf_ok"] = both_pos
        if r["p"] < triple_thresh and both_pos:
            triple_survivors.append(r)
            print(f"    ✓ {r['name'][:100]:100s}\n"
                  f"      n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
                  f"p={r['p']:.2e}  WF tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    if not triple_survivors:
        print("    (no Bonferroni-corrected survivors; top-5 triples by p-value:)")
        for r in triple_results[:5]:
            both_pos = (not np.isnan(r["edge_tr"]) and r["edge_tr"] > 0 and
                        not np.isnan(r["edge_te"]) and r["edge_te"] > 0)
            print(f"    ~ {r['name'][:100]:100s}\n"
                  f"      n={r['n']:4}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
                  f"p={r['p']:.2e}  WF ok={both_pos} tr={r['edge_tr']:+.3f} te={r['edge_te']:+.3f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  SUMMARY for {gate_name}:")
    all_survivors = single_survivors + pair_survivors + triple_survivors
    if all_survivors:
        print(f"  {len(all_survivors)} rescue(s) survived Bonferroni + walk-forward:")
        for r in sorted(all_survivors, key=lambda x: -x["edge"]):
            print(f"    RESCUE: {r['name']}")
            print(f"            n={r['n']}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
                  f"edge={r['edge']:+.3f}  p={r['p']:.2e}")
    else:
        print(f"  NO rescues survived Bonferroni + walk-forward correction.")
        print(f"  Closest candidates (by p-value, all passes combined):")
        all_raw = single_results[:5] + pair_results[:5] + triple_results[:5]
        all_raw.sort(key=lambda x: x["p"])
        for r in all_raw[:8]:
            both_pos = (not np.isnan(r.get("edge_tr", np.nan)) and r.get("edge_tr",0) > 0 and
                        not np.isnan(r.get("edge_te", np.nan)) and r.get("edge_te",0) > 0)
            print(f"    p={r['p']:.2e}  edge={r['edge']:+.3f}  n={r['n']}  WF={both_pos}  → {r['name']}")

    return all_survivors


# ─────────────────────────────────────────────────────────────────────────────
# 6. Run search for G1 and G4
# ─────────────────────────────────────────────────────────────────────────────
rescues_g1 = rescue_search("G1 (State1+sk1h<30)", g1_mask)
rescues_g4 = rescue_search("G4 (State3+sk1h<30)", g4_mask)

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)
print(f"  G1 rescues: {len(rescues_g1)}")
print(f"  G4 rescues: {len(rescues_g4)}")
if not rescues_g1 and not rescues_g4:
    print("  → Both gates are clean blocks. No rescue condition survives scrutiny.")
elif rescues_g1 or rescues_g4:
    print("  → Add listed rescue(s) as exceptions inside the gate logic.")
print("\nDone.")
