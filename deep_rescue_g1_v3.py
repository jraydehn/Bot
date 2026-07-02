"""
G1 deep rescue search v3 — exhaustive single + pairwise + triple combos
with ALL available features including 15m archive, offset/tau/drift/vol
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from itertools import combinations

FLAT = 100.0
KEY  = ["logged_at","contract_ticker"]

# ─── load archives ──────────────────────────────────────────────────────────
main = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)
main["logged_at"] = pd.to_datetime(main["logged_at"], format="mixed", utc=True, errors="coerce")
for c in main.select_dtypes("object").columns:
    if c not in KEY: main[c] = pd.to_numeric(main[c], errors="coerce")

mr  = pd.read_parquet("results/btc_scan_archive_mr.parquet")
mr["logged_at"]  = pd.to_datetime(mr["logged_at"],  format="mixed", utc=True, errors="coerce")

hmm = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
hmm["logged_at"] = pd.to_datetime(hmm["logged_at"], format="mixed", utc=True, errors="coerce")
hmm_merge_cols = [c for c in ["logged_at","contract_ticker","stoch_k_1h","stoch_k_4h","stoch_k_15m",
    "rsi_1h","rsi_4h","bp_1h","chg_1h","macd_hist_1h","macd_hist_4h","adx_1h","adx_4h",
    "hmm_vol_state","stoch_k_5m"] if c in hmm.columns]

erh = pd.read_parquet("results/btc_scan_archive_error_hmm.parquet")
erh["logged_at"] = pd.to_datetime(erh["logged_at"], format="mixed", utc=True, errors="coerce")
erh_merge_cols = [c for c in ["logged_at","contract_ticker","rolling_cal_err","model_disagree","state_3"] if c in erh.columns]

he  = pd.read_parquet("results/btc_scan_archive_he.parquet")
he["logged_at"]  = pd.to_datetime(he["logged_at"], format="mixed", utc=True, errors="coerce")
he_extra = [c for c in he.columns if c not in main.columns and c not in KEY]

smc = pd.read_csv("results/btc_scan_archive_smc.csv", low_memory=False,
    usecols=["logged_at","contract_ticker","smc_4h","smc_1h","in_supply","supply_pct","demand_pct"])
smc["logged_at"] = pd.to_datetime(smc["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["smc_4h","smc_1h","in_supply","supply_pct","demand_pct"]:
    if c in smc.columns: smc[c] = pd.to_numeric(smc[c], errors="coerce")

arc15m = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
arc15m["logged_at"] = pd.to_datetime(arc15m["logged_at"], format="mixed", utc=True, errors="coerce")
for c in arc15m.select_dtypes("object").columns:
    if c not in KEY: arc15m[c] = pd.to_numeric(arc15m[c], errors="coerce")
hmm_cols_set = set(hmm.columns)
unique_15m = [c for c in arc15m.columns if c not in main.columns and c not in hmm_cols_set and c not in KEY]

# ─── build merged df ────────────────────────────────────────────────────────
df = main.merge(mr[KEY+["mr_state"]], on=KEY, how="left")
df = df.merge(hmm[hmm_merge_cols], on=KEY, how="left")
df = df.merge(erh[erh_merge_cols], on=KEY, how="left")
df = df.merge(he[KEY+he_extra], on=KEY, how="left")
df = df.merge(smc, on=KEY, how="left")
if unique_15m:
    df = df.merge(arc15m[KEY+unique_15m], on=KEY, how="left")

for c in df.select_dtypes("object").columns:
    if c not in KEY: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["mr_state","resolved_yes"]).reset_index(drop=True)
df["resolved_yes"] = df["resolved_yes"].astype(int)
df = df.sort_values("logged_at").reset_index(drop=True)

# rename adx duplicate if present
if "adx_1h_x" in df.columns and "adx_1h_y" not in df.columns:
    df.rename(columns={"adx_1h_x":"adx_1h"}, inplace=True)
elif "adx_1h_x" in df.columns:
    df.rename(columns={"adx_1h_x":"adx_1h","adx_1h_y":"adx_1h_alt"}, inplace=True)

# ─── G1 zone ────────────────────────────────────────────────────────────────
sk1h = pd.to_numeric(df["stoch_k_1h"], errors="coerce")
g1   = df[(df["mr_state"]==1) & (sk1h<30)].copy().reset_index(drop=True)
BE   = g1["p_market"].mean()
WR   = g1["resolved_yes"].mean()
N    = len(g1)
print(f"G1 zone:  n={N:,}  WR={WR:.3f}  BE={BE:.3f}  edge={WR-BE:+.3f}")

mid_idx = len(g1)//2

def eval_mask(mask, zone=g1):
    sub = zone[mask]
    n = len(sub)
    if n < 30: return None
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    pnl  = sum(FLAT*(1.0/p - 1.0)*y - FLAT*(1.0-y)
               for p,y in zip(sub["p_market"], sub["resolved_yes"]))
    # walk-forward: train (first half), test (second half)
    tr = sub[sub.index < mid_idx]
    te = sub[sub.index >= mid_idx]
    wf_tr = (tr["resolved_yes"].mean() - tr["p_market"].mean()) if len(tr)>=15 else np.nan
    wf_te = (te["resolved_yes"].mean() - te["p_market"].mean()) if len(te)>=15 else np.nan
    return dict(n=n, wr=wr, be=be, edge=edge, pnl=pnl, wf_tr=wf_tr, wf_te=wf_te)

def mcpt(mask, zone=g1, n_perm=2000):
    sub = zone[mask]
    obs_edge = sub["resolved_yes"].mean() - sub["p_market"].mean()
    null = []
    labels = sub["resolved_yes"].values.copy()
    for _ in range(n_perm):
        np.random.shuffle(labels)
        null.append(labels.mean() - sub["p_market"].mean())
    z = (obs_edge - np.mean(null)) / (np.std(null)+1e-12)
    p = (np.array(null) >= obs_edge).mean()
    return z, p

# ─── build candidate conditions ─────────────────────────────────────────────
candidates = []

# Helper: add threshold conditions for a feature
def add_thresh(col, lo, hi, step):
    arr = pd.to_numeric(g1[col], errors="coerce")
    vals = np.arange(lo, hi+step/2, step)
    for v in vals:
        candidates.append((col, "<=", v))
        candidates.append((col, ">=", v))

# OFFSET_PCT — fine bins (the ITM rescue hypothesis)
if "offset_pct" in g1.columns:
    for v in np.arange(-15, 5.5, 0.5):
        candidates.append(("offset_pct", "<=", round(v,1)))
        candidates.append(("offset_pct", ">=", round(v,1)))

# TAU_MINUTES — fine bins
if "tau_minutes" in g1.columns:
    for v in [10,15,20,25,30,45,60,90,120,180]:
        candidates.append(("tau_minutes", "<=", v))
        candidates.append(("tau_minutes", ">=", v))

# P_MARKET — fine bins
if "p_market" in g1.columns:
    for v in np.arange(0.30, 0.95, 0.03):
        candidates.append(("p_market", "<=", round(v,2)))
        candidates.append(("p_market", ">=", round(v,2)))

# Z_DRIFT_6H
if "z_drift_6h" in g1.columns:
    for v in [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]:
        candidates.append(("z_drift_6h", "<=", round(v,1)))
        candidates.append(("z_drift_6h", ">=", round(v,1)))

# REALIZED VOL
if "realized_vol_annual" in g1.columns:
    add_thresh("realized_vol_annual", 0.2, 2.0, 0.1)

# RVOL
if "rvol_1h" in g1.columns:
    for v in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
        candidates.append(("rvol_1h", "<=", v))
        candidates.append(("rvol_1h", ">=", v))

# VOL_RATIO (15m vs baseline)
if "vol_ratio" in g1.columns:
    for v in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        candidates.append(("vol_ratio", "<=", v))
        candidates.append(("vol_ratio", ">=", v))

# ATR_RATIO
if "atr_ratio_15m" in g1.columns:
    for v in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        candidates.append(("atr_ratio_15m", "<=", v))
        candidates.append(("atr_ratio_15m", ">=", v))

# PRICE CHANGE features
for col in ["chg_5m","chg_10m","chg_30m","chg_1m","chg_15m","pm_drift_5m","chg_1h"]:
    if col in g1.columns:
        for v in [-1.5,-1.0,-0.75,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,0.75,1.0,1.5]:
            candidates.append((col, "<=", round(v,2)))
            candidates.append((col, ">=", round(v,2)))

# STOCH features
for col in ["stoch_k","stoch_k_15m","stoch_k_4h","stoch_k_5m"]:
    if col in g1.columns:
        for v in [10,15,20,25,30,40,50,60,70,80]:
            candidates.append((col, "<=", v))
            candidates.append((col, ">=", v))

# RSI
for col in ["rsi_1h","rsi_4h"]:
    if col in g1.columns:
        for v in [20,25,30,35,40,45,50,55,60,70]:
            candidates.append((col, "<=", v))
            candidates.append((col, ">=", v))

# ADX (trend strength)
for col in ["adx_1h","adx_4h"]:
    if col in g1.columns:
        for v in [15,20,25,30,35,40]:
            candidates.append((col, "<=", v))
            candidates.append((col, ">=", v))

# MACD
for col in ["macd_hist_1h","macd_hist_4h"]:
    if col in g1.columns:
        for v in [-200,-100,-50,-20,-10,0,10,20,50,100,200]:
            candidates.append((col, "<=", v))
            candidates.append((col, ">=", v))

# BP features
for col in ["bp_1h","bp_5m","bp_15m"]:
    if col in g1.columns:
        for v in [0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75]:
            candidates.append((col, "<=", v))
            candidates.append((col, ">=", v))

# CANDLE SHAPE (15m)
for col in ["lower_wick_15m","upper_wick_15m","range_ratio_15m","body_15m"]:
    if col in g1.columns:
        for v in np.arange(-0.005, 0.010, 0.001):
            candidates.append((col, "<=", round(v,4)))
            candidates.append((col, ">=", round(v,4)))

# NEAREST RES DIST
if "nearest_res_dist_pct" in g1.columns:
    for v in [0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0,3.0]:
        candidates.append(("nearest_res_dist_pct", "<=", v))
        candidates.append(("nearest_res_dist_pct", ">=", v))

# VWAP DIST
if "vwap_dist" in g1.columns:
    for v in np.arange(-0.03, 0.03, 0.005):
        candidates.append(("vwap_dist", "<=", round(v,4)))
        candidates.append(("vwap_dist", ">=", round(v,4)))

# EMA BIAS
if "ema_bias" in g1.columns:
    for v in [-0.02,-0.01,-0.005,0,0.005,0.01,0.02]:
        candidates.append(("ema_bias", "<=", round(v,4)))
        candidates.append(("ema_bias", ">=", round(v,4)))

# HMM VOL STATE
if "hmm_vol_state" in g1.columns:
    candidates.append(("hmm_vol_state", "==", 0))
    candidates.append(("hmm_vol_state", "==", 1))

# ROLLING CAL ERR
if "rolling_cal_err" in g1.columns:
    for v in [-0.30,-0.20,-0.10,-0.05,0,0.10,0.20,0.30,0.40]:
        candidates.append(("rolling_cal_err", "<=", round(v,2)))
        candidates.append(("rolling_cal_err", ">=", round(v,2)))

# VOL_SCORE
if "vol_score" in g1.columns:
    for v in [-2,-1,0,1,2]:
        candidates.append(("vol_score", "<=", v))
        candidates.append(("vol_score", ">=", v))

# LIQ_SCORE
if "liq_score" in g1.columns:
    for v in [-2,-1,0,1,2]:
        candidates.append(("liq_score", "<=", v))
        candidates.append(("liq_score", ">=", v))

# COMPOSITE P_UP
if "composite_p_up" in g1.columns:
    for v in [0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
        candidates.append(("composite_p_up", "<=", round(v,2)))
        candidates.append(("composite_p_up", ">=", round(v,2)))

# COMPOSITE TREND / REV
if "composite_trend" in g1.columns:
    for v in [-3,-2,-1,0,1,2,3]:
        candidates.append(("composite_trend", "<=", v))
        candidates.append(("composite_trend", ">=", v))
if "composite_rev" in g1.columns:
    for v in [-3,-2,-1,0,1,2,3,4]:
        candidates.append(("composite_rev", "<=", v))
        candidates.append(("composite_rev", ">=", v))

# EMA STACK
if "ema_stack_bias" in g1.columns:
    candidates.append(("ema_stack_bias", "==", -1))
    candidates.append(("ema_stack_bias", "==", 0))
    candidates.append(("ema_stack_bias", "==", 1))

# LS LONG PCT
if "ls_long_pct" in g1.columns:
    for v in [45,50,55,60,65,70,75]:
        candidates.append(("ls_long_pct", "<=", v))
        candidates.append(("ls_long_pct", ">=", v))

# OI CHG
if "oi_chg_pct" in g1.columns:
    for v in [-2,-1,-0.5,-0.2,0,0.2,0.5,1,2]:
        candidates.append(("oi_chg_pct", "<=", round(v,1)))
        candidates.append(("oi_chg_pct", ">=", round(v,1)))

# SWING HIGH DIST
if "dist_high_60" in g1.columns:
    for v in [0.001,0.005,0.01,0.02,0.03,0.05,0.1]:
        candidates.append(("dist_high_60", "<=", v))
        candidates.append(("dist_high_60", ">=", v))
if "dist_high_240" in g1.columns:
    for v in [0.001,0.005,0.01,0.02,0.03,0.05,0.1]:
        candidates.append(("dist_high_240", "<=", v))
        candidates.append(("dist_high_240", ">=", v))

# DC DIRECTION
if "dc_direction_60" in g1.columns:
    candidates.append(("dc_direction_60", "==", 1))
    candidates.append(("dc_direction_60", "==", -1))
if "dc_direction_240" in g1.columns:
    candidates.append(("dc_direction_240", "==", 1))
    candidates.append(("dc_direction_240", "==", -1))

# FEAR GREED
if "fear_greed" in g1.columns:
    for v in [20,30,40,50,60,70,80]:
        candidates.append(("fear_greed", "<=", v))
        candidates.append(("fear_greed", ">=", v))

# IN SUPPLY
if "in_supply" in g1.columns:
    candidates.append(("in_supply", "==", 1))
    candidates.append(("in_supply", "==", 0))

# P_UP_V2
if "p_up_v2_btc" in g1.columns:
    for v in [0.30,0.35,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.60]:
        candidates.append(("p_up_v2_btc", "<=", round(v,2)))
        candidates.append(("p_up_v2_btc", ">=", round(v,2)))

print(f"\nTotal candidate conditions: {len(candidates):,}")

# ─── make mask from condition tuple ─────────────────────────────────────────
def make_mask(cond, zone=g1):
    col, op, val = cond
    if col not in zone.columns:
        return pd.Series([False]*len(zone), index=zone.index)
    arr = pd.to_numeric(zone[col], errors="coerce")
    if op == "<=": return arr <= val
    if op == ">=": return arr >= val
    if op == "==": return arr == val
    return pd.Series([False]*len(zone), index=zone.index)

# ─── PASS 1: single conditions ───────────────────────────────────────────────
print("\n=== PASS 1: Single conditions (rescue = WR > BE) ===")
singles = []
for cond in candidates:
    mask = make_mask(cond)
    if mask.sum() < 30: continue
    r = eval_mask(mask)
    if r is None: continue
    if r["edge"] > 0:
        singles.append((cond, r))

print(f"Conditions with edge > 0: {len(singles)}")

# Bonferroni threshold
bon_thresh = 0.05 / max(len(singles), 1)
print(f"Bonferroni threshold: p < {bon_thresh:.6f}")

# Filter: both WF halves must be positive
wf_positive = [(c,r) for c,r in singles if
               (not np.isnan(r["wf_tr"]) and r["wf_tr"] > 0) and
               (not np.isnan(r["wf_te"]) and r["wf_te"] > 0)]
print(f"WF-positive (both halves): {len(wf_positive)}")

# MCPT on survivors
mcpt_results = []
for cond, r in wf_positive:
    mask = make_mask(cond)
    z, p = mcpt(mask)
    if p < bon_thresh:
        mcpt_results.append((cond, r, z, p))

mcpt_results.sort(key=lambda x: x[1]["edge"], reverse=True)
print(f"\nMCPT-significant singles (p < Bonferroni): {len(mcpt_results)}")
for cond, r, z, p in mcpt_results[:30]:
    print(f"  {cond[0]} {cond[1]} {cond[2]:6.2f}  n={r['n']:4d}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
          f"edge={r['edge']:+.3f}  wf_tr={r['wf_tr']:+.3f}  wf_te={r['wf_te']:+.3f}  z={z:+.1f}  p={p:.4f}")

# Also print ALL singles with edge>0 even if not WF-positive, for visibility
if len(singles) > 0 and len(wf_positive) == 0:
    print("\nAll singles with edge>0 (even without WF positivity):")
    singles.sort(key=lambda x: x[1]["edge"], reverse=True)
    for cond, r in singles[:30]:
        print(f"  {cond[0]} {cond[1]} {cond[2]:6.3f}  n={r['n']:4d}  "
              f"WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
              f"wf_tr={r['wf_tr']:+.3f}  wf_te={r['wf_te']:+.3f}")

# ─── PASS 2: pairwise combinations of top single survivors ──────────────────
print("\n=== PASS 2: Pairwise (from top-100 edge singles + offset×tau mechanics) ===")

# top 100 by edge among all singles (not filtered)
all_singles_sorted = sorted(singles, key=lambda x: x[1]["edge"], reverse=True)
top_singles = [c for c,r in all_singles_sorted[:100]]

# Always add mechanical candidates
mechanical = [
    ("offset_pct", "<=", -3.0),
    ("offset_pct", "<=", -2.0),
    ("offset_pct", "<=", -1.0),
    ("offset_pct", "<=", -5.0),
    ("tau_minutes", "<=", 15),
    ("tau_minutes", "<=", 20),
    ("tau_minutes", "<=", 30),
    ("tau_minutes", "<=", 45),
    ("p_market", ">=", 0.75),
    ("p_market", ">=", 0.80),
    ("p_market", ">=", 0.85),
    ("p_market", ">=", 0.90),
]
if "z_drift_6h" in g1.columns:
    mechanical += [
        ("z_drift_6h", "<=", -2.0),
        ("z_drift_6h", "<=", -1.5),
        ("z_drift_6h", "<=", -1.0),
    ]

all_pool = list({str(c):c for c in top_singles + mechanical}.values())
print(f"Pairwise pool size: {len(all_pool)} → up to {len(all_pool)**2//2:,} pairs")

pairs = []
for c1, c2 in combinations(all_pool, 2):
    mask = make_mask(c1) & make_mask(c2)
    if mask.sum() < 20: continue
    r = eval_mask(mask)
    if r is None: continue
    if r["edge"] > 0:
        pairs.append(((c1,c2), r))

print(f"Pairs with edge > 0: {len(pairs):,}")
bon_thresh2 = 0.05 / max(len(pairs), 1)

wf_pairs = [(c,r) for c,r in pairs if
            (not np.isnan(r["wf_tr"]) and r["wf_tr"] > 0) and
            (not np.isnan(r["wf_te"]) and r["wf_te"] > 0)]
print(f"WF-positive pairs: {len(wf_pairs):,}")

mcpt_pairs = []
for conds, r in wf_pairs:
    mask = make_mask(conds[0]) & make_mask(conds[1])
    z, p = mcpt(mask)
    if p < bon_thresh2:
        mcpt_pairs.append((conds, r, z, p))

mcpt_pairs.sort(key=lambda x: x[1]["pnl"], reverse=True)
print(f"\nMCPT-significant pairs: {len(mcpt_pairs)}")
for conds, r, z, p in mcpt_pairs[:30]:
    c1, c2 = conds
    print(f"  ({c1[0]} {c1[1]} {c1[2]}) & ({c2[0]} {c2[1]} {c2[2]})  "
          f"n={r['n']:4d}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
          f"wf_tr={r['wf_tr']:+.3f}  wf_te={r['wf_te']:+.3f}  z={z:+.1f}  p={p:.4f}")

# ─── PASS 3: triples from MCPT pairs ─────────────────────────────────────────
print("\n=== PASS 3: Triples (extend MCPT pairs with each pool condition) ===")
triples = []
bon_thresh3 = bon_thresh2 / max(len(all_pool), 1)
for conds, _, _, _ in mcpt_pairs[:50]:
    c1, c2 = conds
    for c3 in all_pool:
        mask = make_mask(c1) & make_mask(c2) & make_mask(c3)
        if mask.sum() < 15: continue
        r = eval_mask(mask)
        if r is None: continue
        if r["edge"] > 0 and (not np.isnan(r["wf_tr"]) and r["wf_tr"] > 0) and \
           (not np.isnan(r["wf_te"]) and r["wf_te"] > 0):
            z, p = mcpt(mask)
            if p < bon_thresh3:
                triples.append(((c1,c2,c3), r, z, p))

triples.sort(key=lambda x: x[1]["pnl"], reverse=True)
print(f"MCPT-significant triples: {len(triples)}")
for conds, r, z, p in triples[:20]:
    c1,c2,c3 = conds
    print(f"  ({c1[0]} {c1[1]} {c1[2]}) & ({c2[0]} {c2[1]} {c2[2]}) & ({c3[0]} {c3[1]} {c3[2]})  "
          f"n={r['n']:4d}  WR={r['wr']:.3f}  edge={r['edge']:+.3f}  "
          f"wf_tr={r['wf_tr']:+.3f}  wf_te={r['wf_te']:+.3f}  z={z:+.1f}")

# ─── MECHANICAL CHECK: offset_pct / tau / pm fine grid ───────────────────────
print("\n=== MECHANICAL CHECK: Deep ITM × Short Tau × High PM ===")
mech_checks = [
    ("offset_pct<=−3 & tau<=30",  (g1["offset_pct"]<=-3.0) & (g1["tau_minutes"]<=30)),
    ("offset_pct<=−3 & tau<=20",  (g1["offset_pct"]<=-3.0) & (g1["tau_minutes"]<=20)),
    ("offset_pct<=−2 & tau<=30",  (g1["offset_pct"]<=-2.0) & (g1["tau_minutes"]<=30)),
    ("pm>=0.80",                   g1["p_market"]>=0.80),
    ("pm>=0.85",                   g1["p_market"]>=0.85),
    ("pm>=0.90",                   g1["p_market"]>=0.90),
    ("offset_pct<=−5",             g1["offset_pct"]<=-5.0),
    ("offset_pct<=−3",             g1["offset_pct"]<=-3.0),
    ("offset_pct<=−2",             g1["offset_pct"]<=-2.0),
    ("offset_pct<=−1",             g1["offset_pct"]<=-1.0),
    ("tau<=15",                    g1["tau_minutes"]<=15),
    ("tau<=20",                    g1["tau_minutes"]<=20),
    ("tau<=30",                    g1["tau_minutes"]<=30),
]
if "z_drift_6h" in g1.columns:
    mech_checks += [
        ("z_drift_6h<=-2 & tau<=30", (g1["z_drift_6h"]<=-2.0) & (g1["tau_minutes"]<=30)),
        ("z_drift_6h<=-2 & pm>=0.70", (g1["z_drift_6h"]<=-2.0) & (g1["p_market"]>=0.70)),
        ("z_drift_6h<=-1.5",          g1["z_drift_6h"]<=-1.5),
        ("z_drift_6h<=-2.0",          g1["z_drift_6h"]<=-2.0),
    ]

for label, mask in mech_checks:
    mask = pd.Series(mask).fillna(False)
    sub = g1[mask.values]
    n = len(sub)
    if n < 5:
        print(f"  {label:45s}  n={n:4d}  (too small)")
        continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    pnl = sum(FLAT*(1.0/p-1.0)*y - FLAT*(1.0-y) for p,y in zip(sub["p_market"],sub["resolved_yes"]))
    print(f"  {label:45s}  n={n:4d}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  pnl=${pnl:+,.0f}")

print("\nDone.")
