"""
deep_rescue_mr_hmm_v2.py — Corrected deep rescue search for G1 and G4

Key fix: MIN_EDGE = 0.0 (require only WR > BE, not WR > BE+0.05)
         We show all buckets with positive edge, Bonferroni separates real from noise.

Findings from v1 manual inspection:
  G1 (State1+sk1h<30): WR=0.494 vs BE=0.685 → deeply negative everywhere
      Best single: ce<0 → edge=-0.026. Need combos.
  G4 (State3+sk1h<30): WR=0.597 vs BE=0.628
      Strong rescue: rolling_cal_err < 0.20-0.25 → edge=+0.019 to +0.073

This script:
  Phase A — G1: exhaustive single+pair+triple search with MIN_EDGE=0
  Phase B — G4: exhaustive single+pair+triple search + deep cal_err× decomposition
  Phase C — Walk-forward + MCPT (1000 permutations) on any survivors
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from itertools import combinations
import time

FLAT     = 100.0
MIN_N    = 30
MIN_EDGE = 0.0      # just needs WR > BE
TOP_SINGLE = 60
TOP_PAIR   = 30
SEED     = 42
N_PERM   = 1000

print("=" * 72)
print("DEEP RESCUE SEARCH v2 — MR HMM Gates G1 and G4")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# Load and merge
# ─────────────────────────────────────────────────────────────────────────────
KEY = ["logged_at","contract_ticker"]

main = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)
main["logged_at"] = pd.to_datetime(main["logged_at"], format="mixed", utc=True, errors="coerce")
for c in main.select_dtypes("object").columns:
    if c not in KEY: main[c] = pd.to_numeric(main[c], errors="coerce")

mr  = pd.read_parquet("results/btc_scan_archive_mr.parquet")
mr["logged_at"] = pd.to_datetime(mr["logged_at"], format="mixed", utc=True, errors="coerce")

hmm_arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
hmm_arc["logged_at"] = pd.to_datetime(hmm_arc["logged_at"], format="mixed", utc=True, errors="coerce")
hmm_extra = [c for c in hmm_arc.columns if c not in main.columns and c not in KEY]
hmm_merge = KEY + hmm_extra + [c for c in ["stoch_k_1h","stoch_k_4h","stoch_k_15m",
    "rsi_1h","bp_1h","chg_1h","macd_hist_1h","hmm_vol_state","rsi_4h","macd_hist_4h","adx_4h"] if c in hmm_arc.columns]
hmm_merge = list(set(hmm_merge))

erh = pd.read_parquet("results/btc_scan_archive_error_hmm.parquet")
erh["logged_at"] = pd.to_datetime(erh["logged_at"], format="mixed", utc=True, errors="coerce")
erh_cols = KEY + [c for c in ["rolling_cal_err","model_disagree","state_2","state_3","state_4"] if c in erh.columns]

he = pd.read_parquet("results/btc_scan_archive_he.parquet")
he["logged_at"] = pd.to_datetime(he["logged_at"], format="mixed", utc=True, errors="coerce")
he_extra = [c for c in he.columns if c not in main.columns and c not in KEY and c not in hmm_arc.columns]
he_merge = KEY + he_extra

smc = pd.read_csv("results/btc_scan_archive_smc.csv", low_memory=False,
                  usecols=["logged_at","contract_ticker","smc_4h","smc_1h","choch_4h","in_supply"])
smc["logged_at"] = pd.to_datetime(smc["logged_at"], format="mixed", utc=True, errors="coerce")

# Build merged df
df = main.merge(mr[KEY+["mr_state"]], on=KEY, how="left")
df = df.merge(hmm_arc[hmm_merge], on=KEY, how="left")
df = df.merge(erh[erh_cols], on=KEY, how="left")
df = df.merge(he[he_merge], on=KEY, how="left")
df = df.merge(smc, on=KEY, how="left")

for c in df.select_dtypes("object").columns:
    if c not in KEY: df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["mr_state","resolved_yes"]).reset_index(drop=True)
df["resolved_yes"] = df["resolved_yes"].astype(int)
df["stoch_k_1h"] = pd.to_numeric(df.get("stoch_k_1h", np.nan), errors="coerce")
df = df.sort_values("logged_at").reset_index(drop=True)

print(f"Merged: {len(df):,} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# Danger zones — use rows where stoch_k_1h is populated
# ─────────────────────────────────────────────────────────────────────────────
valid_sk1h = df["stoch_k_1h"].notna()
g1_mask = (df["mr_state"]==1) & (df["stoch_k_1h"]<30) & valid_sk1h
g4_mask = (df["mr_state"]==3) & (df["stoch_k_1h"]<30) & valid_sk1h

for name, mask in [("G1",g1_mask),("G4",g4_mask)]:
    sub = df[mask]
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    n  = len(sub)
    print(f"  {name}: n={n:,}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Condition builder
# ─────────────────────────────────────────────────────────────────────────────
conditions = {}

def add_thresh(short, col, thresholds):
    v = df.get(col)
    if v is None or col not in df.columns: return
    v = pd.to_numeric(v, errors="coerce")
    for thr in thresholds:
        conditions[f"{short}_ge_{thr}"] = v >= thr
        conditions[f"{short}_lt_{thr}"] = v < thr

def add_cat(short, col, values):
    v = df.get(col)
    if v is None or col not in df.columns: return
    v = pd.to_numeric(v, errors="coerce")
    for val in values:
        conditions[f"{short}_eq_{val}"] = v == val
    conditions[f"{short}_ge_0"] = v >= 0
    conditions[f"{short}_lt_0"] = v < 0

# p_market
add_thresh("pm", "p_market", [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90])
for lo,hi in [(0.30,0.60),(0.40,0.70),(0.50,0.80),(0.60,0.90),(0.70,1.0),(0.80,1.0)]:
    conditions[f"pm_in_{lo}_{hi}"] = (df["p_market"]>=lo) & (df["p_market"]<hi)

# offset
add_thresh("off", "offset_pct", [-10,-5,-3,-2,-1,0,1,2,3,5,10])
conditions["off_otm"] = df["offset_pct"] < 0
conditions["off_itm"] = df["offset_pct"] >= 0

# rolling_cal_err — KEY FEATURE
add_thresh("ce", "rolling_cal_err", [-0.30,-0.20,-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40])

# model_disagree
add_thresh("md", "model_disagree", [-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20])

# cal_err error HMM states
for col in ["state_2","state_3","state_4"]:
    v = df.get(col)
    if v is not None and col in df.columns:
        v = pd.to_numeric(v, errors="coerce")
        short = col.replace("state_","est")
        n_states = int(v.dropna().max())+1 if v.notna().any() else 0
        for s in range(n_states):
            conditions[f"{short}_eq_{s}"] = v == s

# stoch oscillators
add_thresh("sk15", "stoch_k_15m", [20,25,30,35,40,50,60,70,75,80])
add_thresh("sk4h", "stoch_k_4h",  [20,25,30,35,40,50,60,70,75,80])
add_thresh("sk5m", "stoch_k_5m",  [20,30,40,50,60,70,80])
add_thresh("sk",   "stoch_k",     [20,30,40,50,60,70,80])

# RSI
add_thresh("rsi1h","rsi_1h",  [25,30,35,40,45,50,55,60,65,70])
add_thresh("rsi4h","rsi_4h",  [25,30,35,40,45,50,55,60,65,70])

# MACD hist
add_thresh("mcd1h","macd_hist_1h", [-150,-100,-50,-20,0,20,50,100,150])
add_thresh("mcd4h","macd_hist_4h", [-150,-100,-50,-20,0,20,50,100,150])

# EMA
add_cat("ema","ema_stack_bias",[-1,0,1])

# composite trend / rev
add_thresh("ct","composite_trend",[-5,-4,-3,-2,-1,0,1,2,3,4,5])
add_thresh("cr","composite_rev",  [-3,-2,-1,0,1,2,3,4])
add_thresh("cpu","composite_p_up",[0.35,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.60])

# p_gbdt, p_up_v2
add_thresh("pgbdt","p_gbdt",   [0.35,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.60])
add_thresh("pv2",  "p_up_v2",  [0.38,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.60])

# vol / RVOL
add_thresh("rvol","rvol_1h",   [0.5,0.7,1.0,1.2,1.5,2.0,2.5])
v = df.get("vol_score")
if v is not None and "vol_score" in df.columns:
    v = pd.to_numeric(v, errors="coerce")
    conditions["vs_pos"] = v > 0
    conditions["vs_neg"] = v < 0
    conditions["vs_ge1"] = v >= 1

# bp_1h, chg_1h
add_thresh("bp1h","bp_1h",  [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65])
add_thresh("c1h", "chg_1h", [-0.5,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.5])

# Order flow
add_cat("fund","funding_bias",[-1,0,1])
add_thresh("obi","obi_score",[-2,-1,0,1,2])
add_thresh("vpin","vpin_score",[-2,-1,0,1,2])
add_thresh("liq","liq_score",[-2,-1,0,1,2])
add_thresh("lsl","ls_long_pct",[45,50,55,60,65,70])
add_thresh("oi","oi_chg_pct",[-5,-2,0,2,5])

# ADX
add_thresh("adx","adx_1h",  [10,15,20,25,30,35])
add_thresh("adx4h","adx_4h",[10,15,20,25,30,35])

# HMM vol state
v = df.get("hmm_vol_state")
if v is not None and "hmm_vol_state" in df.columns:
    v = pd.to_numeric(v, errors="coerce")
    conditions["hvs_r0"] = v == 0
    conditions["hvs_r1"] = v == 1

# DC direction / HE
add_cat("dc60",  "dc_direction_60",  [-1,0,1])
add_cat("dc240", "dc_direction_240", [-1,0,1])
add_cat("l1dir", "he_l1_direction",  [-1,0,1])
add_cat("l2dir", "he_l2_direction",  [-1,0,1])
add_thresh("dh60",  "dist_high_60",  [0.5,1.0,2.0,3.0,5.0,10.0])
add_thresh("dh240", "dist_high_240", [0.5,1.0,2.0,3.0,5.0,10.0])

# SMC
add_cat("smc4h","smc_4h",[-1,0,1])
add_cat("smc1h","smc_1h",[-1,0,1])
for col in ["choch_4h","in_supply"]:
    if col in df.columns:
        v = pd.to_numeric(df[col], errors="coerce")
        conditions[f"{col}_yes"] = v == 1
        conditions[f"{col}_no"]  = v == 0

# vwap/ema stretch
v = df.get("vwap_stretch_score")
if v is not None and "vwap_stretch_score" in df.columns:
    v = pd.to_numeric(v, errors="coerce")
    conditions["vws_pos"] = v > 0
    conditions["vws_neg"] = v < 0
    conditions["vws_ge1"] = v >= 1
    conditions["vws_ge2"] = v >= 2
    conditions["vws_le-1"] = v <= -1
    conditions["vws_le-2"] = v <= -2

# squeeze
if "squeeze_1h" in df.columns:
    v = pd.to_numeric(df["squeeze_1h"], errors="coerce")
    conditions["sqz_on"] = v == 1
    conditions["sqz_off"] = v == 0

# short-term price action
for col, short in [("chg_5m","c5m"),("chg_10m","c10m"),("chg_30m","c30m")]:
    add_thresh(short, col, [-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3])
    if col in df.columns:
        conditions[f"{short}_pos"] = pd.to_numeric(df[col], errors="coerce") > 0

add_thresh("bp5m","bp_5m",[0.30,0.40,0.50,0.60,0.70])

# tau (time to expiry)
add_thresh("tau","tau_minutes",[30,60,90,120,180,240])

# Hour of day
df["_hour"] = pd.to_datetime(df["logged_at"]).dt.hour
for h in range(0,24,4):
    conditions[f"hr_{h}_{h+4}"] = (df["_hour"]>=h) & (df["_hour"]<h+4)

# confirmation/no score
add_thresh("conf","confirmation_score",[-3,-2,-1,0,1,2,3])
add_thresh("nos","no_score",[-2,-1,0,1,2])

print(f"  Built {len(conditions):,} candidate conditions")

# ─────────────────────────────────────────────────────────────────────────────
# Rescue search engine
# ─────────────────────────────────────────────────────────────────────────────
def compute_pnl(sub):
    return sum(FLAT*(1-r.p_market)/r.p_market if r.resolved_yes==1 else -FLAT
               for _, r in sub.iterrows())

def run_rescue_search(gate_name, danger_mask, top_s=TOP_SINGLE, top_p=TOP_PAIR):
    t0 = time.time()
    danger = df[danger_mask].copy()
    n_total = len(danger)
    wr_base = danger["resolved_yes"].mean()
    be_base = danger["p_market"].mean()
    midpoint = danger["logged_at"].median()
    train = danger[danger["logged_at"] < midpoint]
    test  = danger[danger["logged_at"] >= midpoint]

    print(f"\n{'='*72}")
    print(f"GATE {gate_name}  n={n_total:,}  WR={wr_base:.3f}  BE={be_base:.3f}  edge={wr_base-be_base:+.3f}")
    print(f"  Train/Test: {len(train)}/{len(test)} at {midpoint.date()}")
    print(f"{'='*72}")

    def test_cond(cond_series, sub_df, tr_df, te_df):
        m  = cond_series.reindex(sub_df.index, fill_value=False).fillna(False).astype(bool)
        mt = cond_series.reindex(tr_df.index, fill_value=False).fillna(False).astype(bool)
        me = cond_series.reindex(te_df.index, fill_value=False).fillna(False).astype(bool)
        sub = sub_df[m]; s_tr = tr_df[mt]; s_te = te_df[me]
        n = len(sub)
        if n < MIN_N: return None
        wr = sub["resolved_yes"].mean()
        be = sub["p_market"].mean()
        edge = wr - be
        if edge < MIN_EDGE: return None
        wins = int(sub["resolved_yes"].sum())
        bt   = binomtest(wins, n, be, alternative="greater")
        wr_tr = s_tr["resolved_yes"].mean() if len(s_tr)>=10 else np.nan
        be_tr = s_tr["p_market"].mean()     if len(s_tr)>=1  else np.nan
        wr_te = s_te["resolved_yes"].mean() if len(s_te)>=10 else np.nan
        be_te = s_te["p_market"].mean()     if len(s_te)>=1  else np.nan
        e_tr  = wr_tr - be_tr if not (np.isnan(wr_tr) or np.isnan(be_tr)) else np.nan
        e_te  = wr_te - be_te if not (np.isnan(wr_te) or np.isnan(be_te)) else np.nan
        return dict(n=n, wr=wr, be=be, edge=edge, p=bt.pvalue,
                    n_tr=len(s_tr), e_tr=e_tr, n_te=len(s_te), e_te=e_te)

    # ── Phase 1: single ────────────────────────────────────────────────────────
    single = []
    for cname, cseries in conditions.items():
        if not isinstance(cseries, pd.Series): continue
        r = test_cond(cseries, danger, train, test)
        if r: single.append({**r, "name": cname})
    single.sort(key=lambda x: x["p"])
    n_s = len(single)
    bonf_s = 0.05 / max(n_s, 1)
    print(f"\n  Single-feature pass: {n_s} with edge>0  (Bonf p<{bonf_s:.2e})")

    surv_s = [r for r in single if r["p"] < bonf_s and
              not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
              not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0]
    for r in surv_s:
        print(f"    ✓ SINGLE  {r['name']:45s}  n={r['n']:5}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
              f"edge={r['edge']:+.3f}  p={r['p']:.2e}  WF={r['e_tr']:+.3f}/{r['e_te']:+.3f}")
    if not surv_s:
        print("    (no survivors; top candidates:)")
        for r in single[:8]:
            wf_ok = (not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
                     not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0)
            print(f"    ~ {r['name']:45s}  n={r['n']:5}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
                  f"edge={r['edge']:+.3f}  p={r['p']:.2e}  WF={r.get('e_tr',np.nan):+.3f}/{r.get('e_te',np.nan):+.3f}  wf_ok={wf_ok}")

    # ── Phase 2: pairwise ─────────────────────────────────────────────────────
    top_s_cands = single[:top_s]
    pairs = []
    for r1, r2 in combinations(top_s_cands, 2):
        c1 = conditions[r1["name"]]
        c2 = conditions[r2["name"]]
        try:
            combined = c1 & c2
        except Exception:
            continue
        r = test_cond(combined, danger, train, test)
        if r: pairs.append({**r, "name": f"{r1['name']} & {r2['name']}"})
    pairs.sort(key=lambda x: x["p"])
    n_p = len(pairs)
    bonf_p = 0.05 / max(n_p, 1)
    print(f"\n  Pairwise pass: {n_p} with edge>0  (Bonf p<{bonf_p:.2e})")

    surv_p = [r for r in pairs if r["p"] < bonf_p and
              not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
              not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0]
    for r in surv_p:
        print(f"    ✓ PAIR   {r['name'][:80]:80s}")
        print(f"             n={r['n']:5}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
              f"p={r['p']:.2e}  WF={r['e_tr']:+.3f}/{r['e_te']:+.3f}")
    if not surv_p:
        print("    (no survivors; top candidates:)")
        for r in pairs[:10]:
            wf_ok = (not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
                     not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0)
            print(f"    ~ {r['name'][:80]:80s}")
            print(f"      n={r['n']:5}  edge={r['edge']:+.3f}  p={r['p']:.2e}  "
                  f"WF={r.get('e_tr',np.nan):+.3f}/{r.get('e_te',np.nan):+.3f}  ok={wf_ok}")

    # ── Phase 3: triple from top pairs ────────────────────────────────────────
    top_p_cands = pairs[:top_p]
    triples = []
    if top_p_cands:
        print(f"\n  Triple pass: {len(top_p_cands)} pairs × {len(conditions)} conditions...")
        for pr in top_p_cands:
            p1n, p2n = pr["name"].split(" & ", 1)
            c1 = conditions.get(p1n); c2 = conditions.get(p2n)
            if c1 is None or c2 is None: continue
            for cname, c3 in conditions.items():
                if cname in [p1n, p2n]: continue
                try: combined = c1 & c2 & c3
                except Exception: continue
                r = test_cond(combined, danger, train, test)
                if r: triples.append({**r, "name": f"{p1n} & {p2n} & {cname}"})
        triples.sort(key=lambda x: x["p"])
        n_t = len(triples)
        bonf_t = 0.05 / max(n_t, 1)
        print(f"    {n_t} with edge>0  (Bonf p<{bonf_t:.2e})")

        surv_t = [r for r in triples if r["p"] < bonf_t and
                  not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
                  not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0]
        for r in surv_t:
            print(f"    ✓ TRIPLE {r['name'][:80]:80s}")
            print(f"             n={r['n']:5}  WR={r['wr']:.3f}  BE={r['be']:.3f}  edge={r['edge']:+.3f}  "
                  f"p={r['p']:.2e}  WF={r['e_tr']:+.3f}/{r['e_te']:+.3f}")
        if not surv_t:
            print("    (no survivors; top triples:)")
            for r in triples[:8]:
                wf_ok = (not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
                         not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0)
                print(f"    ~ {r['name'][:80]:80s}")
                print(f"      n={r['n']:5}  edge={r['edge']:+.3f}  p={r['p']:.2e}  ok={wf_ok}")
    else:
        surv_t = []; triples = []

    # ── MCPT on all survivors ──────────────────────────────────────────────────
    all_surv = surv_s + surv_p + surv_t
    if all_surv:
        print(f"\n  MCPT (n={N_PERM}) on {len(all_surv)} survivor(s)...")
        rng = np.random.default_rng(SEED)
        for r in all_surv:
            # Find the condition mask for this survivor
            parts = r["name"].split(" & ")
            try:
                cond = conditions[parts[0]]
                for p in parts[1:]: cond = cond & conditions[p]
                m = cond.reindex(danger.index, fill_value=False).fillna(False).astype(bool)
                sub = danger[m]
                obs_wins  = sub["resolved_yes"].sum()
                obs_edge  = sub["resolved_yes"].mean() - sub["p_market"].mean()
                n_sub     = len(sub)
                ry_vals   = danger["resolved_yes"].values.copy()
                perm_edges = []
                for _ in range(N_PERM):
                    rng.shuffle(ry_vals)
                    perm_edges.append(ry_vals[m.values][:n_sub].mean() - sub["p_market"].mean())
                mcpt_z = (obs_edge - np.mean(perm_edges)) / (np.std(perm_edges) + 1e-9)
                mcpt_p = (np.array(perm_edges) >= obs_edge).mean()
                print(f"    {r['name'][:60]:60s}  z={mcpt_z:+.2f}  p={mcpt_p:.3f}")
            except Exception as e:
                print(f"    {r['name'][:60]:60s}  MCPT error: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  ── Summary {gate_name} ({time.time()-t0:.0f}s) ──")
    if all_surv:
        print(f"  {len(all_surv)} rescue(s) survived Bonferroni + walk-forward:")
        for r in sorted(all_surv, key=lambda x: -x["edge"]):
            print(f"    RESCUE: {r['name']}")
            print(f"            n={r['n']}  WR={r['wr']:.3f}  BE={r['be']:.3f}  "
                  f"edge={r['edge']:+.3f}  p={r['p']:.2e}")
    else:
        print(f"  NO rescue survives all filters.")
        # Show best non-survivor
        all_cands = single + pairs + triples
        all_cands.sort(key=lambda x: x["p"])
        print(f"  Closest candidates across all passes (by p-value):")
        for r in all_cands[:6]:
            wf = (not np.isnan(r.get("e_tr",np.nan)) and r.get("e_tr",0)>0 and
                  not np.isnan(r.get("e_te",np.nan)) and r.get("e_te",0)>0)
            print(f"    p={r['p']:.2e}  edge={r['edge']:+.3f}  n={r['n']}  WF={wf}  → {r['name'][:80]}")
    return all_surv


# ─────────────────────────────────────────────────────────────────────────────
# Run searches
# ─────────────────────────────────────────────────────────────────────────────
rescues_g1 = run_rescue_search("G1 (State1+sk1h<30)", g1_mask)
rescues_g4 = run_rescue_search("G4 (State3+sk1h<30)", g4_mask)

# ─────────────────────────────────────────────────────────────────────────────
# Deep dive: cal_err × G4 decomposition with fine thresholds
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("DEEP DIVE: rolling_cal_err × G4 fine decomposition")
print("(finding the precise threshold where edge flips from positive to negative)")
print(f"{'='*72}")

g4 = df[g4_mask].copy()
ce = pd.to_numeric(g4.get("rolling_cal_err", np.nan), errors="coerce")

midpoint = g4["logged_at"].median()

print(f"\n  {'cal_err range':25s} {'n':>6} {'WR':>6} {'BE':>6} {'edge':>7} {'p-val':>10} {'TR edge':>8} {'TE edge':>8}")
for lo, hi in [(-1,-0.20),(-0.20,-0.15),(-0.15,-0.10),(-0.10,-0.05),
               (-0.05,0.0),(0.0,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),
               (0.20,0.25),(0.25,0.30),(0.30,0.35),(0.35,0.40),(0.40,1.0)]:
    mask = (ce >= lo) & (ce < hi)
    sub  = g4[mask]
    if len(sub) < 10: continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    wins = int(sub["resolved_yes"].sum())
    bt = binomtest(wins, len(sub), be, alternative="greater")

    tr = sub[sub["logged_at"] < midpoint]
    te = sub[sub["logged_at"] >= midpoint]
    e_tr = tr["resolved_yes"].mean() - tr["p_market"].mean() if len(tr)>=5 else np.nan
    e_te = te["resolved_yes"].mean() - te["p_market"].mean() if len(te)>=5 else np.nan

    flag = " ← RESCUE" if edge > 0 else ""
    print(f"  ce[{lo:+.2f},{hi:+.2f})  {len(sub):>6}  {wr:.3f}  {be:.3f}  {edge:+.4f}  "
          f"{bt.pvalue:>10.2e}  {e_tr:+.3f}  {e_te:+.3f}{flag}")

# Find the crossover threshold
print(f"\n  Scanning threshold where edge flips positive:")
for thr_lo in np.arange(-0.20, 0.40, 0.01):
    mask = ce < thr_lo
    sub  = g4[mask]
    if len(sub) < 30: continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    if abs(edge) < 0.005:
        print(f"    Zero crossing near ce < {thr_lo:+.2f}  (n={len(sub)}, WR={wr:.3f}, BE={be:.3f})")
        break

# Optimal rescue threshold grid
print(f"\n  Rescue threshold grid (ce < X):")
print(f"  {'ce_threshold':>14} {'n_rescue':>9} {'WR':>6} {'BE':>6} {'edge':>7} {'p':>10} {'WF-TR':>7} {'WF-TE':>7}")
for thr in [0.05, 0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
    mask = ce < thr
    sub  = g4[mask]
    if len(sub) < 10: continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    wins = int(sub["resolved_yes"].sum())
    bt = binomtest(wins, len(sub), be, alternative="greater")
    tr = sub[sub["logged_at"] < midpoint]
    te = sub[sub["logged_at"] >= midpoint]
    e_tr = tr["resolved_yes"].mean() - tr["p_market"].mean() if len(tr)>=5 else np.nan
    e_te = te["resolved_yes"].mean() - te["p_market"].mean() if len(te)>=5 else np.nan
    flag = " ✓" if edge > 0 else ""
    print(f"  ce < {thr:+.2f}       {len(sub):>9}  {wr:.3f}  {be:.3f}  {edge:+.4f}  "
          f"{bt.pvalue:>10.2e}  {e_tr:+.3f}  {e_te:+.3f}{flag}")

# For G1: same cal_err analysis
print(f"\n{'='*72}")
print("DEEP DIVE: rolling_cal_err × G1 fine decomposition")
print(f"{'='*72}")

g1 = df[g1_mask].copy()
ce1 = pd.to_numeric(g1.get("rolling_cal_err", np.nan), errors="coerce")
mid1 = g1["logged_at"].median()

print(f"\n  {'cal_err range':25s} {'n':>6} {'WR':>6} {'BE':>6} {'edge':>7} {'p-val':>10} {'TR edge':>8} {'TE edge':>8}")
for lo, hi in [(-1,-0.20),(-0.20,-0.10),(-0.10,-0.05),(-0.05,0.0),
               (0.0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.30),(0.30,1.0)]:
    mask = (ce1 >= lo) & (ce1 < hi)
    sub  = g1[mask]
    if len(sub) < 10: continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    wins = int(sub["resolved_yes"].sum())
    bt = binomtest(wins, len(sub), be, alternative="greater")
    tr = sub[sub["logged_at"] < mid1]
    te = sub[sub["logged_at"] >= mid1]
    e_tr = tr["resolved_yes"].mean() - tr["p_market"].mean() if len(tr)>=5 else np.nan
    e_te = te["resolved_yes"].mean() - te["p_market"].mean() if len(te)>=5 else np.nan
    flag = " ← RESCUE?" if edge > 0 else ""
    print(f"  ce[{lo:+.2f},{hi:+.2f})  {len(sub):>6}  {wr:.3f}  {be:.3f}  {edge:+.4f}  "
          f"{bt.pvalue:>10.2e}  {e_tr:+.3f}  {e_te:+.3f}{flag}")

# G1 × cal_err × other features — best combo scan
print(f"\n  G1 + ce<-0.10 × additional features:")
sub_ce_neg = g1[ce1 < -0.10].copy()
print(f"  Base (G1 + ce<-0.10): n={len(sub_ce_neg)}  WR={sub_ce_neg['resolved_yes'].mean():.3f}  "
      f"BE={sub_ce_neg['p_market'].mean():.3f}  edge={sub_ce_neg['resolved_yes'].mean()-sub_ce_neg['p_market'].mean():+.3f}")

for cname in ["ema_eq_-1","ema_eq_1","ct_ge_4","ct_ge_-4","pm_ge_0.80","pm_lt_0.50",
              "hvs_r0","hvs_r1","ce_lt_-0.10","pv2_ge_0.50","rsi1h_ge_50","mcd1h_ge_0",
              "dc240_eq_-1","dc240_eq_1","sk4h_lt_30","sk4h_ge_70","fund_eq_1","fund_eq_-1"]:
    if cname not in conditions: continue
    c = conditions[cname]
    m = c.reindex(sub_ce_neg.index, fill_value=False).fillna(False).astype(bool)
    sub = sub_ce_neg[m]
    if len(sub) < 20: continue
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    edge = wr - be
    wins = int(sub["resolved_yes"].sum())
    bt = binomtest(wins, len(sub), be, alternative="greater")
    flag = " ← RESCUE?" if edge > 0 else ""
    print(f"    + {cname:40s} n={len(sub):4}  WR={wr:.3f}  BE={be:.3f}  edge={edge:+.3f}  p={bt.pvalue:.2e}{flag}")

print("\n" + "="*72)
print("FINAL VERDICT")
print("="*72)
print(f"  G1 rescues (Bonf+WF): {len(rescues_g1)}")
print(f"  G4 rescues (Bonf+WF): {len(rescues_g4)}")
if not rescues_g1:
    print("  G1 → CLEAN BLOCK (no rescue survives exhaustive search)")
if rescues_g4:
    print("  G4 → CONDITIONAL BLOCK (apply listed rescue condition)")
elif not rescues_g4:
    print("  G4 → CLEAN BLOCK (no rescue survives exhaustive search)")
print("\nDone.")
