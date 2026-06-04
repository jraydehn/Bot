"""
validate_gate_permutation.py

Three significance tests for BTC gate conditions:

1. STANDARD permutation test (default)
   H0: the gate's improvement is due to chance.
   Fixed gate; shuffles all outcomes N times.

2. MCPT — optimization-bias test (--mcpt)
   H0: picking the best threshold/state from a sweep is due to chance.
   Optimizes on real data AND on each permuted dataset.

3. WALK-FORWARD (--walkforward)  ← most rigorous
   H0: the gate discovered on training data doesn't transfer to OOS data.
   Train set (first --train_frac): discover/confirm gate parameter.
   Test set  (remaining): evaluate on UNSEEN data.
   Permutation: shuffle ONLY test outcomes — train stays real.
   This mirrors walkforward_donch(start_index=train_window): the model learns
   from real history; only the future it has to predict is randomised.

   Combined --walkforward --mcpt: discover threshold on train, evaluate OOS,
   permute only OOS outcomes. The strongest available test.

   Sweeps (--mcpt or --walkforward --mcpt):
     stoch_no    : threshold 40–90 (step 5) → 11 candidates
     hmm_mtf_st3 : all 9 HMM states → 9 candidates

Usage:
  python3 validate_gate_permutation.py                           # standard
  python3 validate_gate_permutation.py --mcpt --gate all         # MCPT
  python3 validate_gate_permutation.py --walkforward --gate all  # walk-forward
  python3 validate_gate_permutation.py --walkforward --mcpt --gate all  # WF-MCPT
"""
import argparse, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
MODELS  = BASE / "models"
RESULTS = BASE / "results"
DATA    = BASE / "data"

FEE       = 0.07   # Kalshi fee rate
FLAT_BET  = 10.0   # flat $ per trade for simulation

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n_perms", type=int, default=500)
parser.add_argument("--method",  choices=["label", "price"], default="label")
parser.add_argument("--gate",    choices=["hmm_mtf_st3", "stoch_no", "bp_1h_no",
                                          "btc_highpm_no", "semi_markov_r1",
                                          "near_itm", "beardrift",
                                          "near_atm_ema", "strong_trend_nearatm",
                                          "all", "all_yes"],
                    default="hmm_mtf_st3")
parser.add_argument("--csv",     choices=["scan_1h", "scan_15m"], default="scan_1h")
parser.add_argument("--mcpt",        action="store_true",
                    help="Run MCPT (sweep parameter, compare best vs permuted best)")
parser.add_argument("--walkforward", action="store_true",
                    help="Walk-forward test: discover on train, evaluate OOS, permute only OOS")
parser.add_argument("--train_frac",  type=float, default=0.75,
                    help="Fraction of data (by time) used as train set (default 0.75)")
args = parser.parse_args()

# ── Permutation function (from get_permutation.py) ────────────────────────────

def get_permutation(ohlc, start_index=0, seed=None):
    np.random.seed(seed)
    n_bars    = len(ohlc)
    perm_n    = n_bars - (start_index + 1)
    perm_idx  = start_index + 1

    log_bars      = np.log(ohlc[["open", "high", "low", "close"]].values.astype(float))
    start_bar     = log_bars[start_index]

    r_o = log_bars[perm_idx:, 0] - log_bars[perm_idx - 1: n_bars - 1, 3]
    r_h = log_bars[perm_idx:, 1] - log_bars[perm_idx:, 0]
    r_l = log_bars[perm_idx:, 2] - log_bars[perm_idx:, 0]
    r_c = log_bars[perm_idx:, 3] - log_bars[perm_idx:, 0]

    idx   = np.arange(perm_n)
    perm1 = np.random.permutation(idx)
    perm2 = np.random.permutation(idx)

    r_h = r_h[perm1]; r_l = r_l[perm1]; r_c = r_c[perm1]
    r_o = r_o[perm2]

    perm_log = np.zeros((n_bars, 4))
    perm_log[:perm_idx] = log_bars[:perm_idx]
    for i in range(perm_idx, n_bars):
        k = i - perm_idx
        perm_log[i, 0] = perm_log[i - 1, 3] + r_o[k]
        perm_log[i, 1] = perm_log[i, 0] + r_h[k]
        perm_log[i, 2] = perm_log[i, 0] + r_l[k]
        perm_log[i, 3] = perm_log[i, 0] + r_c[k]

    perm_prices = np.exp(perm_log)
    return pd.DataFrame(perm_prices, index=ohlc.index,
                        columns=["open", "high", "low", "close"])


# ── Load scan archive ─────────────────────────────────────────────────────────

# Prefer the HMM-enriched parquet (saved by backfill_hmm_mtf_features.py);
# fall back to the original CSV if the parquet doesn't exist yet.
_ARCHIVE_MAP = {
    "scan_1h":  (RESULTS / "btc_scan_archive_hmm.parquet",
                 RESULTS / "btc_scan_archive.csv"),
    "scan_15m": (RESULTS / "btc_scan_archive_15m_hmm.parquet",
                 RESULTS / "btc_scan_archive_15m.csv"),
}
_pq, _csv = _ARCHIVE_MAP[args.csv]
print(f"Loading scan archive ({args.csv}) …")
if _pq.exists():
    sa = pd.read_parquet(_pq)
    print(f"  Using enriched parquet: {_pq.name}")
else:
    sa = pd.read_csv(_csv, low_memory=False)
    print(f"  Using original CSV (run backfill_hmm_mtf_features.py for full features)")
sa = sa[sa["resolved_yes"].notna()].copy()
sa["close_ts"]  = pd.to_datetime(sa["close_ts"],  errors="coerce", utc=True)
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
sa["p_market"]  = pd.to_numeric(sa["p_market"],   errors="coerce")

# Feature columns available in archive for HMM MTF
HMM_FEATS   = ["stoch_k_5m", "stoch_k_15m", "stoch_k_1h", "rsi_1h",
               "bp_1h", "chg_1h", "macd_hist_1h", "adx_1h", "macd_hist_4h", "adx_4h"]
AVAIL_FEATS = [f for f in HMM_FEATS if f in sa.columns]
MISS_FEATS  = [f for f in HMM_FEATS if f not in sa.columns]

for f in AVAIL_FEATS:
    sa[f] = pd.to_numeric(sa[f], errors="coerce")

print(f"  Rows: {len(sa)}  |  HMM features: {len(AVAIL_FEATS)}/10 present, "
      f"missing={MISS_FEATS} (zero-filled)")

# ── Load HMM MTF model ────────────────────────────────────────────────────────

print("Loading HMM MTF model …")
with open(MODELS / "hmm_mtf_momentum_btc15m.pkl", "rb") as fh:
    mtf_pkg     = pickle.load(fh)
mtf_model   = mtf_pkg["model"]
mtf_scaler  = mtf_pkg["scaler"]

col_medians = {f: float(sa[f].median()) if f in sa.columns else 0.0 for f in HMM_FEATS}

def classify_states(df):
    """Return array of HMM MTF state indices (vectorized)."""
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").fillna(col_medians[f]).values
        if f in df.columns else np.full(len(df), col_medians[f])
        for f in HMM_FEATS
    ]).astype(float)
    X = np.nan_to_num(X, nan=0.0)
    return mtf_model.predict(mtf_scaler.transform(X))

print("  Classifying states …", end=" ", flush=True)
sa["hmm_state"] = classify_states(sa)
state_counts = sa["hmm_state"].value_counts().sort_index().to_dict()
print(f"done.  St3 rows: {state_counts.get(3, 0)}/{len(sa)}")

# ── Gate side (determines won_arr definition) ─────────────────────────────────

GATE_SIDE = {
    "hmm_mtf_st3":          "no",
    "stoch_no":             "no",
    "bp_1h_no":             "no",
    "btc_highpm_no":        "no",
    "semi_markov_r1":       "no",
    "near_itm":             "yes",
    "beardrift":            "yes",
    "near_atm_ema":         "yes",
    "strong_trend_nearatm": "yes",
}


# ── Gate masks (vectorized) ───────────────────────────────────────────────────

def make_gate_mask(df, gate_name):
    """Return boolean Series: True = BLOCK this row (NO-bet simulation)."""
    if gate_name == "hmm_mtf_st3":
        is_st3 = df["hmm_state"] == 3
        off    = pd.to_numeric(df.get("offset_pct",   0.10), errors="coerce").fillna(0.10)
        macd   = pd.to_numeric(df.get("macd_hist_1h", 0.0),  errors="coerce").fillna(0.0)
        rescue = (off >= 0.0) & (off < 0.05) & (macd < -50.0)
        return is_st3 & ~rescue

    elif gate_name == "stoch_no":
        sk = pd.to_numeric(df.get("stoch_k_1h", 50.0), errors="coerce").fillna(50.0)
        return sk >= 60.0

    elif gate_name == "bp_1h_no":
        bp = pd.to_numeric(df.get("bp_1h",    0.5), errors="coerce").fillna(0.5)
        pm = pd.to_numeric(df.get("p_market", 0.5), errors="coerce").fillna(0.5)
        return (bp >= 0.55) & (pm >= 0.40)

    elif gate_name == "btc_highpm_no":
        # Block NO when pm>0.70 AND composite_rev>=0
        # Rescue: composite_rev<0 (bearish reversal signal overrides)
        pm  = pd.to_numeric(df.get("p_market",     0.5), errors="coerce").fillna(0.5)
        rev = pd.to_numeric(df.get("composite_rev", 0),  errors="coerce").fillna(0)
        return (pm > 0.70) & (rev >= 0)

    elif gate_name == "semi_markov_r1":
        # Simplified: block when hmm_vol_state==1 (R1, high-vol).
        # Full gate also uses depth + macro regime; this tests the core R1 signal.
        hvs = pd.to_numeric(df.get("hmm_vol_state", 0), errors="coerce").fillna(0)
        return hvs == 1.0

    # ── YES-side gates ────────────────────────────────────────────────────────

    elif gate_name == "near_itm":
        # Block YES when pm>0.50 AND 4h RSI>62 AND 4h MACD hist>80 (overbought exhaustion).
        pm   = pd.to_numeric(df.get("p_market",     0.5), errors="coerce").fillna(0.5)
        rsi4 = pd.to_numeric(df.get("rsi_4h",       50.0), errors="coerce").fillna(50.0)
        mcd4 = pd.to_numeric(df.get("macd_hist_4h", 0.0), errors="coerce").fillna(0.0)
        return (pm > 0.50) & (rsi4 > 62) & (mcd4 > 80)

    elif gate_name == "beardrift":
        # Block YES when EMA bearish + composite_rev<=3 + stoch>=35 (no oversold bounce).
        # Rescue: vpin_score==1 OR ema_stretch_score==1.
        # Arm 2: EMA bearish + rev<=3 + stoch<25 + OTM (offset>0).
        ema  = pd.to_numeric(df.get("ema_stack_bias",   0), errors="coerce").fillna(0)
        rev  = pd.to_numeric(df.get("composite_rev",    0), errors="coerce").fillna(0)
        sk   = pd.to_numeric(df.get("stoch_k",         50), errors="coerce").fillna(50)
        off  = pd.to_numeric(df.get("offset_pct",       0), errors="coerce").fillna(0)
        vpin = pd.to_numeric(df.get("vpin_score",       0), errors="coerce").fillna(0)
        ema_ex = pd.to_numeric(df.get("ema_stretch_score", 0), errors="coerce").fillna(0)

        base   = (ema == -1) & (rev <= 3)
        rescue = (vpin == 1) | (ema_ex == 1)
        arm1   = base & (sk >= 35) & ~rescue
        arm2   = base & (sk <  25) & (off > 0)
        return arm1 | arm2

    elif gate_name == "near_atm_ema":
        # Block YES when pm∈[0.50,0.60) AND ema_stack∈{0,+1} (no mean-reversion setup).
        pm  = pd.to_numeric(df.get("p_market",       0.5), errors="coerce").fillna(0.5)
        ema = pd.to_numeric(df.get("ema_stack_bias",   0), errors="coerce").fillna(0)
        return (pm >= 0.50) & (pm < 0.60) & ema.isin([0, 1])

    elif gate_name == "strong_trend_nearatm":
        # Block YES when pm∈[0.55,0.60) AND composite_trend>=3 (chasing extended move).
        pm    = pd.to_numeric(df.get("p_market",        0.5), errors="coerce").fillna(0.5)
        trend = pd.to_numeric(df.get("composite_trend",   0), errors="coerce").fillna(0)
        return (pm >= 0.55) & (pm < 0.60) & (trend >= 3)

    else:
        raise ValueError(f"Unknown gate: {gate_name}")


# ── Vectorized P&L ────────────────────────────────────────────────────────────

def compute_pnl_vec(pm_arr, won_arr, block_mask=None):
    """Flat NO-bet P&L: sum over unblocked rows.

    pm_arr   : 1-D float array of p_market values
    won_arr  : 1-D bool array (True = NO wins = resolved_yes==0)
    block_mask: 1-D bool array (True = skip this row); None = trade everything
    Returns (total_pnl, n_blocked).
    """
    if block_mask is not None:
        keep = ~block_mask
        n_blocked = int(block_mask.sum())
    else:
        keep = np.ones(len(pm_arr), dtype=bool)
        n_blocked = 0
    pm  = pm_arr[keep]
    won = won_arr[keep]
    pnl = np.where(won, (1 - pm) * (1 - FEE), -pm * (1 - FEE)) * FLAT_BET
    return float(pnl.sum()), n_blocked


# ── Method A: label shuffle ────────────────────────────────────────────────────

def shuffle_labels_within_pm_bins(df, seed=None):
    """Shuffle resolved_yes within p_market bins to preserve calibration."""
    rng  = np.random.default_rng(seed)
    out  = df.copy()
    bins = np.arange(0.0, 1.05, 0.10)
    pm   = pd.to_numeric(out["p_market"], errors="coerce")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pm >= lo) & (pm < hi)
        idx  = out.index[mask]
        if len(idx) > 1:
            vals = out.loc[idx, "resolved_yes"].values.copy()
            rng.shuffle(vals)
            out.loc[idx, "resolved_yes"] = vals
    return out


# ── Method B: price path permutation ─────────────────────────────────────────

def load_1m_btc(date_start, date_end):
    pq_path = sorted(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"))[-1]
    df = pd.read_parquet(pq_path, columns=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df.loc[date_start:date_end].copy()


def rederive_outcomes(df_sa, perm_1m):
    """Re-derive resolved_yes for each contract from the permuted 1m close series."""
    out = df_sa.copy()
    perm_close = perm_1m["close"].rename("perm_close")
    # Merge-asof: for each close_ts find nearest 1m bar (within 2 minutes)
    sa_ts = out["close_ts"].sort_values()
    idx   = pd.Index(perm_1m.index)
    new_resolved = []
    for ts in out["close_ts"]:
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0 or pos >= len(perm_1m):
            new_resolved.append(np.nan)
        else:
            perm_price = float(perm_1m["close"].iloc[pos])
            strike     = float(out.loc[out["close_ts"] == ts, "strike"].iloc[0])
            new_resolved.append(1 if perm_price >= strike else 0)
    out["resolved_yes"] = new_resolved
    return out.dropna(subset=["resolved_yes"])


# ── MCPT sweep helpers ────────────────────────────────────────────────────────

# Parameter grids
STOCH_THRESHOLDS = np.arange(40, 91, 5)          # 11 values: 40 45 … 90
HMM_STATES       = list(range(9))                # states 0–8

def _best_stoch_delta(pm_arr, won_arr, sk_arr):
    """Sweep stoch thresholds; return (best_delta, best_threshold)."""
    pnl_base, _ = compute_pnl_vec(pm_arr, won_arr)
    best_delta, best_thresh = -np.inf, -1
    for thresh in STOCH_THRESHOLDS:
        block = sk_arr >= thresh
        delta = compute_pnl_vec(pm_arr, won_arr, block)[0] - pnl_base
        if delta > best_delta:
            best_delta, best_thresh = delta, int(thresh)
    return best_delta, best_thresh


def _best_hmm_state_delta(pm_arr, won_arr, state_arr):
    """Sweep HMM states; return (best_delta, best_state)."""
    pnl_base, _ = compute_pnl_vec(pm_arr, won_arr)
    best_delta, best_state = -np.inf, -1
    for s in HMM_STATES:
        block = state_arr == s
        delta = compute_pnl_vec(pm_arr, won_arr, block)[0] - pnl_base
        if delta > best_delta:
            best_delta, best_state = delta, s
    return best_delta, best_state


def run_mcpt(gate_name, pm_arr, won_arr, state_arr, sk_arr,
             pm_bins, n_perms, rng, method, df_1m=None):
    """Full MCPT for a gate: optimise on real data, compare to permuted optima."""

    # Choose sweep function
    if gate_name == "stoch_no":
        sweep_fn  = lambda p, w: _best_stoch_delta(p, w, sk_arr)
        param_lbl = "threshold"
    elif gate_name == "hmm_mtf_st3":
        sweep_fn  = lambda p, w: _best_hmm_state_delta(p, w, state_arr)
        param_lbl = "state"
    elif gate_name in ("bp_1h_no", "btc_highpm_no", "semi_markov_r1",
                       "near_itm", "beardrift", "near_atm_ema", "strong_trend_nearatm"):
        # Fixed-condition gates — no parameter to sweep.
        # MCPT reduces to standard Perm for these; skip MCPT and use standard test instead.
        print(f"\n  MCPT: {gate_name} — fixed condition, no parameter to sweep. "
              f"Use standard Perm test (already run above).")
        return None, None, None
    else:
        raise ValueError(gate_name)

    # Real optimum
    real_best_delta, real_best_param = sweep_fn(pm_arr, won_arr)

    print(f"\n  MCPT: {gate_name}  ({len(STOCH_THRESHOLDS) if gate_name=='stoch_no' else len(HMM_STATES)} candidates)")
    print(f"    Real best {param_lbl}: {real_best_param}  →  Δ = ${real_best_delta:+,.2f}")
    print(f"    Running {n_perms} permutations …", end=" ", flush=True)

    perm_best_deltas = []
    for i in range(n_perms):
        if method == "label":
            wp = won_arr.copy()
            for b in range(11):
                idx = np.where(pm_bins == b)[0]
                if len(idx) > 1:
                    vals = wp[idx]; rng.shuffle(vals); wp[idx] = vals
        else:
            # price-path permutation reuses df_1m (caller must supply)
            perm_1m = get_permutation(df_1m, seed=i)
            df_tmp  = rederive_outcomes(sa, perm_1m)
            if len(df_tmp) < 10:
                continue
            wp = (pd.to_numeric(df_tmp["resolved_yes"],
                                errors="coerce").fillna(-1) == 0).values

        perm_best_deltas.append(sweep_fn(pm_arr, wp)[0])
        if (i + 1) % 100 == 0:
            print(f"{i+1}", end=" ", flush=True)
    print("done.")

    pd_arr  = np.array(perm_best_deltas)
    p_val   = float((pd_arr >= real_best_delta).mean())
    pct5    = float(np.percentile(pd_arr, 5))
    pct50   = float(np.percentile(pd_arr, 50))
    pct95   = float(np.percentile(pd_arr, 95))

    sig = p_val < 0.05
    print(f"\n    Null p5/p50/p95 : ${pct5:+,.2f} / ${pct50:+,.2f} / ${pct95:+,.2f}")
    print(f"    p-value         : {p_val:.3f}  ({'SIGNIFICANT ✓' if sig else 'not significant'})")
    print(f"    Rank            : {int((pd_arr < real_best_delta).sum())}/{len(pd_arr)}")
    if sig:
        print(f"    Interpretation  : Best {param_lbl}={real_best_param} survives "
              f"parameter search — genuine signal, not optimization bias.")
    else:
        print(f"    Interpretation  : Best {param_lbl}={real_best_param} does NOT "
              f"beat permuted optima — result is optimization bias over "
              f"{len(STOCH_THRESHOLDS) if gate_name=='stoch_no' else len(HMM_STATES)} candidates.")
    return real_best_delta, real_best_param, p_val


# ── Walk-forward helpers ──────────────────────────────────────────────────────

def _shuffle_test_only(won_arr, test_mask, pm_bins, rng):
    """Shuffle resolved_yes within p_market bins for test rows only.
    Train rows (test_mask==False) are left untouched — identical to
    get_permutation(start_index=train_window): only the OOS period is randomised.
    """
    wp = won_arr.copy()
    for b in range(11):
        idx = np.where((pm_bins == b) & test_mask)[0]
        if len(idx) > 1:
            vals = wp[idx]; rng.shuffle(vals); wp[idx] = vals
    return wp


def run_walkforward(gate_name, sa_sorted, pm_arr, won_arr, state_arr, sk_arr,
                    pm_bins, train_frac, n_perms, rng, do_mcpt=False):
    """Walk-forward significance test.

    Train set : first train_frac rows (by logged_at) — used to discover gate param.
    Test set  : remaining rows — evaluated OOS.
    Permutation: only test outcomes are shuffled; train stays real.

    do_mcpt=False : use fixed pre-specified gate (threshold from make_gate_mask)
    do_mcpt=True  : discover best threshold on train, apply to test (WF-MCPT)
    """
    n_total   = len(sa_sorted)
    split_idx = int(n_total * train_frac)
    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[split_idx:] = True

    train_idx = np.where(~test_mask)[0]
    test_idx  = np.where( test_mask)[0]

    split_ts = sa_sorted["logged_at"].iloc[split_idx]
    print(f"\n  Split: train {split_idx:,} rows → {split_ts.date()}  |  "
          f"test {len(test_idx):,} rows")

    pm_test  = pm_arr[test_idx]
    won_test = won_arr[test_idx]

    _FIXED_GATES = {"bp_1h_no", "btc_highpm_no", "semi_markov_r1",
                    "near_itm", "beardrift", "near_atm_ema", "strong_trend_nearatm"}

    if do_mcpt and gate_name not in _FIXED_GATES:
        # Discover best gate parameter on TRAIN set (sweep gates only)
        pm_tr  = pm_arr[train_idx]
        won_tr = won_arr[train_idx]
        if gate_name == "stoch_no":
            sweep_fn_tr = lambda p, w: _best_stoch_delta(
                p, w, sk_arr[train_idx])
            sweep_fn_te = lambda p, w, param: (
                compute_pnl_vec(p, w, sk_arr[test_idx] >= param)[0]
                - compute_pnl_vec(p, w)[0])
            param_lbl = "threshold"
        else:
            sweep_fn_tr = lambda p, w: _best_hmm_state_delta(
                p, w, state_arr[train_idx])
            sweep_fn_te = lambda p, w, param: (
                compute_pnl_vec(p, w, state_arr[test_idx] == param)[0]
                - compute_pnl_vec(p, w)[0])
            param_lbl = "state"

        best_tr_delta, best_param = sweep_fn_tr(pm_tr, won_tr)
        real_oos_delta = sweep_fn_te(pm_test, won_test, best_param)

        print(f"  Train best {param_lbl}: {best_param}  train-Δ=${best_tr_delta:+,.2f}")
        print(f"  OOS Δ (real)        : ${real_oos_delta:+,.2f}")

        def perm_delta(wp):
            return sweep_fn_te(pm_test, wp[test_idx], best_param)

    else:
        # Fixed gate: apply pre-specified gate mask to test set
        block_te = make_gate_mask(sa_sorted.iloc[test_idx].reset_index(drop=True),
                                  gate_name).values
        base_oos, _ = compute_pnl_vec(pm_test, won_test)
        gate_oos, n_blk = compute_pnl_vec(pm_test, won_test, block_te)
        real_oos_delta = gate_oos - base_oos

        wr_blk = won_test[block_te].mean() if block_te.sum() else float("nan")
        bk_blk = pm_test[block_te].mean()  if block_te.sum() else float("nan")
        print(f"  OOS ungated : ${base_oos:+,.2f}  ({len(test_idx):,} bets)")
        print(f"  OOS gated   : ${gate_oos:+,.2f}  ({len(test_idx)-n_blk:,} bets, {n_blk} blocked)")
        print(f"  OOS Δ (real): ${real_oos_delta:+,.2f}")
        if block_te.sum():
            print(f"  Blocked OOS : WR={wr_blk:.1%}  bkev={bk_blk:.1%}")

        def perm_delta(wp):
            g, _ = compute_pnl_vec(pm_test, wp[test_idx], block_te)
            u, _ = compute_pnl_vec(pm_test, wp[test_idx])
            return g - u

    # Permute only test rows
    print(f"  Running {n_perms} permutations (OOS only) …", end=" ", flush=True)
    perm_oos = []
    for i in range(n_perms):
        wp = _shuffle_test_only(won_arr, test_mask, pm_bins, rng)
        perm_oos.append(perm_delta(wp))
        if (i + 1) % 100 == 0:
            print(f"{i+1}", end=" ", flush=True)
    print("done.")

    pd_arr = np.array(perm_oos)
    p_val  = float((pd_arr >= real_oos_delta).mean())
    pct5   = float(np.percentile(pd_arr, 5))
    pct50  = float(np.percentile(pd_arr, 50))
    pct95  = float(np.percentile(pd_arr, 95))
    sig    = p_val < 0.05

    print(f"\n  OOS null p5/p50/p95 : ${pct5:+,.2f} / ${pct50:+,.2f} / ${pct95:+,.2f}")
    print(f"  p-value             : {p_val:.3f}  ({'SIGNIFICANT ✓' if sig else 'not significant'})")
    print(f"  Rank                : {int((pd_arr < real_oos_delta).sum())}/{len(pd_arr)}")
    if sig:
        print(f"  → Gate{'(discovered on train) ' if do_mcpt else ' '}transfers to OOS — "
              f"genuine predictive signal.")
    else:
        print(f"  → Gate does NOT beat permuted OOS — "
              f"{'overfit to train period.' if do_mcpt else 'no OOS edge.'}")
    return real_oos_delta, p_val


# ── Run tests ─────────────────────────────────────────────────────────────────

_ALL_NO_GATES  = ["hmm_mtf_st3", "stoch_no", "bp_1h_no", "btc_highpm_no", "semi_markov_r1"]
_ALL_YES_GATES = ["near_itm", "beardrift", "near_atm_ema", "strong_trend_nearatm"]
_ALL_GATES     = _ALL_NO_GATES + _ALL_YES_GATES

if args.gate == "all":
    gate_names = _ALL_GATES
elif args.gate == "all_yes":
    gate_names = _ALL_YES_GATES
else:
    gate_names = [args.gate]

print(f"\n{'='*70}")
print(f"  PERMUTATION TEST   method={args.method}   n_perms={args.n_perms}")
print(f"{'='*70}\n")

if args.method == "price":
    t_start = sa["logged_at"].min() - pd.Timedelta(hours=1)
    t_end   = sa["close_ts"].max()   + pd.Timedelta(hours=1)
    print(f"Loading 1m BTC data {t_start.date()} → {t_end.date()} …", end=" ", flush=True)
    df_1m = load_1m_btc(t_start, t_end)
    print(f"{len(df_1m)} bars.")

for gate_name in gate_names:
    print(f"\n── Gate: {gate_name} ──────────────────────────────────────────────")

    # Precompute arrays and gate mask on real data
    side      = GATE_SIDE.get(gate_name, "no")
    pm_real   = sa["p_market"].astype(float).values
    ry        = pd.to_numeric(sa["resolved_yes"], errors="coerce").fillna(-1)
    won_real  = (ry == (1 if side == "yes" else 0)).values
    block_real = make_gate_mask(sa, gate_name).values

    pnl_all,  _       = compute_pnl_vec(pm_real, won_real)
    pnl_gate, n_blk   = compute_pnl_vec(pm_real, won_real, block_real)
    real_delta = pnl_gate - pnl_all

    # Blocked-trade stats
    blk_idx = block_real
    blk_pm  = pm_real[blk_idx]
    blk_won = won_real[blk_idx]
    blk_pnl, _ = compute_pnl_vec(blk_pm, blk_won)
    blk_wr   = float(blk_won.mean()) if blk_idx.sum() else float("nan")
    # bkev: for NO bets = 1-pm; for YES bets = pm
    blk_bkev = float(blk_pm.mean() if side == "yes" else 1 - blk_pm.mean()) \
               if blk_idx.sum() else float("nan")

    print(f"  Ungated  : ${pnl_all:+.2f}  ({len(sa)} bets)")
    print(f"  Gated    : ${pnl_gate:+.2f}  ({len(sa)-n_blk} bets, {n_blk} blocked)")
    print(f"  Δ (real) : ${real_delta:+.2f}")
    if blk_idx.sum():
        print(f"  Blocked  : n={blk_idx.sum()}  WR={blk_wr:.1%}  bkev={blk_bkev:.1%}  "
              f"would_pnl=${blk_pnl:+.2f}")

    # Permutations — only shuffle resolved_yes, reuse precomputed pm + mask
    print(f"\n  Running {args.n_perms} permutations …", end=" ", flush=True)
    perm_deltas = []
    rng = np.random.default_rng(0)
    pm_bins = np.floor(pm_real * 10).astype(int)  # 0..10 bins

    for i in range(args.n_perms):
        if args.method == "label":
            won_perm = won_real.copy()
            for b in range(11):
                idx = np.where(pm_bins == b)[0]
                if len(idx) > 1:
                    vals = won_perm[idx]
                    rng.shuffle(vals)
                    won_perm[idx] = vals   # must write back — fancy indexing returns a copy
        else:
            perm_1m  = get_permutation(df_1m, seed=i)
            df_perm  = rederive_outcomes(sa, perm_1m)
            if len(df_perm) < 10:
                continue
            won_perm = (pd.to_numeric(df_perm["resolved_yes"],
                                      errors="coerce").fillna(-1) == 0).values

        p_all,  _ = compute_pnl_vec(pm_real, won_perm)
        p_gate, _ = compute_pnl_vec(pm_real, won_perm, block_real)
        perm_deltas.append(p_gate - p_all)

        if (i + 1) % 50 == 0:
            print(f"{i+1}", end=" ", flush=True)
    print("done.")

    perm_deltas = np.array(perm_deltas)
    p_val   = float((perm_deltas >= real_delta).mean())
    pct95   = float(np.percentile(perm_deltas, 95))
    pct50   = float(np.percentile(perm_deltas, 50))
    pct05   = float(np.percentile(perm_deltas, 5))

    print(f"\n  Results:")
    print(f"    Real Δ         : ${real_delta:+.2f}")
    print(f"    Null p5/p50/p95: ${pct05:+.2f} / ${pct50:+.2f} / ${pct95:+.2f}")
    print(f"    p-value        : {p_val:.3f}  ({'SIGNIFICANT ✓' if p_val < 0.05 else 'not significant'})")
    print(f"    Rank           : {int((perm_deltas < real_delta).sum())}/{len(perm_deltas)}")

    if real_delta > pct95:
        print(f"    Interpretation : Gate improvement is in top 5% of null — "
              f"genuine temporal signal (not overfit).")
    elif real_delta > pct50:
        print(f"    Interpretation : Gate improvement beats median null but not "
              f"95th percentile — weak / marginal signal.")
    else:
        print(f"    Interpretation : Gate improvement does NOT beat median null — "
              f"likely overfit to data.")

print(f"\n{'='*70}")

# ── MCPT section ──────────────────────────────────────────────────────────────

if args.mcpt:
    print(f"\n{'='*70}")
    print(f"  MCPT (optimization-bias test)   n_perms={args.n_perms}")
    print(f"{'='*70}")

    _pm_mcpt    = sa["p_market"].astype(float).values
    _ry_mcpt    = pd.to_numeric(sa["resolved_yes"], errors="coerce").fillna(-1)
    _state_mcpt = sa["hmm_state"].values
    _sk_mcpt    = pd.to_numeric(sa.get("stoch_k_1h", 50.0), errors="coerce").fillna(50.0).values
    _bins_mcpt  = np.floor(_pm_mcpt * 10).astype(int)
    _rng_mcpt   = np.random.default_rng(42)

    _df1m_mcpt = None
    if args.method == "price":
        import os as _os
        _pq1m = max(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"), key=_os.path.getmtime)
        _df1m_raw = pd.read_parquet(_pq1m, columns=["open","high","low","close"])
        _df1m_raw.index = pd.to_datetime(_df1m_raw.index, utc=True)
        t0 = sa["logged_at"].min() - pd.Timedelta(hours=1)
        t1 = sa["close_ts"].max()  + pd.Timedelta(hours=1)
        _df1m_mcpt = _df1m_raw[(_df1m_raw.index >= t0) & (_df1m_raw.index <= t1)]

    for gate_name in gate_names:
        print(f"\n── Gate: {gate_name}")
        _side_g   = GATE_SIDE.get(gate_name, "no")
        _won_mcpt = (_ry_mcpt == (1 if _side_g == "yes" else 0)).values
        run_mcpt(
            gate_name, _pm_mcpt, _won_mcpt, _state_mcpt, _sk_mcpt,
            _bins_mcpt, args.n_perms, _rng_mcpt,
            method=args.method, df_1m=_df1m_mcpt,
        )

    print(f"\n{'='*70}")

# ── Walk-forward section ──────────────────────────────────────────────────────

if args.walkforward:
    print(f"\n{'='*70}")
    mode_lbl = "WALK-FORWARD MCPT" if args.mcpt else "WALK-FORWARD"
    print(f"  {mode_lbl}   train={args.train_frac:.0%}   n_perms={args.n_perms}")
    if args.mcpt:
        print(f"  Discover gate on train → evaluate OOS → permute only OOS outcomes.")
    else:
        print(f"  Fixed gate on OOS data → permute only OOS outcomes.")
    print(f"{'='*70}")

    # Sort archive chronologically — walk-forward requires time ordering
    _sa_wf = sa.sort_values("logged_at").reset_index(drop=True)

    _pm_wf    = _sa_wf["p_market"].astype(float).values
    _ry_wf    = pd.to_numeric(_sa_wf["resolved_yes"], errors="coerce").fillna(-1)
    _state_wf = _sa_wf["hmm_state"].values
    _sk_wf    = pd.to_numeric(_sa_wf.get("stoch_k_1h", 50.0), errors="coerce").fillna(50.0).values
    _bins_wf  = np.floor(_pm_wf * 10).astype(int)
    _rng_wf   = np.random.default_rng(99)

    for gate_name in gate_names:
        print(f"\n── Gate: {gate_name}")
        _side_g  = GATE_SIDE.get(gate_name, "no")
        _won_wf  = (_ry_wf == (1 if _side_g == "yes" else 0)).values
        run_walkforward(
            gate_name, _sa_wf, _pm_wf, _won_wf, _state_wf, _sk_wf,
            _bins_wf, args.train_frac, args.n_perms, _rng_wf,
            do_mcpt=args.mcpt,
        )

    print(f"\n{'='*70}")
