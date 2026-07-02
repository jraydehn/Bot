"""
simulate_new_trend_signals.py

Simulates adding top 4h candidate signals to the trend score and measures
the P&L impact vs the current composite model on the btc_scan_archive.

Methodology:
  1. Load full 1h/4h OHLCV history (1970-01-01 → present)
  2. Compute new 4h vote signals at every 1h bar:
       - stoch_kd_diff_4h  (K-D cross, IC=+0.2554)
       - ema_dist_z_20_4h  (EMA z-score, IC=+0.1414)
       - cci_20_4h         (CCI, IC=+0.0960)
       - di_diff_4h        (DI+−DI−, IC=+0.0762)
     Also test rev candidate:
       - bb_diff_4h_vs_1h  (inverted: IC=−0.2038 → add to rev as +(bb_4h−bb_1h))
  3. Build new calibration table: (new_trend_bin, rev_bin) → next_up
     Train period: 2025-01-01 → 2026-01-01 (OOS from scan archive which starts May 2026)
  4. Join new signal values to btc_scan_archive.csv by 1h bar timestamp
  5. Simulate flat $1000 bankroll, simplified gate (saturation + R:R floor):
       - YES: if new_p_model_yes > pm + MIN_EDGE and p_model > 0.10 → bet YES
       - NO:  if new_p_model_no  > (1-pm) + MIN_EDGE and p_no > 0.10 → bet NO
     Same simulation applied to OLD model for apples-to-apples comparison.
  6. Report: wins/losses blocked or gained, WR, net P&L delta.

Flat bankroll, non-compounding. Same PnL formula as other simulate_* scripts.
"""

import glob
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
SYM      = "BTCUSDT"

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")   # OOS from scan archive (May 2026+)

BANKROLL    = 1000.0
MIN_EDGE    = 0.04    # minimum edge to bet (same threshold as runner)
MAX_KELLY   = 0.15    # Kelly cap (15% of bankroll)
K_DRIFT_YES = 1.40    # BTC YES drift multiplier
K_DRIFT_NO  = 0.30    # BTC NO drift multiplier
SMOOTH_K    = 30      # calibration pseudo-count

TREND_CLIP  = 6       # existing clip range for old trend
NEW_CLIP    = 8       # extended clip for trend + new votes (4 new signals, up to +2 each)
REV_CLIP    = 11      # existing rev clip

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores, _rsi, _stoch_k, _atr, _bb_pct, _keltner_pct,
    lookup_p_up, BASELINE_UP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_ohlcv():
    def pick_latest(pattern):
        files = sorted(glob.glob(str(DATA_DIR / pattern)))
        if not files:
            raise FileNotFoundError(f"No file matching {pattern}")
        return files[-1]

    def load(p):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = df.columns.str.lower()
        return df.sort_index()

    o1h = load(pick_latest(f"binanceus_{SYM}_1h_1970-01-01_*.parquet"))
    o4h = load(pick_latest(f"binanceus_{SYM}_4h_1970-01-01_*.parquet"))
    # Also load 15m and 1m for compute_scores baseline (may be slower)
    o15m = load(sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_15m_2024-01-01_*.parquet")))[-1])
    o1m  = load(sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_1m_2024-01-01_*.parquet")))[-1])
    print(f"  1h: {len(o1h):,} rows  {o1h.index[-1].date()}")
    print(f"  4h: {len(o4h):,} rows  {o4h.index[-1].date()}")
    return o1h, o4h, o15m, o1m


# ─────────────────────────────────────────────────────────────────────────────
# New signal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _adx_di(h, l, c, p=14):
    hp   = h.shift(1);  lp = l.shift(1)
    dm_p = np.where((h-hp) > (lp-l), np.maximum(h-hp, 0), 0)
    dm_m = np.where((lp-l) > (h-hp), np.maximum(lp-l, 0), 0)
    atr  = _atr(h, l, c, p)
    di_p = pd.Series(dm_p, index=c.index).ewm(com=p-1, adjust=False).mean() / atr * 100
    di_m = pd.Series(dm_m, index=c.index).ewm(com=p-1, adjust=False).mean() / atr * 100
    dx   = (di_p - di_m).abs() / (di_p + di_m).replace(0, float("nan")) * 100
    adx  = dx.ewm(com=p-1, adjust=False).mean()
    return adx, di_p, di_m


def _cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, float("nan")))


def _ema_dist_z(c, span=20, roll=48):
    ema   = c.ewm(span=span, adjust=False).mean()
    dist  = (c - ema) / ema
    mu    = dist.rolling(roll).mean()
    sigma = dist.rolling(roll).std()
    return (dist - mu) / sigma.replace(0, float("nan"))


def compute_new_signals(o1h, o4h):
    """Compute 4h candidate signals, resampled to 1h index."""
    c1h = o1h["close"].astype(float)
    h1h = o1h["high"].astype(float)
    l1h = o1h["low"].astype(float)
    c4h = o4h["close"].astype(float)
    h4h = o4h["high"].astype(float)
    l4h = o4h["low"].astype(float)
    ts  = c1h.index

    def to_1h(s):
        return s.resample("1h", origin="start_day").last().reindex(ts, method="ffill")

    # ── stoch_kd_diff_4h ─────────────────────────────────────────────────────
    stk4h   = _stoch_k(h4h, l4h, c4h, 14)
    stk_d4h = stk4h.ewm(span=3, adjust=False).mean()
    kd_diff = to_1h(stk4h - stk_d4h)

    # ── ema_dist_z_20_4h ─────────────────────────────────────────────────────
    ema_z_4h = to_1h(_ema_dist_z(c4h, span=20, roll=48))

    # ── cci_20_4h ─────────────────────────────────────────────────────────────
    cci_4h = to_1h(_cci(h4h, l4h, c4h, 20))

    # ── di_diff_4h (DI+ − DI−) ───────────────────────────────────────────────
    _, di_p4h, di_m4h = _adx_di(h4h, l4h, c4h, 14)
    di_diff_4h = to_1h(di_p4h - di_m4h)

    # ── bb_diff (rev candidate): bb_4h − bb_1h → mean-reversion when negative ─
    bb4h_pct = _bb_pct(h4h, l4h, c4h, 20)
    bb1h_pct = _bb_pct(h1h, l1h, c1h, 20)
    bb_diff  = to_1h(bb4h_pct) - bb1h_pct    # positive = 4h > 1h (bearish for rev)

    return pd.DataFrame({
        "kd_diff_4h":  kd_diff,
        "ema_z_4h":    ema_z_4h,
        "cci_4h":      cci_4h,
        "di_diff_4h":  di_diff_4h,
        "bb_diff":     bb_diff,
    })


def signals_to_votes(sig_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw signal values to ±1/±2 votes using percentile-based thresholds.

    Thresholds set at ~10th/25th and 75th/90th percentiles of the train period
    so votes are roughly symmetric and well-populated.
    """
    votes = pd.DataFrame(index=sig_df.index)
    mask  = (sig_df.index >= TRAIN_START) & (sig_df.index < TRAIN_END)

    def thresh(col, pcts=(10, 25, 75, 90)):
        s = sig_df[col][mask].dropna()
        return [np.percentile(s, p) for p in pcts]

    # stoch_kd_diff_4h  — positive = K crossed above D (bullish momentum)
    t = thresh("kd_diff_4h")
    kd = sig_df["kd_diff_4h"]
    v = pd.Series(0, index=sig_df.index, dtype=float)
    v[kd >= t[3]] =  2
    v[(kd >= t[2]) & (kd < t[3])] =  1
    v[(kd <= t[1]) & (kd > t[0])] = -1
    v[kd <= t[0]]  = -2
    votes["v_kd4h"] = v
    print(f"  stoch_kd_diff_4h thresholds: {[round(x,1) for x in t]}")

    # ema_dist_z_20_4h  — positive z = above EMA (trend-following → bullish)
    t = thresh("ema_z_4h")
    ez = sig_df["ema_z_4h"]
    v = pd.Series(0, index=sig_df.index, dtype=float)
    v[ez >= t[3]] =  2
    v[(ez >= t[2]) & (ez < t[3])] =  1
    v[(ez <= t[1]) & (ez > t[0])] = -1
    v[ez <= t[0]]  = -2
    votes["v_emaz4h"] = v
    print(f"  ema_dist_z_20_4h thresholds: {[round(x,2) for x in t]}")

    # cci_20_4h  — positive CCI = bullish (trend-following at 4h)
    t = thresh("cci_4h")
    cc = sig_df["cci_4h"]
    v = pd.Series(0, index=sig_df.index, dtype=float)
    v[cc >= t[3]] =  2
    v[(cc >= t[2]) & (cc < t[3])] =  1
    v[(cc <= t[1]) & (cc > t[0])] = -1
    v[cc <= t[0]]  = -2
    votes["v_cci4h"] = v
    print(f"  cci_20_4h thresholds: {[round(x,1) for x in t]}")

    # di_diff_4h (DI+ − DI−)  — positive = DI+ > DI- (bullish)
    t = thresh("di_diff_4h")
    dd = sig_df["di_diff_4h"]
    v = pd.Series(0, index=sig_df.index, dtype=float)
    v[dd >= t[3]] =  2
    v[(dd >= t[2]) & (dd < t[3])] =  1
    v[(dd <= t[1]) & (dd > t[0])] = -1
    v[dd <= t[0]]  = -2
    votes["v_di4h"] = v
    print(f"  di_diff_4h thresholds: {[round(x,1) for x in t]}")

    # bb_diff (bb_4h − bb_1h) → rev vote: negative = 1h is extended vs 4h → expect down
    # Add to rev with sign so: when 1h BB > 4h BB → bearish mean-reversion = negative rev vote
    # i.e., rev_add = -(bb_4h - bb_1h) if we think IC=-0.2038 means 1h>4h→expect down
    # bb_diff = bb_4h - bb_1h; positive = 4h ABOVE 1h; we want: high bb_diff → bullish
    # So direct vote on bb_diff for rev: positive bb_diff → +1 rev (mean-reversion bullish)
    t = thresh("bb_diff")
    bd = sig_df["bb_diff"]
    v = pd.Series(0, index=sig_df.index, dtype=float)
    v[bd >= t[3]] =  2
    v[(bd >= t[2]) & (bd < t[3])] =  1
    v[(bd <= t[1]) & (bd > t[0])] = -1
    v[bd <= t[0]]  = -2
    votes["v_bb_diff"] = v
    print(f"  bb_diff_4h_vs_1h thresholds: {[round(x,3) for x in t]}")

    return votes


# ─────────────────────────────────────────────────────────────────────────────
# Calibration table
# ─────────────────────────────────────────────────────────────────────────────
def build_calibration(trend_ser, rev_ser, next_up, mask, clip_t, clip_r,
                      label=""):
    """Build (trend_bin, rev_bin) → p_up table on masked rows."""
    df = pd.DataFrame({
        "trend": trend_ser[mask],
        "rev":   rev_ser[mask],
        "up":    next_up[mask],
    }).dropna()
    df["tb"] = df["trend"].clip(-clip_t, clip_t).astype(int)
    df["rb"] = df["rev"].clip(-clip_r, clip_r).astype(int)

    baseline = df["up"].mean()
    table = {}
    for tb in range(-clip_t, clip_t + 1):
        for rb in range(-clip_r, clip_r + 1):
            cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
            n    = len(cell)
            wr   = cell["up"].mean() if n >= 10 else float("nan")
            if np.isnan(wr):
                p = baseline
            else:
                p = (n * wr + SMOOTH_K * baseline) / (n + SMOOTH_K)
            table[f"{tb},{rb}"] = float(p)
    table["__baseline__"] = float(baseline)
    if label:
        print(f"  [{label}] baseline={baseline:.4f}  cells={len(df):,}")
    return table, baseline


def lookup_new(table, baseline, trend_val, rev_val, clip_t, clip_r):
    tb  = int(np.clip(round(trend_val), -clip_t, clip_t))
    rb  = int(np.clip(round(rev_val),   -clip_r, clip_r))
    return table.get(f"{tb},{rb}", baseline)


# ─────────────────────────────────────────────────────────────────────────────
# P-model computation
# ─────────────────────────────────────────────────────────────────────────────
def p_model_yes(p_up, spot, strike, sigma_tau, k_drift=K_DRIFT_YES):
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift  = norm.ppf(max(0.01, min(0.99, p_up))) * k_drift
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def p_model_no(p_up, spot, strike, sigma_tau, k_drift=K_DRIFT_NO):
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift  = norm.ppf(max(0.01, min(0.99, p_up))) * k_drift
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.01, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate(df: pd.DataFrame, table, baseline, clip_t, clip_r,
             trend_col="composite_trend", rev_col="composite_rev",
             label="Baseline"):
    """
    Simplified simulation on scan archive rows.

    For each resolved row:
      - Determine YES or NO candidate (can evaluate both)
      - For YES: pm = p_market; edge = p_yes_model - pm
      - For NO:  pm = 1-p_market; edge = p_no_model - pm
      - Kelly fraction = edge / (1 - pm) for YES; edge / pm for NO
      - Bet only if edge > MIN_EDGE and p_model > 0.10 and kf > 0
      - PnL: YES win → +bet*(1/pm-1); YES loss → -bet
             NO  win → +bet*(1/pm_cost-1); NO loss → -bet
    """
    results = []
    for _, row in df.iterrows():
        trend = row[trend_col]
        rev   = row[rev_col]
        spot  = float(row["spot"])
        strike = float(row["strike"])
        pm    = float(row["p_market"])
        vol   = float(row["vol_eff"])
        tau   = float(row["tau_minutes"])
        outcome = row["resolved_yes"]   # True/False (1/0)

        if pd.isna(trend) or pd.isna(rev) or tau <= 0 or vol <= 0:
            continue

        sigma_tau = vol * math.sqrt(tau)
        p_up = lookup_new(table, baseline, trend, rev, clip_t, clip_r)

        pm_yes = p_model_yes(p_up, spot, strike, sigma_tau)
        pm_no  = p_model_no(p_up, spot, strike, sigma_tau)

        # YES evaluation
        edge_yes = pm_yes - pm
        kf_yes   = edge_yes / (1 - pm) if pm < 1 else 0
        # NO evaluation
        cost_no  = 1 - pm
        edge_no  = pm_no - cost_no
        kf_no    = edge_no / pm if pm > 0 else 0

        bet_yes = min(max(kf_yes, 0), MAX_KELLY) * BANKROLL
        bet_no  = min(max(kf_no, 0), MAX_KELLY) * BANKROLL

        # Gate: saturation (p_model < 0.10) and minimum edge
        if edge_yes >= MIN_EDGE and pm_yes >= 0.10 and bet_yes > 0:
            pnl = bet_yes * (1/pm - 1) if outcome else -bet_yes
            results.append({"side": "YES", "pnl": pnl, "win": outcome,
                            "bet": bet_yes, "edge": edge_yes, "pm": pm})
        if edge_no >= MIN_EDGE and pm_no >= 0.10 and bet_no > 0:
            pnl = bet_no * (1/cost_no - 1) if not outcome else -bet_no
            results.append({"side": "NO", "pnl": pnl, "win": not outcome,
                            "bet": bet_no, "edge": edge_no, "pm": cost_no})

    res = pd.DataFrame(results)
    if res.empty:
        print(f"  [{label}] No trades simulated")
        return res

    total_pnl = res["pnl"].sum()
    n         = len(res)
    wr        = res["win"].mean()
    yes_n     = (res["side"] == "YES").sum()
    no_n      = (res["side"] == "NO").sum()

    print(f"  [{label}]  n={n:4d}  YES={yes_n:3d}  NO={no_n:3d}  "
          f"WR={wr:.1%}  PnL=${total_pnl:+,.0f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Loading OHLCV data...")
    print("=" * 70)
    o1h, o4h, o15m, o1m = load_ohlcv()

    c1h = o1h["close"].astype(float)
    h1h = o1h["high"].astype(float)
    l1h = o1h["low"].astype(float)
    v1h = o1h["volume"].astype(float)
    c4h = o4h["close"].astype(float)
    h4h = o4h["high"].astype(float)
    l4h = o4h["low"].astype(float)
    v4h = o4h["volume"].astype(float)
    c15m = o15m["close"].astype(float)
    h15m = o15m["high"].astype(float)
    l15m = o15m["low"].astype(float)
    c1m  = o1m["close"].astype(float)
    v1m  = o1m["volume"].astype(float)
    ts_1h = c1h.index

    # ── Target ────────────────────────────────────────────────────────────────
    next_ret = np.log(c1h / c1h.shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(float)

    # ── Existing composite scores ─────────────────────────────────────────────
    print("\nComputing existing composite scores (60-90s)...")
    trend_old, rev_old = compute_scores(
        c1h, h1h, l1h, v1h,
        c4h, h4h, l4h, v4h,
        c15m, h15m, l15m,
        c1m, v1m, ts_1h,
    )

    # ── New signal votes ──────────────────────────────────────────────────────
    print("\nComputing new candidate signals...")
    sig_df = compute_new_signals(o1h, o4h)
    print("\nVote thresholds (percentile-based on train period):")
    votes  = signals_to_votes(sig_df)

    # ── New trend and rev ─────────────────────────────────────────────────────
    # Test combinations to find best composite
    v_trend = votes["v_kd4h"] + votes["v_emaz4h"] + votes["v_cci4h"] + votes["v_di4h"]
    trend_new_all  = (trend_old + v_trend).clip(-NEW_CLIP, NEW_CLIP)
    trend_new_kd   = (trend_old + votes["v_kd4h"]).clip(-NEW_CLIP, NEW_CLIP)
    trend_new_top2 = (trend_old + votes["v_kd4h"] + votes["v_emaz4h"]).clip(-NEW_CLIP, NEW_CLIP)

    rev_new = (rev_old + votes["v_bb_diff"]).clip(-REV_CLIP, REV_CLIP)

    # ── Build calibration tables (train period: 2025-01-01 → 2026-01-01) ─────
    train_mask = (ts_1h >= TRAIN_START) & (ts_1h < TRAIN_END)

    print(f"\nBuilding calibration tables (train: {TRAIN_START.date()} → {TRAIN_END.date()})...")
    tbl_old, b_old = build_calibration(
        trend_old, rev_old, next_up, train_mask,
        TREND_CLIP, REV_CLIP, "OLD trend")
    tbl_new_all, b_new_all = build_calibration(
        trend_new_all, rev_old, next_up, train_mask,
        NEW_CLIP, REV_CLIP, "NEW trend (all 4 signals)")
    tbl_new_kd, b_new_kd = build_calibration(
        trend_new_kd, rev_old, next_up, train_mask,
        NEW_CLIP, REV_CLIP, "NEW trend (stoch_kd only)")
    tbl_new_top2, b_new_top2 = build_calibration(
        trend_new_top2, rev_old, next_up, train_mask,
        NEW_CLIP, REV_CLIP, "NEW trend (stoch_kd + ema_z)")
    tbl_new_rev, b_new_rev = build_calibration(
        trend_old, rev_new, next_up, train_mask,
        TREND_CLIP, REV_CLIP, "NEW rev (bb_diff added)")

    # ── Walk-forward IC check on OOS (2026-01-01+) ────────────────────────────
    oos_mask = ts_1h >= TRAIN_END
    print(f"\n  Walk-forward IC on OOS ({TRAIN_END.date()} → {ts_1h[oos_mask][-1].date()},"
          f" n={oos_mask.sum():,} 1h bars):")

    def wf_ic(table, baseline, trend_ser, rev_ser, clip_t, clip_r, lbl):
        sub = pd.DataFrame({"t": trend_ser[oos_mask], "r": rev_ser[oos_mask],
                            "up": next_up[oos_mask]}).dropna()
        preds = [lookup_new(table, baseline,
                            row["t"], row["r"], clip_t, clip_r)
                 for _, row in sub.iterrows()]
        r, p = pearsonr(preds, sub["up"].values)
        print(f"    {lbl:<35}  IC={r:+.4f}  p={p:.4f}")

    wf_ic(tbl_old,     b_old,     trend_old,     rev_old, TREND_CLIP, REV_CLIP, "OLD model")
    wf_ic(tbl_new_all, b_new_all, trend_new_all, rev_old, NEW_CLIP,   REV_CLIP, "NEW (all 4 signals)")
    wf_ic(tbl_new_kd,  b_new_kd,  trend_new_kd,  rev_old, NEW_CLIP,  REV_CLIP, "NEW (stoch_kd only)")
    wf_ic(tbl_new_top2,b_new_top2,trend_new_top2,rev_old,NEW_CLIP,   REV_CLIP, "NEW (stoch_kd + ema_z)")
    wf_ic(tbl_new_rev, b_new_rev, trend_old,     rev_new, TREND_CLIP, REV_CLIP, "NEW (bb_diff in rev)")

    # ── Load scan archive ─────────────────────────────────────────────────────
    print("\nLoading btc_scan_archive.csv...")
    archive = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
    archive["logged_at"] = pd.to_datetime(archive["logged_at"], utc=True, errors="coerce")
    archive = archive.dropna(subset=["logged_at", "composite_trend", "composite_rev",
                                     "spot", "strike", "p_market", "vol_eff",
                                     "tau_minutes", "resolved_yes"])
    archive["resolved_yes"] = archive["resolved_yes"].map(
        {True: True, False: False, "True": True, "False": False,
         "1": True, "0": False, 1: True, 0: False, 1.0: True, 0.0: False}
    )
    archive = archive.dropna(subset=["resolved_yes"])
    archive["resolved_yes"] = archive["resolved_yes"].astype(bool)

    # Floor logged_at to nearest 1h bar to join with signal time-series
    archive["bar_1h"] = archive["logged_at"].dt.floor("1h")

    # Build lookup dict for new signal votes at each 1h bar
    vote_at = {}
    for col in ["v_kd4h", "v_emaz4h", "v_cci4h", "v_di4h", "v_bb_diff"]:
        vote_at[col] = votes[col].to_dict()

    def get_vote(bar, col):
        return vote_at[col].get(bar, 0.0)

    # Add new vote columns to archive
    print("Joining new signal votes to scan archive...")
    for col in ["v_kd4h", "v_emaz4h", "v_cci4h", "v_di4h", "v_bb_diff"]:
        archive[col] = archive["bar_1h"].map(vote_at[col]).fillna(0.0)

    archive["trend_new_all"]  = (archive["composite_trend"] + archive["v_kd4h"] +
                                  archive["v_emaz4h"] + archive["v_cci4h"] +
                                  archive["v_di4h"]).clip(-NEW_CLIP, NEW_CLIP)
    archive["trend_new_kd"]   = (archive["composite_trend"] + archive["v_kd4h"]).clip(-NEW_CLIP, NEW_CLIP)
    archive["trend_new_top2"] = (archive["composite_trend"] + archive["v_kd4h"] +
                                  archive["v_emaz4h"]).clip(-NEW_CLIP, NEW_CLIP)
    archive["rev_new"]        = (archive["composite_rev"] + archive["v_bb_diff"]).clip(-REV_CLIP, REV_CLIP)

    print(f"  {len(archive):,} resolved rows  "
          f"({archive['logged_at'].min().date()} → {archive['logged_at'].max().date()})")

    # ── Deduplicate to one row per unique bar × contract ──────────────────────
    archive = archive.drop_duplicates(subset=["bar_1h", "contract_ticker"], keep="first")
    print(f"  {len(archive):,} rows after dedup (one per bar×contract)")

    # ── P&L simulation ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  P&L SIMULATION (flat $1000 bankroll, same simplified gate stack)")
    print("=" * 70)
    print(f"  Min edge={MIN_EDGE}  Max Kelly={MAX_KELLY}  Gate: p_model>=0.10\n")

    res_old = simulate(archive, tbl_old, b_old, TREND_CLIP, REV_CLIP,
                       "composite_trend", "composite_rev", "OLD model")
    res_new_kd = simulate(archive, tbl_new_kd, b_new_kd, NEW_CLIP, REV_CLIP,
                          "trend_new_kd", "composite_rev", "NEW stoch_kd only")
    res_new_top2 = simulate(archive, tbl_new_top2, b_new_top2, NEW_CLIP, REV_CLIP,
                            "trend_new_top2", "composite_rev", "NEW stoch_kd+ema_z")
    res_new_all = simulate(archive, tbl_new_all, b_new_all, NEW_CLIP, REV_CLIP,
                           "trend_new_all", "composite_rev", "NEW all 4 signals")
    res_new_rev = simulate(archive, tbl_new_rev, b_new_rev, TREND_CLIP, REV_CLIP,
                           "composite_trend", "rev_new", "NEW bb_diff in rev")

    # ── Delta breakdown: wins gained, losses gained, wins blocked, losses blocked
    print("\n" + "=" * 70)
    print("  DELTA BREAKDOWN vs OLD (simplified gate stack)")
    print("=" * 70)

    def delta_breakdown(res_new, res_old_ref, label):
        if res_new.empty or res_old_ref.empty:
            return
        old_pnl = res_old_ref["pnl"].sum()
        new_pnl = res_new["pnl"].sum()
        old_n   = len(res_old_ref)
        new_n   = len(res_new)
        delta   = new_pnl - old_pnl
        print(f"\n  {label}")
        print(f"    Old: n={old_n}  PnL=${old_pnl:+,.0f}")
        print(f"    New: n={new_n}  PnL=${new_pnl:+,.0f}  delta=${delta:+,.0f}")

        # Categorize: new trades not in old (gained), old trades not in new (lost)
        # Use (side, resolved_yes) as a proxy since we don't have exact row IDs
        old_by_side = res_old_ref.groupby("side")["pnl"].agg(["sum","count","mean"])
        new_by_side = res_new.groupby("side")["pnl"].agg(["sum","count","mean"])
        print(f"    YES: old n={old_by_side.get('count',{}).get('YES',0):.0f}"
              f"  new n={new_by_side.get('count',{}).get('YES',0):.0f}")
        print(f"    NO:  old n={old_by_side.get('count',{}).get('NO',0):.0f}"
              f"  new n={new_by_side.get('count',{}).get('NO',0):.0f}")

    delta_breakdown(res_new_kd,   res_old, "stoch_kd only")
    delta_breakdown(res_new_top2, res_old, "stoch_kd + ema_z")
    delta_breakdown(res_new_all,  res_old, "All 4 signals")
    delta_breakdown(res_new_rev,  res_old, "bb_diff in rev")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
