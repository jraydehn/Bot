"""
simulate_sideways_coarsen.py

Tests coarsening the Sideways regime calibration table from ±5 to ±3 trend bins.

OLD = current live regime tables (Bull/Sideways/Bear all at ±5)
NEW = same but Sideways rebuilt at ±3 (more data per cell, more stable)

Uses archived macro_regime_bull/sdwy/bear probs to apply the 80/20 regime blend
on each scan archive contract.
"""

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
SYM      = "BTCUSDT"

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")
LABELS_PATH = BASE / "reform_results" / "hmm_macro_labels_btc.parquet"

BANKROLL      = 1000.0
MIN_EDGE      = 0.04
MAX_KELLY     = 0.15
K_DRIFT_YES   = 1.40
K_DRIFT_NO    = 0.30
SMOOTH_K      = 30
TREND_CLIP    = 5      # Bull / Bear stay at ±5
SW_CLIP_OLD   = 5      # current Sideways
SW_CLIP_NEW   = 3      # proposed coarser Sideways
REV_CLIP      = 11
POOLED_WEIGHT = 0.20   # matches lookup_p_up_regime pooled_fallback

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores, lookup_p_up, BASELINE_UP,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_ohlcv():
    def pick(pat):
        f = sorted(glob.glob(str(DATA_DIR / pat)))
        return f[-1]
    def load(p):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = df.columns.str.lower()
        return df.sort_index()
    o1h  = load(pick(f"binanceus_{SYM}_1h_1970-01-01_*.parquet"))
    o4h  = load(pick(f"binanceus_{SYM}_4h_1970-01-01_*.parquet"))
    o15m = load(sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_15m_2024-01-01_*.parquet")))[-1])
    o1m  = load(sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_1m_2024-01-01_*.parquet")))[-1])
    print(f"  1h: {len(o1h):,}  4h: {len(o4h):,}")
    return o1h, o4h, o15m, o1m


def build_regime_table(df_regime, baseline, tc, label):
    """Build (tb,rb)->p_up table for one regime slice at given trend clip."""
    counts = {}
    for t in range(-tc, tc + 1):
        for r in range(-REV_CLIP, REV_CLIP + 1):
            cell = df_regime[(df_regime["tb_full"] == t) & (df_regime["rb"] == r)]
            n    = len(cell)
            wr   = cell["next_up"].mean() if n >= 10 else float("nan")
            counts[(t, r)] = (n, wr)

    tbl = {}
    filled = 0
    for (t, r), (n, wr) in counts.items():
        if np.isnan(wr):
            p = baseline
        else:
            w = min(1.0, n / SMOOTH_K)
            p = w * wr + (1 - w) * baseline
            filled += 1
        tbl[(t, r)] = p

    print(f"  {label}: {filled}/{len(tbl)} filled  baseline={baseline:.4f}")
    return tbl, baseline


def lookup_regime(tbl, baseline, t, r, tc):
    t = int(np.clip(t, -tc, tc))
    r = int(np.clip(r, -REV_CLIP, REV_CLIP))
    return tbl.get((t, r), baseline)


def blend_p_up(trend, rev, regime_probs, tbls, baselines, clips):
    """80% regime-blended + 20% pooled (matches lookup_p_up_regime)."""
    p_regime = 0.0
    total_w  = 0.0
    for reg, prob in regime_probs.items():
        tbl      = tbls[reg]
        bl       = baselines[reg]
        tc       = clips[reg]
        cell_val = lookup_regime(tbl, bl, trend, rev, tc)
        p_regime += prob * cell_val
        total_w  += prob
    if total_w < 1e-6:
        return BASELINE_UP
    p_regime /= total_w
    p_pooled  = lookup_p_up(trend, rev, asset="BTC")
    return (1.0 - POOLED_WEIGHT) * p_regime + POOLED_WEIGHT * p_pooled


def p_model_yes(p_up, spot, strike, tau_min, vol_eff):
    if tau_min <= 0 or vol_eff <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    tau   = tau_min / (252 * 390)
    sigma = vol_eff * np.sqrt(tau)
    drift = K_DRIFT_YES * (p_up - 0.5) * sigma
    return float(np.clip(1 - norm.cdf((np.log(strike / spot) - drift) / sigma), 0.01, 0.99))


def p_model_no(p_up, spot, strike, tau_min, vol_eff):
    if tau_min <= 0 or vol_eff <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    tau   = tau_min / (252 * 390)
    sigma = vol_eff * np.sqrt(tau)
    drift = K_DRIFT_NO * (p_up - 0.5) * sigma
    return float(np.clip(norm.cdf((np.log(strike / spot) - drift) / sigma), 0.01, 0.99))


def kelly_bet(p_model, pm, bankroll):
    pm   = np.clip(pm, 0.01, 0.99)
    edge = p_model - pm
    if edge < MIN_EDGE:
        return 0.0
    odds = (1 - pm) / pm
    k    = (p_model * odds - (1 - p_model)) / odds
    k    = np.clip(k, 0, MAX_KELLY)
    return round(bankroll * k, 2)


def simulate(archive, tbls, baselines, clips, label):
    pnl = 0.0
    trades = []
    for _, row in archive.iterrows():
        t   = row["trend_cur"]
        r   = row["composite_rev"]
        rp  = {"Bull": row["macro_regime_bull"],
               "Sideways": row["macro_regime_sdwy"],
               "Bear": row["macro_regime_bear"]}
        pu  = blend_p_up(t, r, rp, tbls, baselines, clips)

        spot   = row["spot"];   strike = row["strike"]
        pm     = row["p_market"]; tau  = row["tau_minutes"]
        vol    = row["vol_eff"]; out   = row["resolved_yes"]

        pmy  = p_model_yes(pu, spot, strike, tau, vol)
        bety = kelly_bet(pmy, pm, BANKROLL)
        if bety > 0:
            gain = bety * (1 - pm) / pm if out else -bety
            trades.append({"side": "YES", "pnl": gain, "won": out, "regime": _reg(rp)})
            pnl += gain
            continue

        pmn  = p_model_no(pu, spot, strike, tau, vol)
        betn = kelly_bet(pmn, 1 - pm, BANKROLL)
        if betn > 0:
            gain = betn * pm / (1 - pm) if not out else -betn
            trades.append({"side": "NO", "pnl": gain, "won": not out, "regime": _reg(rp)})
            pnl += gain

    df  = pd.DataFrame(trades)
    n   = len(df)
    wr  = df["won"].mean() if n > 0 else float("nan")
    print(f"\n  {label}: n={n:,}  WR={wr:.1%}  PnL={pnl:+,.0f}")
    return pnl, n, wr, df


def _reg(rp):
    return max(rp, key=rp.get)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading OHLCV...")
    o1h, o4h, o15m, o1m = load_ohlcv()

    c1h  = o1h["close"].astype(float); h1h = o1h["high"].astype(float)
    l1h  = o1h["low"].astype(float);   v1h = o1h["volume"].astype(float)
    c4h  = o4h["close"].astype(float); h4h = o4h["high"].astype(float)
    l4h  = o4h["low"].astype(float);   v4h = o4h["volume"].astype(float)
    c15m = o15m["close"].astype(float); h15m = o15m["high"].astype(float)
    l15m = o15m["low"].astype(float)
    c1m  = o1m["close"].astype(float); v1m = o1m["volume"].astype(float)
    ts_1h = c1h.index

    print("\nComputing composite scores...")
    trend_ser, rev_ser = compute_scores(
        c1h, h1h, l1h, v1h, c4h, h4h, l4h, v4h,
        c15m, h15m, l15m, c1m, v1m, ts_1h,
    )

    next_ret  = np.log(c1h / c1h.shift(1)).shift(-1)
    next_up   = (next_ret > 0).astype(float)
    test_mask = ts_1h >= TRAIN_START

    # Build scored df (full test period)
    df_scores = pd.DataFrame({
        "trend": trend_ser, "rev": rev_ser, "next_up": next_up,
    }).dropna()
    df_scores = df_scores[df_scores.index >= TRAIN_START]
    df_scores["tb_full"] = df_scores["trend"].clip(-TREND_CLIP, TREND_CLIP).astype(int)
    df_scores["rb"]      = df_scores["rev"].clip(-REV_CLIP, REV_CLIP).astype(int)

    print("\nJoining HMM regime labels...")
    labels = pd.read_parquet(LABELS_PATH)
    labels.index = pd.to_datetime(labels.index, utc=True)
    idx = np.searchsorted(labels.index.values, df_scores.index.values, side="right") - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    df_scores["regime"] = labels["regime"].values[idx]

    baseline_all = df_scores["next_up"].mean()
    REGIMES = ["Bull", "Sideways", "Bear"]

    print("\nBuilding per-regime tables...")
    tbls_old = {}; tbls_new = {}
    baselines_dict = {}
    clips_old = {"Bull": TREND_CLIP, "Sideways": SW_CLIP_OLD, "Bear": TREND_CLIP}
    clips_new = {"Bull": TREND_CLIP, "Sideways": SW_CLIP_NEW, "Bear": TREND_CLIP}

    for reg in REGIMES:
        sub = df_scores[df_scores["regime"] == reg]
        bl  = sub["next_up"].mean() if len(sub) > 0 else baseline_all
        baselines_dict[reg] = bl

        # OLD: Sideways at ±5, others at ±5
        tc_old = clips_old[reg]
        # For OLD, need to re-clip tb at the old tc
        sub2 = sub.copy()
        sub2["tb_full"] = sub2["trend"].clip(-tc_old, tc_old).astype(int)
        tbl, _ = build_regime_table(sub2, bl, tc_old, f"OLD {reg} (±{tc_old})")
        tbls_old[reg] = tbl

        # NEW: Sideways at ±3, others unchanged
        tc_new = clips_new[reg]
        sub3 = sub.copy()
        sub3["tb_full"] = sub3["trend"].clip(-tc_new, tc_new).astype(int)
        tbl2, _ = build_regime_table(sub3, bl, tc_new, f"NEW {reg} (±{tc_new})")
        tbls_new[reg] = tbl2

    # Load scan archive
    print("\nLoading scan archive...")
    arc = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
    arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce")
    arc = arc.dropna(subset=["logged_at", "composite_rev", "spot", "strike",
                              "p_market", "vol_eff", "tau_minutes", "resolved_yes"])
    arc["resolved_yes"] = arc["resolved_yes"].map(
        {True: True, False: False, "True": True, "False": False,
         "1": True, "0": False, 1: True, 0: False, 1.0: True, 0.0: False}
    )
    arc = arc.dropna(subset=["resolved_yes"])
    arc["resolved_yes"] = arc["resolved_yes"].astype(bool)
    arc = arc.sort_values("logged_at").drop_duplicates(subset=["contract_ticker"], keep="first")
    print(f"  {len(arc):,} resolved contracts")

    # Join current trend score
    trend_dict = trend_ser.to_dict()
    arc["bar_1h"]    = arc["logged_at"].dt.floor("1h")
    arc["trend_cur"] = arc["bar_1h"].map(trend_dict).fillna(0.0)

    # Join HMM regime label (hard assignment — macro_regime_* cols are NaN in archive)
    print("  Joining HMM regime labels from parquet...")
    labels = pd.read_parquet(LABELS_PATH)
    labels.index = pd.to_datetime(labels.index, utc=True)
    arc_times   = arc["bar_1h"].values
    label_times = labels.index.values
    idx = np.searchsorted(label_times, arc_times, side="right") - 1
    idx = np.clip(idx, 0, len(label_times) - 1)
    arc["regime_label"] = labels["regime"].values[idx]
    # Convert hard label to probability dict (100% to dominant regime)
    arc["macro_regime_bull"] = (arc["regime_label"] == "Bull").astype(float)
    arc["macro_regime_sdwy"] = (arc["regime_label"] == "Sideways").astype(float)
    arc["macro_regime_bear"] = (arc["regime_label"] == "Bear").astype(float)
    print(f"  Regime distribution: {arc['regime_label'].value_counts().to_dict()}")

    print("\nSimulating...")
    pnl_old, n_old, wr_old, df_old = simulate(arc, tbls_old, baselines_dict, clips_old, "OLD (Sideways ±5)")
    pnl_new, n_new, wr_new, df_new = simulate(arc, tbls_new, baselines_dict, clips_new, "NEW (Sideways ±3)")

    print(f"\n{'='*70}")
    print("  SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"  {'':30}  {'n':>6}  {'WR':>7}  {'P&L':>9}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*7}  {'-'*9}")
    print(f"  {'OLD (Sideways ±5)':30}  {n_old:>6,}  {wr_old:>7.1%}  {pnl_old:>+9,.0f}")
    print(f"  {'NEW (Sideways ±3)':30}  {n_new:>6,}  {wr_new:>7.1%}  {pnl_new:>+9,.0f}")
    print(f"  {'Delta':30}  {n_new-n_old:>+6,}  {'':>7}  {pnl_new-pnl_old:>+9,.0f}")

    # Breakdown by regime
    print(f"\n{'='*70}")
    print("  P&L DELTA BY REGIME")
    print(f"{'='*70}")
    for reg in REGIMES:
        mask = arc.apply(lambda r: _reg({"Bull": r["macro_regime_bull"],
                                          "Sideways": r["macro_regime_sdwy"],
                                          "Bear": r["macro_regime_bear"]}) == reg, axis=1)
        sub  = arc[mask]
        if len(sub) == 0:
            continue
        p_o, n_o, w_o, _ = simulate(sub, tbls_old, baselines_dict, clips_old, f"  OLD {reg}")
        p_n, n_n, w_n, _ = simulate(sub, tbls_new, baselines_dict, clips_new, f"  NEW {reg}")
        print(f"  {reg:<10}  OLD: n={n_o:,} WR={w_o:.1%} PnL={p_o:+,.0f}  |  "
              f"NEW: n={n_n:,} WR={w_n:.1%} PnL={p_n:+,.0f}  |  Δ={p_n-p_o:+,.0f}")

    # Trade delta
    if len(df_old) > 0 and len(df_new) > 0:
        gained = set(df_new.index) - set(df_old.index)
        lost   = set(df_old.index) - set(df_new.index)
        print(f"\n{'='*70}")
        print("  TRADE DELTA")
        print(f"{'='*70}")
        if gained:
            g = df_new.loc[list(gained)]
            print(f"  Gained: n={len(gained)}  wins={g['won'].sum():.0f}  PnL={g['pnl'].sum():+,.0f}")
        if lost:
            l = df_old.loc[list(lost)]
            print(f"  Lost:   n={len(lost)}  wins={l['won'].sum():.0f}  PnL={l['pnl'].sum():+,.0f}  (removed)")
        if not gained and not lost:
            print("  Same contracts traded — only sizing changed")

    print(f"\n{'='*70}")
    verdict = "IMPLEMENT" if pnl_new >= pnl_old else "REJECT"
    print(f"  VERDICT: {verdict}  (delta = {pnl_new-pnl_old:+,.0f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
