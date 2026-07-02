"""
simulate_vol_dir_boost.py

Simulates boosting vol_dir_4h vote weight from ±1 to ±2 and measures P&L impact.

OLD = current live model (kd_diff ±2 + ema_z ±2 + 6 legacy signals ±1, trend clip ±5)
NEW = same but vol_dir_4h boosted from ±1 to ±2 (delta = +1 when high_vol_up, -1 when down)

Flat $1000 bankroll, same MIN_EDGE/MAX_KELLY as runner.
Reports: wins blocked, losses blocked, net P&L delta, WR.
"""

import glob
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

BANKROLL   = 1000.0
MIN_EDGE   = 0.04
MAX_KELLY  = 0.15
K_DRIFT_YES = 1.40
K_DRIFT_NO  = 0.30
SMOOTH_K   = 30
TREND_CLIP = 5    # current live clip (expanded from 3 after kd_diff/ema_z)
REV_CLIP   = 11

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores, _stoch_k, _vol_signal_4h,
    BASELINE_UP,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_ohlcv():
    def pick(pat):
        f = sorted(glob.glob(str(DATA_DIR / pat)))
        if not f: raise FileNotFoundError(pat)
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


def build_calibration(trend_ser, rev_ser, next_up, train_mask, label):
    tb = trend_ser.clip(-TREND_CLIP, TREND_CLIP).astype(int)
    rb = rev_ser.clip(-REV_CLIP, REV_CLIP).astype(int)
    df = pd.DataFrame({"tb": tb, "rb": rb, "up": next_up})[train_mask].dropna()
    baseline = df["up"].mean()
    tbl = {}
    for t in range(-TREND_CLIP, TREND_CLIP + 1):
        for r in range(-REV_CLIP, REV_CLIP + 1):
            cell = df[(df["tb"] == t) & (df["rb"] == r)]
            n = len(cell)
            if n >= 10:
                w = min(1.0, n / SMOOTH_K)
                tbl[(t, r)] = w * cell["up"].mean() + (1 - w) * baseline
    print(f"  {label}: {len(tbl)} cells  baseline={baseline:.4f}")
    return tbl, baseline


def lookup(tbl, baseline, t, r):
    t = int(np.clip(t, -TREND_CLIP, TREND_CLIP))
    r = int(np.clip(r, -REV_CLIP,   REV_CLIP))
    return tbl.get((t, r), baseline)


def p_model_yes(p_up, spot, strike, tau_min, vol_eff):
    if tau_min <= 0 or vol_eff <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    tau = tau_min / (252 * 390)
    sigma = vol_eff * np.sqrt(tau)
    drift = K_DRIFT_YES * (p_up - 0.5) * sigma
    log_d = np.log(strike / spot) - drift
    return float(np.clip(1 - norm.cdf(log_d / sigma), 0.01, 0.99))


def p_model_no(p_up, spot, strike, tau_min, vol_eff):
    if tau_min <= 0 or vol_eff <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    tau = tau_min / (252 * 390)
    sigma = vol_eff * np.sqrt(tau)
    drift = K_DRIFT_NO * (p_up - 0.5) * sigma
    log_d = np.log(strike / spot) - drift
    return float(np.clip(norm.cdf(log_d / sigma), 0.01, 0.99))


def kelly_bet(p_model, pm, edge_min, max_k, bankroll):
    pm = np.clip(pm, 0.01, 0.99)
    edge = p_model - pm
    if edge < edge_min:
        return 0.0
    odds = (1 - pm) / pm
    k = (p_model * odds - (1 - p_model)) / odds
    k = np.clip(k, 0, max_k)
    return round(bankroll * k, 2)


def simulate(archive, tbl, baseline, trend_col):
    pnl = 0.0
    trades = []
    for _, row in archive.iterrows():
        t = row[trend_col]
        r = row["composite_rev"]
        pu = lookup(tbl, baseline, t, r)
        spot   = row["spot"]
        strike = row["strike"]
        pm     = row["p_market"]
        tau    = row["tau_minutes"]
        vol    = row["vol_eff"]
        out    = row["resolved_yes"]

        # YES side
        pmy = p_model_yes(pu, spot, strike, tau, vol)
        bet_y = kelly_bet(pmy, pm, MIN_EDGE, MAX_KELLY, BANKROLL)
        if bet_y > 0:
            if out:
                gain = bet_y * (1 - pm) / pm
            else:
                gain = -bet_y
            trades.append({"side": "YES", "bet": bet_y, "pnl": gain, "won": out,
                           "pmy": pmy, "pm": pm})
            pnl += gain
            continue

        # NO side
        pmn = p_model_no(pu, spot, strike, tau, vol)
        bet_n = kelly_bet(pmn, 1 - pm, MIN_EDGE, MAX_KELLY, BANKROLL)
        if bet_n > 0:
            if not out:
                gain = bet_n * pm / (1 - pm)
            else:
                gain = -bet_n
            trades.append({"side": "NO", "bet": bet_n, "pnl": gain, "won": not out,
                           "pmn": pmn, "pm": pm})
            pnl += gain

    df = pd.DataFrame(trades)
    n = len(df)
    wr = df["won"].mean() if n > 0 else float("nan")
    return pnl, n, wr, df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading OHLCV...")
    o1h, o4h, o15m, o1m = load_ohlcv()

    c1h  = o1h["close"].astype(float);  h1h = o1h["high"].astype(float)
    l1h  = o1h["low"].astype(float);    v1h = o1h["volume"].astype(float)
    c4h  = o4h["close"].astype(float);  h4h = o4h["high"].astype(float)
    l4h  = o4h["low"].astype(float);    v4h = o4h["volume"].astype(float)
    c15m = o15m["close"].astype(float); h15m = o15m["high"].astype(float)
    l15m = o15m["low"].astype(float)
    c1m  = o1m["close"].astype(float);  v1m  = o1m["volume"].astype(float)
    ts_1h = c1h.index

    print("\nComputing composite scores (current model with kd_diff + ema_z)...")
    trend_cur, rev_cur = compute_scores(
        c1h, h1h, l1h, v1h,
        c4h, h4h, l4h, v4h,
        c15m, h15m, l15m,
        c1m, v1m, ts_1h,
    )

    # Vol_dir delta: +1 when high_vol_up (was +1, will be +2 → delta=+1)
    #               -1 when high_vol_down (was -1, will be -2 → delta=-1)
    #                0 otherwise
    print("\nComputing vol_dir delta (±1 boost)...")
    vsig = _vol_signal_4h(c4h, v4h)
    vol_delta_4h = pd.Series(0.0, index=vsig.index)
    vol_delta_4h[vsig == "high_vol_up"]   =  1.0
    vol_delta_4h[vsig == "high_vol_down"] = -1.0
    vol_delta_1h = vol_delta_4h.reindex(ts_1h, method="ffill").fillna(0.0)

    trend_new = (trend_cur + vol_delta_1h).clip(-TREND_CLIP - 1, TREND_CLIP + 1)

    # Next-up target
    next_ret = np.log(c1h / c1h.shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(float)
    train_mask = (ts_1h >= TRAIN_START) & (ts_1h < TRAIN_END)

    print("\nBuilding calibration tables...")
    tbl_old, b_old = build_calibration(trend_cur, rev_cur, next_up, train_mask, "OLD (current live)")
    tbl_new, b_new = build_calibration(trend_new, rev_cur, next_up, train_mask, "NEW (vol_dir ±2)")

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

    # Dedup: keep first scan per ticker (most timely signal)
    arc = arc.sort_values("logged_at").drop_duplicates(subset=["contract_ticker"], keep="first")
    arc = arc[arc["resolved_yes"].notna()].copy()
    print(f"  {len(arc):,} resolved contracts")

    # Join current trend and vol_delta to archive
    arc["bar_1h"] = arc["logged_at"].dt.floor("1h")
    trend_cur_dict  = trend_cur.to_dict()
    vol_delta_dict  = vol_delta_1h.to_dict()

    arc["trend_cur"]   = arc["bar_1h"].map(trend_cur_dict).fillna(0.0)
    arc["vol_delta"]   = arc["bar_1h"].map(vol_delta_dict).fillna(0.0)
    arc["trend_new_v"] = (arc["trend_cur"] + arc["vol_delta"]).clip(-TREND_CLIP - 1, TREND_CLIP + 1)

    print("\nSimulating...")
    pnl_old, n_old, wr_old, df_old = simulate(arc, tbl_old, b_old, "trend_cur")
    pnl_new, n_new, wr_new, df_new = simulate(arc, tbl_new, b_new, "trend_new_v")

    print(f"\n{'='*70}")
    print("  SIMULATION RESULTS  (flat $1,000 bankroll, full resolved archive)")
    print(f"{'='*70}")
    print(f"  {'':30}  {'n':>6}  {'WR':>7}  {'P&L':>9}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*7}  {'-'*9}")
    print(f"  {'OLD (vol_dir ±1 — current)':30}  {n_old:>6,}  {wr_old:>7.1%}  {pnl_old:>+9,.0f}")
    print(f"  {'NEW (vol_dir ±2)':30}  {n_new:>6,}  {wr_new:>7.1%}  {pnl_new:>+9,.0f}")
    print(f"  {'Delta':30}  {n_new-n_old:>+6,}  {'':>7}  {pnl_new-pnl_old:>+9,.0f}")

    # Wins vs losses changed
    old_tickers = set(df_old.index) if len(df_old) > 0 else set()
    # Compare by checking contracts that switched bet/no-bet
    print(f"\n{'='*70}")
    print("  TRADE DELTA BREAKDOWN")
    print(f"{'='*70}")

    # Contracts bet in OLD but not NEW
    # Join on archive row to compare
    arc_traded_old = arc[arc.index.isin(df_old.index)] if len(df_old) > 0 else arc.iloc[:0]
    arc_traded_new = arc[arc.index.isin(df_new.index)] if len(df_new) > 0 else arc.iloc[:0]

    old_set = set(df_old.index) if len(df_old) > 0 else set()
    new_set = set(df_new.index) if len(df_new) > 0 else set()

    gained  = new_set - old_set
    lost    = old_set - new_set
    kept    = old_set & new_set

    if gained:
        g_rows = df_new.loc[list(gained)]
        g_wins = g_rows["won"].sum(); g_pnl = g_rows["pnl"].sum()
        print(f"  Contracts GAINED by NEW:  n={len(gained):,}  wins={g_wins:.0f}  losses={len(gained)-g_wins:.0f}  PnL={g_pnl:+,.0f}")
    if lost:
        l_rows = df_old.loc[list(lost)]
        l_wins = l_rows["won"].sum(); l_pnl = l_rows["pnl"].sum()
        print(f"  Contracts LOST from OLD:  n={len(lost):,}  wins={l_wins:.0f}  losses={len(lost)-l_wins:.0f}  PnL={l_pnl:+,.0f}  (this PnL is REMOVED)")
    print(f"  Contracts kept (both):     n={len(kept):,}")

    # By regime if available
    if "macro_regime_bull" in arc.columns:
        print(f"\n{'='*70}")
        print("  P&L DELTA BY REGIME")
        print(f"{'='*70}")
        def regime_label(row):
            if row.get("macro_regime_bull", 0) > 0.5: return "Bull"
            if row.get("macro_regime_sdwy", 0) > 0.5: return "Sideways"
            if row.get("macro_regime_bear", 0) > 0.5: return "Bear"
            return "Unknown"
        arc["regime"] = arc.apply(regime_label, axis=1)
        for r in ["Bull", "Sideways", "Bear"]:
            mask = arc["regime"] == r
            sub  = arc[mask]
            p_o, n_o, w_o, d_o = simulate(sub, tbl_old, b_old, "trend_cur")
            p_n, n_n, w_n, d_n = simulate(sub, tbl_new, b_new, "trend_new_v")
            print(f"  {r:<10}  OLD: n={n_o:,} WR={w_o:.1%} PnL={p_o:+,.0f}  |  "
                  f"NEW: n={n_n:,} WR={w_n:.1%} PnL={p_n:+,.0f}  |  Δ={p_n-p_o:+,.0f}")

    print(f"\n{'='*70}")
    bkv = "IMPLEMENT" if pnl_new > pnl_old else "REJECT"
    print(f"  VERDICT: {bkv}  (delta = {pnl_new-pnl_old:+,.0f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
