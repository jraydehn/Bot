"""
simulate_btc_gate_audit.py

Evaluates ALL BTC YES gates from paper_trade_runner.py against the z_drift model.

Parts:
  1. Simulation on logged gates (blocked_trades.csv): WR, PnL, z_drift overlap
  2. Analytical signal overlap assessment for invisible gates
  3. z_drift alignment analysis
"""

import math
import time
import warnings
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
BLOCKED_CSV   = BASE / "results" / "blocked_trades.csv"
PT15_CSV      = BASE / "results" / "paper_trades_btc15m.csv"
PT_OLD_CSV    = BASE / "results" / "paper_trades.csv"

# ─── z_drift parameters ───────────────────────────────────────────────────────
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5
EDGE_THRESHOLD    = 0.04
MINS_PER_YEAR     = 525_600.0
FLAT_BET          = 10.0

# ─── Gate categories ──────────────────────────────────────────────────────────
# Gates that ARE logged in blocked_trades.csv (BTC YES side)
LOGGED_GATES = [
    "smc_gate",
    "streak_gate",
    "bear_drift",
    "liq_cascade_gate",
    "btc_otmlow_gate",
    "btc_otm_neutral_gate",
    "btc_adx5_gate",
    "btc_falling_knife_gate",
    "btc_body_bp_gate",
    "btc_struct_gate",
    "btc_ema0_stretch2_gate",
    "ema_stack3_gate",
    "btc_ema0_itm_gate",
    "btc_exhaustion_gate",
    "btc_tau_gate",
    "btc_spread_gate",
]

# ─── Binance fetch ─────────────────────────────────────────────────────────────
def fetch_binance_1m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 1m candles from Binance US in batches."""
    url_base = "https://api.binance.us/api/v3/klines"
    all_rows = []
    cur = start_ms
    batch = 1000  # max per request
    while cur < end_ms:
        url = (
            f"{url_base}?symbol={symbol}&interval=1m"
            f"&startTime={cur}&endTime={end_ms}&limit={batch}"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  [fetch] error: {e} — sleeping 5s")
            time.sleep(5)
            continue
        if not data:
            break
        all_rows.extend(data)
        last_ts = data[-1][0]
        if last_ts >= end_ms or len(data) < batch:
            break
        cur = last_ts + 60_000
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "qav", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def price_at(binance_1m: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Return the close price of the 1m candle that contains ts."""
    if binance_1m.empty:
        return np.nan
    idx = binance_1m.index.searchsorted(ts, side="right")
    if idx == 0:
        return float(binance_1m.iloc[0]["close"])
    return float(binance_1m.iloc[idx - 1]["close"])


# ─── Build actual_z history from resolved trades ──────────────────────────────
def build_actual_z_series(binance_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Build a chronological series of (close_ts_utc, actual_z, vol_eff, tau_minutes)
    from the union of paper_trades.csv (old format, has vol_eff) and
    paper_trades_btc15m.csv (new format, has realized_vol_annual).
    Returns DataFrame sorted by close_ts_utc.
    """
    rows = []

    # ── Old format (paper_trades.csv): has vol_eff per-minute ─────────────────
    try:
        df_old = pd.read_csv(PT_OLD_CSV, low_memory=False)
        df_old["close_ts_utc"] = pd.to_datetime(df_old["close_ts"], utc=True, errors="coerce")
        df_old = df_old.dropna(subset=["close_ts_utc", "spot", "vol_eff", "tau_minutes", "resolved_yes"])
        df_old = df_old[df_old["vol_eff"] > 0]
        df_old["resolved_yes"] = df_old["resolved_yes"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        )
        df_old = df_old.dropna(subset=["resolved_yes"])
        for _, row in df_old.iterrows():
            ts = row["close_ts_utc"]
            spot = float(row["spot"])
            vol_eff = float(row["vol_eff"])  # per-minute vol
            tau_min = float(row["tau_minutes"])
            sigma_tau = vol_eff * math.sqrt(tau_min)
            btc_expiry = price_at(binance_1m, ts)
            if np.isnan(btc_expiry) or btc_expiry <= 0 or spot <= 0 or sigma_tau <= 0:
                continue
            actual_z = math.log(btc_expiry / spot) / sigma_tau
            rows.append({"close_ts_utc": ts, "actual_z": actual_z,
                         "vol_eff": vol_eff, "sigma_tau": sigma_tau,
                         "tau_minutes": tau_min, "source": "old"})
    except Exception as e:
        print(f"  [actual_z] old format error: {e}")

    # ── New format (paper_trades_btc15m.csv): has realized_vol_annual ─────────
    try:
        df_new = pd.read_csv(PT15_CSV, low_memory=False)
        df_new["close_ts_utc"] = pd.to_datetime(df_new["close_time"], utc=True, errors="coerce")
        df_new = df_new.dropna(subset=["close_ts_utc", "spot", "realized_vol_annual", "tau_minutes", "resolved_yes"])
        df_new["rv_ann"] = df_new["realized_vol_annual"].astype(float)
        df_new = df_new[df_new["rv_ann"] > 0]
        df_new["resolved_yes"] = df_new["resolved_yes"].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        )
        df_new = df_new.dropna(subset=["resolved_yes"])
        for _, row in df_new.iterrows():
            ts = row["close_ts_utc"]
            spot = float(row["spot"])
            rv_ann = float(row["rv_ann"])
            tau_min = float(row["tau_minutes"])
            vol_eff = rv_ann / math.sqrt(MINS_PER_YEAR)
            sigma_tau = vol_eff * math.sqrt(tau_min)
            btc_expiry = price_at(binance_1m, ts)
            if np.isnan(btc_expiry) or btc_expiry <= 0 or spot <= 0 or sigma_tau <= 0:
                continue
            actual_z = math.log(btc_expiry / spot) / sigma_tau
            rows.append({"close_ts_utc": ts, "actual_z": actual_z,
                         "vol_eff": vol_eff, "sigma_tau": sigma_tau,
                         "tau_minutes": tau_min, "source": "new"})
    except Exception as e:
        print(f"  [actual_z] new format error: {e}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("close_ts_utc").drop_duplicates(
        subset=["close_ts_utc"], keep="first"
    )
    print(f"  [actual_z] built {len(df)} resolved z-values "
          f"({df['close_ts_utc'].min().date()} → {df['close_ts_utc'].max().date()})")
    return df


def compute_zdrift_at(actual_z_df: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Compute z_drift using walk-forward — only use trades resolved BEFORE ts."""
    hist = actual_z_df[actual_z_df["close_ts_utc"] < ts]
    if len(hist) < W_SHORT:
        return 0.0
    zs = hist["actual_z"].values
    zs = zs[-max(W_LONG, 50):]  # mirror the live logic
    if len(zs) < W_SHORT:
        return 0.0
    z_short = zs[-W_SHORT:].mean()
    z_long  = zs[-W_LONG:].mean() if len(zs) >= W_LONG else zs.mean()
    raw = ALPHA * z_short + (1 - ALPHA) * z_long
    return float(np.clip(raw, -CAP, CAP))


# ─── rv_ann lookup helper ──────────────────────────────────────────────────────
def build_rv_lookup() -> pd.Series:
    """
    Build a time-indexed rv_ann series from paper_trades_btc15m.csv
    for merge_asof use.
    """
    df = pd.read_csv(PT15_CSV, low_memory=False)
    df["ts"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "realized_vol_annual"])
    df = df.sort_values("ts")
    return df.set_index("ts")["realized_vol_annual"]


def get_rv_ann(rv_series: pd.Series, ts: pd.Timestamp, default: float = 0.30) -> float:
    """Get the most recent rv_ann at or before ts."""
    idx = rv_series.index.searchsorted(ts, side="right")
    if idx == 0:
        return default
    return float(rv_series.iloc[idx - 1])


# ─── PnL helpers ──────────────────────────────────────────────────────────────
def calc_pnl(pm: float, outcome: int) -> float:
    """Flat $10 per trade YES side P&L."""
    if outcome == 1:
        return FLAT_BET * (1.0 - pm) / pm
    return -FLAT_BET


def breakeven_wr(pm: float) -> float:
    return pm


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("GATE AUDIT REPORT — BTC YES gates vs z_drift")
    print("=" * 64)

    # ── Load blocked trades ────────────────────────────────────────────────────
    print("\n[1] Loading blocked_trades.csv …")
    bt = pd.read_csv(BLOCKED_CSV, low_memory=False)
    bt_btc_yes = bt[(bt["asset"] == "BTC") & (bt["side"] == "yes")].copy()
    bt_btc_yes["close_ts_utc"] = pd.to_datetime(bt_btc_yes["close_ts"], utc=True, errors="coerce")
    bt_btc_yes["logged_at_dt"]  = pd.to_datetime(bt_btc_yes["logged_at"], utc=True, errors="coerce")
    bt_btc_yes = bt_btc_yes.dropna(subset=["close_ts_utc"])

    # Deduplicate by (close_ts, strike) — keep first occurrence
    bt_btc_yes = bt_btc_yes.sort_values("logged_at_dt").drop_duplicates(
        subset=["close_ts_utc", "strike"], keep="first"
    )
    print(f"  BTC YES blocked: {len(bt_btc_yes)} rows (after dedup by close_ts+strike)")

    # ── Fetch Binance 1m data ─────────────────────────────────────────────────
    print("\n[2] Fetching Binance 1m BTCUSDT …")
    min_ts = bt_btc_yes["close_ts_utc"].min()
    max_ts = bt_btc_yes["close_ts_utc"].max()
    # Add 1 hour buffer on each end
    start_ms = int((min_ts - pd.Timedelta(hours=1)).timestamp() * 1000)
    end_ms   = int((max_ts + pd.Timedelta(hours=1)).timestamp() * 1000)
    print(f"  Range: {min_ts.date()} → {max_ts.date()}")
    binance_1m = fetch_binance_1m("BTCUSDT", start_ms, end_ms)
    print(f"  Fetched {len(binance_1m)} 1m candles")

    # ── Build actual_z history ─────────────────────────────────────────────────
    print("\n[3] Building actual_z history from paper_trades …")
    actual_z_df = build_actual_z_series(binance_1m)

    # ── rv_ann lookup ──────────────────────────────────────────────────────────
    rv_series = build_rv_lookup()

    # ── Compute per-trade metrics ─────────────────────────────────────────────
    print("\n[4] Computing outcomes, z_drift, p_zd for each blocked trade …")

    records = []
    total = len(bt_btc_yes)
    for i, (_, row) in enumerate(bt_btc_yes.iterrows()):
        if i % 500 == 0:
            print(f"  Processing {i}/{total} …")

        ts    = row["close_ts_utc"]
        strike = float(row["strike"])
        spot   = float(row["spot"])
        pm     = float(row["pm"])
        tau    = float(row["tau_minutes"])
        gate   = row["gate_name"]

        # ── Derive outcome from Binance price at close_ts ──────────────────────
        btc_at_close = price_at(binance_1m, ts)
        if np.isnan(btc_at_close) or btc_at_close <= 0:
            outcome = np.nan
        else:
            outcome = 1 if btc_at_close > strike else 0

        # ── z_drift at this point (walk-forward) ───────────────────────────────
        z_drift = compute_zdrift_at(actual_z_df, ts)

        # ── sigma_tau from rv_ann ──────────────────────────────────────────────
        rv_ann = get_rv_ann(rv_series, ts, default=0.30)
        vol_eff = rv_ann / math.sqrt(MINS_PER_YEAR)
        sigma_tau = vol_eff * math.sqrt(max(tau, 0.1))

        # ── p_zd: norm.cdf(z_drift - log(K/S) / sigma_tau) ───────────────────
        if spot > 0 and sigma_tau > 0:
            d = z_drift - math.log(strike / spot) / sigma_tau
            p_zd = float(norm.cdf(d))
        else:
            p_zd = np.nan

        edge_zd = (p_zd - pm) if not np.isnan(p_zd) else np.nan

        records.append({
            "gate":      gate,
            "ts":        ts,
            "strike":    strike,
            "spot":      spot,
            "pm":        pm,
            "tau":       tau,
            "outcome":   outcome,
            "z_drift":   z_drift,
            "p_zd":      p_zd,
            "edge_zd":   edge_zd,
            "rv_ann":    rv_ann,
            # Signal columns for analytical assessment
            "ema_stack": row.get("ema_stack_bias", np.nan),
            "comp_rev":  row.get("composite_rev", np.nan),
            "comp_trend":row.get("composite_trend", np.nan),
            "stoch_k":   row.get("stoch_k", np.nan),
            "vwap_str":  row.get("vwap_stretch", np.nan),
            "funding":   row.get("funding_bias", np.nan),
            "comp_pup":  row.get("composite_p_up", np.nan),
        })

    df = pd.DataFrame(records)
    df_valid = df.dropna(subset=["outcome"])
    print(f"  Valid outcomes: {len(df_valid)} / {len(df)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: Per-gate statistics
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 64)
    print("PART 1 — PER-GATE SIMULATION RESULTS")
    print("=" * 64)

    gate_results = {}
    for gate in df["gate"].unique():
        gdf = df_valid[df_valid["gate"] == gate].copy()
        if len(gdf) < 3:
            continue

        n = len(gdf)
        wr_all = gdf["outcome"].mean()
        be_all = gdf["pm"].mean()
        pnl_all = sum(calc_pnl(row["pm"], int(row["outcome"])) for _, row in gdf.iterrows())

        # z_drift filter: trade if edge_zd >= EDGE_THRESHOLD
        zd_mask = gdf["edge_zd"] >= EDGE_THRESHOLD
        gdf_zd  = gdf[zd_mask]
        zd_n    = len(gdf_zd)
        zd_wr   = gdf_zd["outcome"].mean() if zd_n > 0 else np.nan
        zd_be   = gdf_zd["pm"].mean() if zd_n > 0 else np.nan
        zd_pnl  = sum(calc_pnl(row["pm"], int(row["outcome"])) for _, row in gdf_zd.iterrows()) if zd_n > 0 else 0.0
        delta   = zd_pnl - pnl_all  # positive = z_drift recovered edge

        # Verdict logic
        if n < 5:
            verdict = "too_few"
        elif zd_n == 0:
            verdict = "zdrift_agrees"  # z_drift also blocks all
        elif wr_all < be_all and zd_wr > be_all:
            verdict = "KEEP_gate"  # gate blocked bad trades, z_drift would err
        elif wr_all >= be_all and delta > 0:
            verdict = "REDUNDANT?"  # gate was blocking profitable trades AND z_drift agrees
        elif wr_all >= be_all:
            verdict = "review"
        else:
            verdict = "keep"

        gate_results[gate] = {
            "n": n, "WR": wr_all, "BE": be_all,
            "pnl_all": pnl_all,
            "zd_n": zd_n, "zd_wr": zd_wr, "zd_be": zd_be, "zd_pnl": zd_pnl,
            "delta": delta,
            "verdict": verdict,
        }

    # Print table
    hdr = (f"{'Gate':<28} | {'n':>5} | {'WR':>6} | {'BE':>6} | {'PnL_all':>8} | "
           f"{'zd_n':>5} | {'zd_WR':>6} | {'zd_PnL':>8} | {'Δ':>7} | Verdict")
    print(hdr)
    print("-" * len(hdr))
    for gate, r in sorted(gate_results.items(), key=lambda x: -abs(x[1]["pnl_all"])):
        wr_s   = f"{r['WR']*100:.1f}%" if not np.isnan(r["WR"]) else "  N/A"
        be_s   = f"{r['BE']*100:.1f}%" if not np.isnan(r["BE"]) else "  N/A"
        zdwr_s = f"{r['zd_wr']*100:.1f}%" if r["zd_n"] > 0 and not np.isnan(r["zd_wr"]) else "  N/A"
        print(
            f"{gate:<28} | {r['n']:>5} | {wr_s:>6} | {be_s:>6} | "
            f"${r['pnl_all']:>7.1f} | {r['zd_n']:>5} | {zdwr_s:>6} | "
            f"${r['zd_pnl']:>7.1f} | ${r['delta']:>6.1f} | {r['verdict']}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: Invisible gate analytical assessment
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 64)
    print("PART 2 — INVISIBLE GATE ANALYTICAL ASSESSMENT")
    print("(Gates that don't log blocks; assessed via signal overlap)")
    print("=" * 64)

    # Load paper_trades_btc15m.csv for allowed trades
    pt15 = pd.read_csv(PT15_CSV, low_memory=False)
    pt15_btc_yes = pt15[
        (pt15["asset"] == "BTC") &
        (pt15["side"] == "yes") &
        (pt15["resolved_yes"].notna())
    ].copy()
    pt15_btc_yes["resolved_yes_int"] = pt15_btc_yes["resolved_yes"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    )
    pt15_btc_yes = pt15_btc_yes.dropna(subset=["resolved_yes_int"])

    # Also add WR/PnL for gate conditions that are MET (these trades PASSED the gate)
    # near_itm_gate: pm>0.50 AND (4h RSI>62 OR 4h MACD hist>80)
    # → proxy: pm>0.50 in general (signal columns not available in 15m file, gate operates on 4h)
    # We'll note this is 4h timescale data not in our csv

    # Check what signal columns exist in pt15_btc_yes
    print(f"\n  Allowed BTC YES resolved trades: {len(pt15_btc_yes)}")
    if len(pt15_btc_yes) > 0:
        wr_overall = pt15_btc_yes["resolved_yes_int"].mean()
        pm_overall = pt15_btc_yes["p_market"].mean()
        pnl_overall = sum(
            calc_pnl(row["p_market"], int(row["resolved_yes_int"]))
            for _, row in pt15_btc_yes.iterrows()
        )
        print(f"  Overall allowed trades — WR={wr_overall*100:.1f}%, "
              f"BE={pm_overall*100:.1f}%, PnL=${pnl_overall:.1f}")

    # ── btc_vol_gate: |z_strike| > 2*vol_factor ───────────────────────────────
    print("\n  [btc_vol_gate] Reachability gate — blocks deep OTM YES when |z_strike|>2×vol_factor")
    print("  → Captures pure reachability (can price move far enough?)")
    print("  → z_drift encodes DIRECTION but not MAGNITUDE of required move")
    print("  → Independent of z_drift → KEEP")

    # ── near_itm_gate: pm>0.50 AND 4h RSI>62 or MACD hist>80 ─────────────────
    print("\n  [near_itm_gate] Block pm>0.50 YES when 4h RSI>62 OR 4h MACD hist>80")
    print("  → 4h timescale overbought filter for near-ITM YES bets")
    print("  → z_drift uses 1h contract data, not 4h momentum indicators")
    print("  → Different timescale and signal class → KEEP")

    # ── cg_fr_gate: fr_vol_1d>0 unless pm>0.60 or vpin>=1 ────────────────────
    print("\n  [cg_fr_gate] Block YES when funding rate vol>0 (funding uncertainty)")
    print("  → Positioning/sentiment data not captured by z_drift")
    print("  → Orthogonal information → KEEP")

    # ── rev_div_gate: ema=+1 AND composite_rev<=-4 AND stoch_k>55 ─────────────
    # Check in blocked_trades for ema_stack=+1, comp_rev<=-4, stoch_k>55 conditions
    cond_rev = (
        (df["ema_stack"] == 1.0) &
        (df["comp_rev"] <= -4) &
        (df["stoch_k"] > 55)
    )
    df_rev = df_valid[cond_rev[df_valid.index]]
    print(f"\n  [rev_div_gate] ema=+1, composite_rev<=-4, stoch_k>55")
    if len(df_rev) >= 3:
        wr_rev = df_rev["outcome"].mean()
        be_rev = df_rev["pm"].mean()
        pnl_rev = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in df_rev.iterrows())
        # z_drift filter
        zd_rev = df_rev[df_rev["edge_zd"] >= EDGE_THRESHOLD]
        zd_wr_rev = zd_rev["outcome"].mean() if len(zd_rev) > 0 else np.nan
        zd_pnl_rev = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in zd_rev.iterrows()) if len(zd_rev) > 0 else 0.0
        print(f"  Matching rows in blocked_trades: n={len(df_rev)}, "
              f"WR={wr_rev*100:.1f}%, BE={be_rev*100:.1f}%, PnL=${pnl_rev:.1f}")
        print(f"  z_drift would trade: {len(zd_rev)}, WR={zd_wr_rev*100:.1f}% if not NaN, PnL=${zd_pnl_rev:.1f}")
        if wr_rev < be_rev:
            print("  → Gate is blocking unprofitable trades → KEEP")
        elif len(zd_rev) > 0 and zd_wr_rev < be_rev:
            print("  → Gate aligns with z_drift on bad trades → may be REDUNDANT")
        else:
            print("  → Inconclusive — monitor")
    else:
        print(f"  Matching rows: {len(df_rev)} (insufficient for analysis)")
    print("  → composite_rev captures mean-reversion signal not in z_drift → lean KEEP")

    # ── cg_oi_stable_yes_gate: oi_stable_pct_4h>2% AND pm<0.50 ───────────────
    print("\n  [cg_oi_stable_yes_gate] Block YES when OI stable 4h>2% and pm<0.50 (OTM)")
    print("  → 4h OI stability = longs trapped/crowded at OTM level")
    print("  → Not captured by z_drift (OI data) → KEEP")

    # ── neutral_ema_g1: ema=0 AND comp_p_up>=0.60 AND stoch_k<40 ─────────────
    cond_g1 = (
        (df["ema_stack"] == 0) &
        (df["comp_pup"] >= 0.60) &
        (df["stoch_k"] < 40)
    )
    df_g1 = df_valid[cond_g1[df_valid.index]]
    print(f"\n  [neutral_ema_g1] ema=0, comp_p_up>=0.60, stoch_k<40")
    if len(df_g1) >= 3:
        wr_g1 = df_g1["outcome"].mean()
        be_g1 = df_g1["pm"].mean()
        pnl_g1 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in df_g1.iterrows())
        zd_g1 = df_g1[df_g1["edge_zd"] >= EDGE_THRESHOLD]
        zd_wr_g1 = zd_g1["outcome"].mean() if len(zd_g1) > 0 else np.nan
        zd_pnl_g1 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in zd_g1.iterrows()) if len(zd_g1) > 0 else 0.0
        print(f"  Matching rows: n={len(df_g1)}, WR={wr_g1*100:.1f}%, BE={be_g1*100:.1f}%, PnL=${pnl_g1:.1f}")
        print(f"  z_drift would trade: {len(zd_g1)}, PnL=${zd_pnl_g1:.1f}")
        if wr_g1 < be_g1:
            print("  → Gate blocking bad trades → KEEP")
        else:
            print("  → Gate possibly blocking good trades → investigate")
    else:
        print(f"  Matching rows in blocked_trades: {len(df_g1)} (insufficient)")
    print("  → stoch_k<40 in neutral EMA = momentum reversal risk → lean KEEP")

    # ── neutral_ema_g2: ema=0 AND vwap=-1 AND pm<0.60 ─────────────────────────
    cond_g2 = (
        (df["ema_stack"] == 0) &
        (df["vwap_str"] == -1) &
        (df["pm"] < 0.60)
    )
    df_g2 = df_valid[cond_g2[df_valid.index]]
    print(f"\n  [neutral_ema_g2] ema=0, vwap_stretch=-1, pm<0.60")
    if len(df_g2) >= 3:
        wr_g2 = df_g2["outcome"].mean()
        be_g2 = df_g2["pm"].mean()
        pnl_g2 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in df_g2.iterrows())
        zd_g2 = df_g2[df_g2["edge_zd"] >= EDGE_THRESHOLD]
        zd_pnl_g2 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in zd_g2.iterrows()) if len(zd_g2) > 0 else 0.0
        print(f"  Matching rows: n={len(df_g2)}, WR={wr_g2*100:.1f}%, BE={be_g2*100:.1f}%, PnL=${pnl_g2:.1f}")
        if wr_g2 < be_g2:
            print("  → Gate blocking bad trades → KEEP")
        else:
            print("  → Gate possibly blocking good trades → investigate")
    else:
        print(f"  Matching rows: {len(df_g2)} (insufficient)")
    print("  → VWAP bearish with neutral EMA = weak structure → lean KEEP")

    # ── neutral_ema_g3: ema=0 AND comp_trend=-1 ───────────────────────────────
    cond_g3 = (
        (df["ema_stack"] == 0) &
        (df["comp_trend"] == -1)
    )
    df_g3 = df_valid[cond_g3[df_valid.index]]
    print(f"\n  [neutral_ema_g3] ema=0, comp_trend=-1")
    if len(df_g3) >= 3:
        wr_g3 = df_g3["outcome"].mean()
        be_g3 = df_g3["pm"].mean()
        pnl_g3 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in df_g3.iterrows())
        zd_g3 = df_g3[df_g3["edge_zd"] >= EDGE_THRESHOLD]
        zd_pnl_g3 = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in zd_g3.iterrows()) if len(zd_g3) > 0 else 0.0
        print(f"  Matching rows: n={len(df_g3)}, WR={wr_g3*100:.1f}%, BE={be_g3*100:.1f}%, PnL=${pnl_g3:.1f}")
        if wr_g3 < be_g3:
            print("  → Gate blocking bad trades → KEEP")
        else:
            print("  → Review needed")
    else:
        print(f"  Matching rows: {len(df_g3)} (insufficient)")
    print("  → comp_trend captures multi-indicator bear consensus → z_drift partial overlap → KEEP")

    # ── smc_gate analytical (in addition to Part 1) ───────────────────────────
    print(f"\n  [smc_gate] 4h bearish structure + OTM YES pm<0.35 (already in Part 1)")
    print("  → SMC uses 4h demand/supply zones not in z_drift → complementary data")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: z_drift alignment analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 64)
    print("PART 3 — z_drift ALIGNMENT ANALYSIS")
    print("(Gate blocks: where z_drift agrees vs disagrees)")
    print("=" * 64)

    # z_drift "would block" = edge_zd < EDGE_THRESHOLD (wouldn't trade)
    df_v = df_valid.copy()
    df_v["zdrift_blocks"] = df_v["edge_zd"] < EDGE_THRESHOLD

    print(f"\n  Total BTC YES blocked trades (with valid outcomes): {len(df_v)}")
    both_block   = df_v[df_v["zdrift_blocks"]]
    gate_only    = df_v[~df_v["zdrift_blocks"]]

    print(f"\n  z_drift AGREES (both block):     {len(both_block):>5} ({len(both_block)/len(df_v)*100:.1f}%)")
    print(f"  z_drift DISAGREES (gate blocks, z_drift would trade): {len(gate_only):>5} ({len(gate_only)/len(df_v)*100:.1f}%)")

    # For the gate-only group: would z_drift have been right?
    if len(gate_only) > 0:
        wr_go = gate_only["outcome"].mean()
        be_go = gate_only["pm"].mean()
        pnl_go = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in gate_only.iterrows())
        print(f"\n  Trades where GATE blocks but z_drift APPROVES:")
        print(f"    WR={wr_go*100:.1f}% vs BE={be_go*100:.1f}%, PnL if z_drift traded=${pnl_go:.1f}")
        if wr_go < be_go:
            print(f"    → Gates are RIGHT: blocking genuinely bad trades z_drift would miss")
        elif pnl_go > 0:
            print(f"    → POTENTIAL: gates may be blocking ${pnl_go:.1f} in profitable trades")
        else:
            print(f"    → Mixed signal")

    # Per-gate alignment breakdown for top gates
    print(f"\n  Per-gate alignment (top gates by volume):")
    print(f"  {'Gate':<28} | {'agree%':>7} | {'gate-only_n':>11} | {'zd-would-WR':>11} | {'zd-would-PnL':>12}")
    print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*11}-+-{'-'*11}-+-{'-'*12}")
    for gate in sorted(gate_results.keys(), key=lambda g: -gate_results[g]["n"]):
        gdf = df_v[df_v["gate"] == gate]
        if len(gdf) < 3:
            continue
        agree = gdf["zdrift_blocks"].sum()
        disagree = (~gdf["zdrift_blocks"]).sum()
        agree_pct = agree / len(gdf) * 100
        g_only = gdf[~gdf["zdrift_blocks"]]
        wr_zo = g_only["outcome"].mean() if len(g_only) > 0 else np.nan
        pnl_zo = sum(calc_pnl(r["pm"], int(r["outcome"])) for _, r in g_only.iterrows()) if len(g_only) > 0 else 0.0
        wr_s = f"{wr_zo*100:.1f}%" if not np.isnan(wr_zo) else "N/A"
        print(f"  {gate:<28} | {agree_pct:>6.1f}% | {disagree:>11} | {wr_s:>11} | ${pnl_zo:>10.1f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 64)
    print("SUMMARY — Gate verdicts and z_drift interaction")
    print("=" * 64)
    print("""
Logged gate verdicts:
""")
    for gate, r in sorted(gate_results.items(), key=lambda x: -abs(x[1]["pnl_all"])):
        wr_s = f"{r['WR']*100:.1f}%" if not np.isnan(r["WR"]) else "N/A"
        be_s = f"{r['BE']*100:.1f}%" if not np.isnan(r["BE"]) else "N/A"
        print(f"  {gate:<28} n={r['n']:>5} WR={wr_s} vs BE={be_s}  PnL_if_traded=${r['pnl_all']:>8.1f}  → {r['verdict']}")

    print("""
Invisible gate verdicts (analytical):
  btc_vol_gate         — reachability (magnitude), not in z_drift → KEEP
  near_itm_gate        — 4h RSI/MACD (different timescale) → KEEP
  cg_fr_gate           — funding rate (positioning data) → KEEP
  rev_div_gate         — composite_rev divergence (not in z_drift) → KEEP (pending live data)
  cg_oi_stable_yes_gate— 4h OI stability (not in z_drift) → KEEP
  neutral_ema_g1/g2/g3 — neutral EMA regime filters → KEEP (partial z_drift overlap)
""")

    print("Done.")


if __name__ == "__main__":
    main()
