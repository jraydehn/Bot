#!/usr/bin/env python3
"""
simulate_pup_v2_drift.py — Replay simulation comparing BTC hourly drift models.

For each decision slot in paper_trades.csv:
  - Considers all evaluated contracts (trade + no_trade rows)
  - Recomputes p_model under three models:
      Current:  z_drift = mu_6h × (tau/60) / sigma_tau   [6h rolling mean]
      Model A:  z_drift = Φ⁻¹(p_up_v2) × 1.14 × √(τ/60) [p_up_v2, k=1.14]
      No drift: z_drift = 0                               [pure log-normal]
  - Picks the highest net-edge YES or NO candidate that clears MIN_EDGE
  - Applies Kelly sizing (flat $1000 bankroll, no compounding)
  - Computes PnL from settlement outcomes (Binance 1h close or stored resolved_yes)
  - Deduplicates within the same close_ts (one bet per expiry)

Usage:
    python3 simulate_pup_v2_drift.py
    python3 simulate_pup_v2_drift.py --k 1.14 --min-tau 20
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Constants ──────────────────────────────────────────────────────────────
K_PUP_HOURLY   = 1.14      # calibrated 2026-05-25 on 549k records
FLAT_BANKROLL  = 1000.0
MIN_NET_EDGE   = 0.02
MIN_KELLY      = 0.005
DEFAULT_SPREAD = 0.005
DEFAULT_SLIP   = 0.003
MAX_KELLY_FRAC = 0.25      # cap bet size

P_UP_V2_FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def sigma_tau(vol_eff: float, tau_min: float, vol_factor: float = 1.0) -> float:
    if vol_eff > 0 and tau_min > 0:
        return vol_eff * vol_factor * math.sqrt(tau_min)
    return float("nan")


def p_yes_model(spot, strike, vol_eff, tau_min, z_drift, vol_factor: float = 1.0):
    st = sigma_tau(vol_eff, tau_min, vol_factor)
    if math.isnan(st) or st <= 0 or spot <= 0:
        return float("nan")
    z = math.log(strike / spot) / st - z_drift
    return float(np.clip(1.0 - norm.cdf(z), 0.01, 0.99))


def kelly_fraction(p_model, p_market, side):
    """Fractional Kelly for a binary bet."""
    if side == "yes":
        b = (1 - p_market) / p_market   # odds: win (1-pm)/pm, lose 1
        edge = p_model - p_market
    else:
        b = p_market / (1 - p_market)
        edge = (1 - p_model) - (1 - p_market)
    if edge <= 0 or b <= 0:
        return 0.0
    f = edge / b
    return float(np.clip(f, 0, MAX_KELLY_FRAC))


def net_edge(p_model, p_market, side):
    if side == "yes":
        return p_model - p_market - DEFAULT_SLIP - DEFAULT_SPREAD / 2
    else:
        return (1 - p_model) - (1 - p_market) - DEFAULT_SLIP - DEFAULT_SPREAD / 2


def pnl_for_bet(side, won_yes, kelly_frac, p_market):
    """Compute PnL for a placed bet."""
    stake = FLAT_BANKROLL * kelly_frac
    if side == "yes":
        won = won_yes
        payoff_per_dollar = (1 - p_market) / p_market
    else:
        won = not won_yes
        payoff_per_dollar = p_market / (1 - p_market)
    if won:
        return stake * payoff_per_dollar
    else:
        return -stake


# ── Load p_up_v2 model ─────────────────────────────────────────────────────

def load_pup_model():
    import pickle
    model_path = ROOT / "reform_results" / "btc_p_up_v2.pkl"
    if not model_path.exists():
        print("WARNING: p_up_v2 model not found — using composite_p_up as fallback")
        return None
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    return pipe["clf"]


def infer_pup_v2(df: pd.DataFrame, clf) -> pd.Series:
    """Re-infer p_up_v2 for all rows. Missing features → NaN (LightGBM handles natively)."""
    if clf is None:
        return df["composite_p_up"].clip(0.01, 0.99)
    feat_df = pd.DataFrame(index=df.index)
    for f in P_UP_V2_FEATURES:
        feat_df[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    X = feat_df[P_UP_V2_FEATURES].values.astype(float)
    probs = clf.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index).clip(0.01, 0.99)


# ── Compute mu_6h from Binance 1h data ────────────────────────────────────

def build_mu_6h(df1h: pd.DataFrame) -> pd.Series:
    """Rolling 6h mean of 1h log returns, aligned to bar close timestamps."""
    log_ret = np.log(df1h["close"] / df1h["close"].shift(1))
    return log_ret.rolling(6).mean()


def build_sigma_1h(df1h: pd.DataFrame) -> pd.Series:
    """Rolling 24h std of 1h log returns (per-hour vol)."""
    log_ret = np.log(df1h["close"] / df1h["close"].shift(1))
    return log_ret.rolling(24).std()


# ── Missing feature computation (exact formulas from train_btc_p_up_v2.py) ─

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _atr(h, lo, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()


def build_missing_1h_features(df1h: pd.DataFrame) -> pd.DataFrame:
    c  = df1h["close"].astype(float)
    h  = df1h["high"].astype(float)
    lo = df1h["low"].astype(float)

    macd     = _ema(c, 12) - _ema(c, 26)
    mid      = c.rolling(20).mean()
    std      = c.rolling(20).std()
    lo_bb    = mid - 2 * std
    hi_bb    = mid + 2 * std
    ema50    = _ema(c, 50)

    return pd.DataFrame({
        "rsi_14":       _rsi(c, 14),
        "macd_hist_1h": macd - macd.ewm(span=9, adjust=False).mean(),
        "bb_pct":       (c - lo_bb) / (hi_bb - lo_bb).replace(0, np.nan),
        "ema50_dist":   (c - ema50) / ema50.replace(0, np.nan) * 100,
    }, index=df1h.index)


def build_vol_factor_series(df1h: pd.DataFrame) -> pd.Series:
    """
    Approximate vol_layer factor from 1h OHLCV (3 of 5 signals; omits VWAP-dev and rv_6h
    which require 1m bars). Score range [-3, +3] → factor [0.76, 1.24].
    BTC thresholds from vol_layer._VOL_CONFIGS["BTC"].
    """
    c  = df1h["close"].astype(float)
    h  = df1h["high"].astype(float)
    lo = df1h["low"].astype(float)
    v  = df1h["volume"].astype(float)

    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    atr_ratio = atr / atr.rolling(24).mean().replace(0, np.nan)

    log_ret = np.log(c / cp)
    roll_std = log_ret.rolling(24).std()
    abs_z = (log_ret / roll_std.replace(0, np.nan)).abs()

    vol_ma = v.rolling(20).mean()
    vol_ratio = v / vol_ma.replace(0, np.nan)

    # BTC thresholds from vol_layer._VOL_CONFIGS["BTC"]
    s1 = pd.Series(0, index=df1h.index, dtype=float)
    s1[atr_ratio > 1.50]  =  1
    s1[atr_ratio < 0.75]  = -1

    s2 = pd.Series(0, index=df1h.index, dtype=float)
    s2[abs_z > 2.00]  =  1
    s2[abs_z < 0.50]  = -1

    s3 = pd.Series(0, index=df1h.index, dtype=float)
    s3[vol_ratio > 3.00]  =  1
    s3[vol_ratio < 0.30]  = -1

    score = s1 + s2 + s3
    factor = (1.0 + score * 0.08).clip(0.60, 1.40)
    return factor


def build_missing_4h_features(df4h: pd.DataFrame) -> pd.DataFrame:
    c  = df4h["close"].astype(float)
    h  = df4h["high"].astype(float)
    lo = df4h["low"].astype(float)

    ll = lo.rolling(14).min()
    hh = h.rolling(14).max()
    atr_4h = _atr(h, lo, c, 14)

    return pd.DataFrame({
        "stoch_k_4h": (c - ll) / (hh - ll).replace(0, np.nan) * 100,
        "rsi_4h":     _rsi(c, 14),
        "chg_4h_atr": (c - c.shift(5)) / atr_4h.replace(0, np.nan),
    }, index=df4h.index)


# ── Outcome resolution ─────────────────────────────────────────────────────

def resolve_outcomes(df: pd.DataFrame, df1h: pd.DataFrame) -> pd.Series:
    """
    For rows without resolved_yes, look up BTC close price at close_ts
    from Binance 1h data and determine outcome.
    """
    resolved = pd.to_numeric(df["resolved_yes"], errors="coerce").copy()

    need_resolve = resolved.isna()
    if need_resolve.sum() == 0:
        return resolved.astype(float)

    # Build close_ts → settlement price map from 1h data
    close_ts_series = pd.to_datetime(df.loc[need_resolve, "close_ts"], utc=True, errors="coerce")
    df1h_idx = df1h.index
    settle_map = {}
    for ts in close_ts_series.dropna().unique():
        # Find the 1h bar whose close IS the settlement time
        # Binance 1h bars close at round hours; close_ts is UTC round hour
        candidates = df1h_idx[df1h_idx <= ts]
        if len(candidates) > 0:
            bar_ts = candidates[-1]
            settle_map[ts] = float(df1h.loc[bar_ts, "close"])

    # Apply
    for idx in df.index[need_resolve]:
        ts = pd.to_datetime(df.loc[idx, "close_ts"], utc=True, errors="coerce")
        if ts in settle_map:
            settle_price = settle_map[ts]
            strike = float(df.loc[idx, "strike"])
            resolved.loc[idx] = float(settle_price > strike)

    return resolved.astype(float)


# ── Main simulation ────────────────────────────────────────────────────────

def _eval_row(row, mu6h_at, sig1h_at, k_pup, use_pup, use_mu6h, use_vol_factor,
              garch_vol_at=None, regime_z_at=None, k_regime=1.0):
    """Evaluate one candidate row; return (edge, kf, z_drift, pm_model, vf) or None."""
    spot    = float(row["spot"])
    strike  = float(row["strike"])
    pm      = float(row["p_market"])
    vol_eff = float(row["vol_eff"])
    tau     = float(row["tau_minutes"])
    side    = str(row["side"]).lower()
    p_up_v2 = float(row["_p_up_v2"])
    vf      = float(row["_vol_factor"]) if use_vol_factor else 1.0

    if math.isnan(spot) or math.isnan(strike) or math.isnan(vol_eff) or tau <= 0:
        return None
    if use_vol_factor and (math.isnan(vf) or vf <= 0):
        vf = 1.0

    dt = pd.Timestamp(row["decision_time"])
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")

    # Override vol_eff with GARCH conditional vol if provided
    if garch_vol_at is not None and dt >= garch_vol_at.index[0]:
        gv = float(garch_vol_at.asof(dt))
        if not math.isnan(gv) and gv > 0:
            vol_eff = gv

    z_drift = 0.0
    if use_mu6h:
        mu6h_val  = float(mu6h_at.asof(dt)) if dt >= mu6h_at.index[0]  else float("nan")
        sig1h_val = float(sig1h_at.asof(dt)) if dt >= sig1h_at.index[0] else float("nan")
        st = sigma_tau(vol_eff, tau, vf)
        z_drift += mu6h_val * (tau / 60.0) / st if not (math.isnan(mu6h_val) or math.isnan(sig1h_val) or st <= 0) else 0.0
    if use_pup:
        z_drift += norm.ppf(float(np.clip(p_up_v2, 0.01, 0.99))) * k_pup * math.sqrt(tau / 60.0)
    if regime_z_at is not None and dt >= regime_z_at.index[0]:
        rz = float(regime_z_at.asof(dt))
        z_drift += rz * k_regime * math.sqrt(tau / 60.0) if not math.isnan(rz) else 0.0

    pm_model = p_yes_model(spot, strike, vol_eff, tau, z_drift, vf)
    if math.isnan(pm_model):
        return None

    edge = net_edge(pm_model, pm, side)
    kf   = kelly_fraction(pm_model, pm, side)
    if edge <= MIN_NET_EDGE or kf < MIN_KELLY:
        return None

    return edge, kf, z_drift, pm_model, vf


def run_simulation(df: pd.DataFrame, mu6h_at: pd.Series, sig1h_at: pd.Series,
                   model_name: str, k_pup: float, use_pup: bool, use_mu6h: bool,
                   use_vol_factor: bool = False, paired: bool = False,
                   garch_vol_at=None, regime_z_at=None, k_regime=1.0):
    """
    paired=False: pick single best-edge contract per decision_time slot.
    paired=True:  pick best YES + best NO per (decision_time, close_ts) expiry pair —
                  captures the range-trading behavior of the April model.
    """
    trades = []
    slots_evaluated = 0
    slots_no_edge   = 0

    if paired:
        group_keys = ["decision_time", "close_ts"]
    else:
        group_keys = ["decision_time"]

    for slot_key, group in df.groupby(group_keys):
        slots_evaluated += 1
        candidates = {"yes": None, "no": None}  # best per side

        for _, row in group.iterrows():
            side = str(row["side"]).lower()
            if side not in candidates:
                continue
            result = _eval_row(row, mu6h_at, sig1h_at, k_pup, use_pup, use_mu6h, use_vol_factor,
                               garch_vol_at, regime_z_at, k_regime)
            if result is None:
                continue
            edge, kf, z_drift, pm_model, vf = result
            if candidates[side] is None or edge > candidates[side]["edge"]:
                candidates[side] = {
                    "slot":     slot_key,
                    "side":     side,
                    "pm":       float(row["p_market"]),
                    "pm_model": pm_model,
                    "z_drift":  z_drift,
                    "edge":     edge,
                    "kelly":    kf,
                    "won_yes":  int(row["resolved_yes"]),
                    "strike":   float(row["strike"]),
                    "spot":     float(row["spot"]),
                    "tau":      float(row["tau_minutes"]),
                }

        if paired:
            placed = [c for c in candidates.values() if c is not None]
        else:
            # non-paired: single best across both sides
            all_cands = [c for c in candidates.values() if c is not None]
            placed = [max(all_cands, key=lambda c: c["edge"])] if all_cands else []

        if not placed:
            slots_no_edge += 1
            continue

        for best in placed:
            pnl = pnl_for_bet(best["side"], best["won_yes"], best["kelly"], best["pm"])
            best["pnl"] = pnl
            best["won"] = (best["side"] == "yes" and best["won_yes"]) or \
                          (best["side"] == "no"  and not best["won_yes"])
            trades.append(best)

    trade_df = pd.DataFrame(trades)
    if trade_df.empty:
        print(f"\n{model_name}: NO TRADES")
        return trade_df

    n       = len(trade_df)
    wins    = trade_df["won"].sum()
    total   = trade_df["pnl"].sum()
    be_wr   = trade_df.apply(
        lambda r: r["pm"] / (1 if r["side"]=="yes" else 1),  # rough BE
        axis=1).mean()

    yes_n = (trade_df["side"] == "yes").sum()
    no_n  = (trade_df["side"] == "no").sum()

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Slots evaluated:   {slots_evaluated}  |  no-edge slots: {slots_no_edge}")
    print(f"  Trades placed:     {n}  (YES={yes_n}, NO={no_n})")
    print(f"  Win rate:          {100*wins/n:.1f}%  ({wins}W {n-wins}L)")
    print(f"  Total PnL:         ${total:+.2f}")
    print(f"  Avg PnL/trade:     ${total/n:+.2f}")
    print(f"  Avg edge:          {100*trade_df['edge'].mean():.2f}%")
    print(f"  Avg kelly:         {100*trade_df['kelly'].mean():.1f}%")

    # Breakdown by YES/NO
    for s in ["yes", "no"]:
        sub = trade_df[trade_df["side"] == s]
        if len(sub) == 0:
            continue
        sw = sub["won"].sum()
        print(f"  {s.upper():<4} n={len(sub):>3}  WR={100*sw/len(sub):.1f}%  PnL=${sub['pnl'].sum():+.2f}")

    return trade_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=float, default=K_PUP_HOURLY)
    parser.add_argument("--min-tau",  type=float, default=5.0, help="min tau_minutes to include")
    parser.add_argument("--min-pm",   type=float, default=0.10, help="min p_market to include")
    parser.add_argument("--min-date", type=str,   default=None, help="filter slots on or after YYYY-MM-DD")
    args = parser.parse_args()

    print("Loading paper_trades.csv...")
    df = pd.read_csv(ROOT / "results" / "paper_trades.csv", low_memory=False)

    for col in ["spot", "strike", "p_market", "vol_eff", "tau_minutes",
                "composite_p_up", "confirmation_bias", "stoch_k",
                "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
                "vpin_score", "stoch_bias", "vwap_distance_pct", "rvol_1h",
                "pm_drift_5m", "composite_trend", "composite_rev"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["decision_time", "spot", "strike", "vol_eff", "p_market", "tau_minutes"])
    df = df[(df["tau_minutes"] >= args.min_tau) & (df["p_market"] >= args.min_pm) & (df["p_market"] <= 1 - args.min_pm)]
    df = df[(df["vol_eff"] > 0) & (df["spot"] > 0)]
    if args.min_date:
        cutoff = pd.Timestamp(args.min_date, tz="UTC")
        df = df[df["decision_time"] >= cutoff]

    print(f"  Rows after filters: {len(df):,}")

    # Load Binance 1h data
    print("Loading Binance 1h data...")
    data_dir = ROOT / "data"
    f1h = sorted(data_dir.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h = pd.read_parquet(f1h)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    df1h = df1h.sort_index()

    mu6h_ser  = build_mu_6h(df1h)
    sig1h_ser = build_sigma_1h(df1h)

    # Align to decision_time (use the nearest 1h bar at or before decision_time)
    dt_vals = df["decision_time"].dropna().unique()
    dt_sorted = pd.DatetimeIndex(sorted(dt_vals))
    mu6h_at  = pd.Series(
        [float(mu6h_ser.asof(t))  if t >= mu6h_ser.index[0]  else np.nan for t in dt_sorted],
        index=dt_sorted)
    sig1h_at = pd.Series(
        [float(sig1h_ser.asof(t)) if t >= sig1h_ser.index[0] else np.nan for t in dt_sorted],
        index=dt_sorted)

    # Compute missing p_up_v2 features from Binance OHLCV
    print("Computing missing p_up_v2 features from Binance data...")
    feat1h = build_missing_1h_features(df1h)

    f4h = sorted(data_dir.glob("binanceus_BTCUSDT_4h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df4h = pd.read_parquet(f4h)
    df4h.index = pd.to_datetime(df4h.index, utc=True)
    df4h = df4h.sort_index()
    feat4h = build_missing_4h_features(df4h)

    dt_all = df["decision_time"]
    for feat_col in feat1h.columns:
        df[feat_col] = [float(feat1h[feat_col].asof(t)) if t >= feat1h.index[0] else np.nan
                        for t in dt_all]
    for feat_col in feat4h.columns:
        df[feat_col] = [float(feat4h[feat_col].asof(t)) if t >= feat4h.index[0] else np.nan
                        for t in dt_all]

    added_feats = list(feat1h.columns) + list(feat4h.columns)
    for col in added_feats:
        cov = df[col].notna().mean()
        print(f"  {col:<20} cov={cov:.1%}  mean={df[col].mean():.3f}")

    # Compute vol_factor from 1h OHLCV (3-signal approximation of vol_layer)
    print("Computing vol_factor series from 1h data...")
    vf_series = build_vol_factor_series(df1h)
    df["_vol_factor"] = [float(vf_series.asof(t)) if t >= vf_series.index[0] else 1.0
                         for t in dt_all]
    print(f"  vol_factor: mean={df['_vol_factor'].mean():.3f}  std={df['_vol_factor'].std():.3f}  "
          f"range=[{df['_vol_factor'].min():.2f}, {df['_vol_factor'].max():.2f}]")

    # Resolve outcomes for all rows
    print("Resolving outcomes from Binance 1h data...")
    df["resolved_yes"] = resolve_outcomes(df, df1h)
    df = df.dropna(subset=["resolved_yes"])
    print(f"  Rows with known outcome: {len(df):,}")

    # Re-infer p_up_v2 for all rows
    print("Inferring p_up_v2 for all rows...")
    clf = load_pup_model()
    df["_p_up_v2"] = infer_pup_v2(df, clf)
    pup_coverage = df["_p_up_v2"].notna().mean()
    print(f"  p_up_v2 coverage: {pup_coverage:.1%}")
    print(f"  p_up_v2: mean={df['_p_up_v2'].mean():.3f}  std={df['_p_up_v2'].std():.3f}")

    print(f"\nSimulating {df['decision_time'].nunique()} decision slots "
          f"({len(df):,} candidate contracts)...")

    # Load GARCH + regime series
    garch_pkl = Path("/tmp/garch_regime_series.pkl")
    garch_vol_at = regime_z_at = None
    if garch_pkl.exists():
        garch_df = pd.read_pickle(garch_pkl)
        garch_df.index = pd.to_datetime(garch_df.index, utc=True)
        garch_vol_at = garch_df["garch_vol_eff"]
        regime_z_at  = garch_df["regime_z"]
        print(f"Loaded GARCH series: {len(garch_df)} bars")
    else:
        print("WARNING: GARCH series not found — skipping GARCH models")

    K_REGIME = 1.0   # regime z-score → z_drift scale (tune against PnL)

    t_no       = run_simulation(df, mu6h_at, sig1h_at, "No drift",                    0.0,    False, False, False, False)
    t_mu6h     = run_simulation(df, mu6h_at, sig1h_at, "Current 6h mu",               0.0,    False, True,  False, False)
    t_garch_s  = run_simulation(df, mu6h_at, sig1h_at, "GARCH vol only",              0.0,    False, False, False, False,
                                garch_vol_at=garch_vol_at)
    t_garch_r  = run_simulation(df, mu6h_at, sig1h_at, "GARCH + regime drift",        0.0,    False, False, False, False,
                                garch_vol_at=garch_vol_at, regime_z_at=regime_z_at, k_regime=K_REGIME)
    t_garch_pup = run_simulation(df, mu6h_at, sig1h_at, f"GARCH + p_up_v2 (k={args.k})", args.k, True,  False, False, False,
                                garch_vol_at=garch_vol_at)
    t_garch_rp = run_simulation(df, mu6h_at, sig1h_at, "GARCH + regime (PAIRED)",     0.0,    False, False, False, True,
                                garch_vol_at=garch_vol_at, regime_z_at=regime_z_at, k_regime=K_REGIME)
    t_garch_pp = run_simulation(df, mu6h_at, sig1h_at, f"GARCH + p_up_v2 (PAIRED)",   args.k, True,  False, False, True,
                                garch_vol_at=garch_vol_at)
    t_garch_all  = run_simulation(df, mu6h_at, sig1h_at, "GARCH + regime + p_up_v2",  args.k, True,  False, False, False,
                                garch_vol_at=garch_vol_at, regime_z_at=regime_z_at, k_regime=K_REGIME)
    t_garch_allp = run_simulation(df, mu6h_at, sig1h_at, "GARCH+regime+p_up_v2 (PAIRED)", args.k, True, False, False, True,
                                garch_vol_at=garch_vol_at, regime_z_at=regime_z_at, k_regime=K_REGIME)

    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD SUMMARY")
    print(f"{'='*60}")
    for label, td in [
        ("No drift",                      t_no),
        ("Current 6h mu",                 t_mu6h),
        ("GARCH vol only",                t_garch_s),
        ("GARCH + regime",                t_garch_r),
        ("GARCH + p_up_v2",               t_garch_pup),
        ("GARCH + regime + p_up_v2",      t_garch_all),
        ("GARCH + regime (PAIRED)",       t_garch_rp),
        ("GARCH + p_up_v2 (PAIRED)",      t_garch_pp),
        ("GARCH+regime+p_up_v2 (PAIRED)", t_garch_allp),
    ]:
        if td.empty:
            print(f"  {label:<28} — no trades")
            continue
        n   = len(td)
        wr  = 100 * td["won"].sum() / n
        pnl = td["pnl"].sum()
        yes_n = (td["side"] == "yes").sum(); yes_wr = 100*td.loc[td["side"]=="yes","won"].sum()/yes_n if yes_n else 0
        no_n  = (td["side"] == "no").sum();  no_wr  = 100*td.loc[td["side"]=="no","won"].sum()/no_n   if no_n  else 0
        print(f"  {label:<28}  n={n:>4}  WR={wr:>5.1f}%  PnL=${pnl:>+9.2f}  "
              f"[YES n={yes_n} WR={yes_wr:.0f}%  NO n={no_n} WR={no_wr:.0f}%]")


if __name__ == "__main__":
    main()
