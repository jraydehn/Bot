"""
eth_no_drift_sim.py — ETH NO-side K_drift parameter sweep via synthetic Kalshi contracts.

Methodology
-----------
Generates the full synthetic contract universe at each hourly tick (Jan–Apr 2026),
avoiding the selection bias of testing against trades the direct model chose to take.

At each tick T:
  1. Composite scorer runs once on the full dataset → causal trend/rev time-series.
  2. Sigma_tau computed from 60-bar rolling 1m realized vol × sqrt(60).
  3. Contracts generated at z-levels {±0.25 … ±2.0} rounded to nearest $1.
  4. Synthetic pm_market = neutral BSM (zero drift) + 1¢ spread — unbiased market proxy.
  5. Log-drift NO model tested at K ∈ {0.0, 0.10, 0.15, 0.20, 0.25, 0.30}.
  6. Trade taken if net_edge > MIN_EDGE and |z| passes gate conditions.
  7. Outcome resolved from actual ETH price at T+60min.

Flat $1 stake per trade — all K values are directly comparable.

Caveats
-------
- pm_market is synthetic (BSM neutral), not actual Kalshi market prices.
- Kalshi's real market maker applies its own drift/spread; this simulation is an
  approximation. K values that look best here may not be exactly right live, but
  the RELATIVE ranking across K values is meaningful.
- Score look-ahead: compute_scores() uses rolling causal indicators on the full
  dataset — no future data leaks into scores at time T.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import compute_scores, lookup_p_up

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Simulation parameters ──────────────────────────────────────────────────
SIM_START  = "2026-01-15"
SIM_END    = "2026-04-15"
K_VALUES   = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

MIN_EDGE   = 0.010   # minimum net edge to take a trade
SPREAD     = 0.010   # half-spread: ask = BSM + SPREAD, bid = BSM - SPREAD
Z_MIN      = 0.45    # |z| floor — mirrors btc_no_z_gate (near-ATM has no structural edge)
Z_MAX_MULT = 2.0     # |z| ceiling multiplier × vol_factor
VOL_FACTOR = 1.0     # simplified (live uses regime-based); 1.0 = conservative
Z_LEVELS   = [-1.5, -1.0, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]


def _load_data() -> tuple:
    df1m = pd.read_parquet(DATA_DIR / "binanceus_ETHUSDT_1m_2024-01-01_2026-05-08.parquet")
    df1h = pd.read_parquet(DATA_DIR / "binanceus_ETHUSDT_1h_2024-01-01_2026-05-08.parquet")
    df4h = pd.read_parquet(DATA_DIR / "binanceus_ETHUSDT_4h_2024-01-01_2026-05-08.parquet")
    df15m = df1m.resample("15min", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return df1m, df1h, df4h, df15m


def _precompute_scores(df1m, df1h, df4h, df15m) -> tuple:
    """Run composite scorer once on the full dataset — O(n) vectorized, no per-tick cost."""
    print("  [sim] Pre-computing composite scores on full ETH dataset …", flush=True)
    trend_s, rev_s = compute_scores(
        df1h["close"], df1h["high"], df1h["low"], df1h["volume"],
        df4h["close"], df4h["high"], df4h["low"], df4h["volume"],
        df15m["close"], df15m["high"], df15m["low"],
        df1m["close"], df1m["volume"],
        df1h.index,
    )
    print(f"  [sim] Scores ready for {len(trend_s)} 1h bars.", flush=True)
    return trend_s.astype(int), rev_s.astype(int)


def _sigma_tau(df1m: pd.DataFrame, T: pd.Timestamp) -> float:
    """1-hour-ahead realized vol from last 60 1m returns."""
    slice_1m = df1m.loc[:T]
    if len(slice_1m) < 61:
        return 0.0
    lr = np.log(slice_1m["close"] / slice_1m["close"].shift(1))
    vol = float(lr.rolling(60).std().iloc[-1])
    return vol * math.sqrt(60) if vol > 0 else 0.0


def _bsm_yes(spot: float, strike: float, sigma_tau: float) -> float:
    """Neutral (zero-drift) log-normal YES probability."""
    if sigma_tau <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    z = math.log(strike / spot) / sigma_tau
    return float(norm.cdf(-z))


def _p_no_logdrift(spot: float, strike: float, sigma_tau: float,
                   p_up: float, K: float) -> float:
    """Log-drift NO probability: Φ(z_strike − Φ⁻¹(p_up) × K)."""
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift  = norm.ppf(max(0.01, min(0.99, p_up))) * K
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.01, 0.99))


def run_simulation() -> pd.DataFrame:
    df1m, df1h, df4h, df15m = _load_data()
    trend_s, rev_s = _precompute_scores(df1m, df1h, df4h, df15m)

    # Pre-compute sigma_tau series from 1m vol (efficient rolling)
    lr_1m = np.log(df1m["close"] / df1m["close"].shift(1))
    vol60_1m = lr_1m.rolling(60).std() * math.sqrt(60)

    sim_start = pd.Timestamp(SIM_START, tz="UTC")
    sim_end   = pd.Timestamp(SIM_END,   tz="UTC")
    sim_ticks = df1h.index[(df1h.index >= sim_start) & (df1h.index <= sim_end)]
    print(f"  [sim] Simulation ticks: {len(sim_ticks)}  ({SIM_START} → {SIM_END})", flush=True)

    records = []
    skipped = 0

    for i, T in enumerate(sim_ticks):
        if i % 200 == 0:
            print(f"  [sim] {i}/{len(sim_ticks)} ticks …", flush=True)

        # Sigma_tau at T
        if T not in vol60_1m.index:
            skipped += 1
            continue
        sigma_tau = float(vol60_1m.loc[T])
        if not np.isfinite(sigma_tau) or sigma_tau <= 0:
            skipped += 1
            continue

        # Spot at T
        if T not in df1m.index:
            skipped += 1
            continue
        spot = float(df1m.loc[T, "close"])

        # Composite scores at T
        if T not in trend_s.index:
            skipped += 1
            continue
        trend = int(trend_s.loc[T])
        rev   = int(rev_s.loc[T])
        p_up  = lookup_p_up(trend, rev, asset="ETH")

        # Expiry price: first 1m bar at or after T+60min
        T_exp = T + pd.Timedelta(hours=1)
        future_bars = df1m.index[df1m.index >= T_exp]
        if len(future_bars) == 0:
            skipped += 1
            continue
        spot_exp = float(df1m.loc[future_bars[0], "close"])

        # Generate synthetic contracts
        for z_level in Z_LEVELS:
            strike = round(spot * math.exp(z_level * sigma_tau))
            if strike <= 0:
                continue

            z_strike = math.log(strike / spot) / sigma_tau if sigma_tau > 0 else z_level

            # Gate: z too close to ATM
            if abs(z_strike) < Z_MIN:
                continue

            # Gate: OTM NO (z_strike > 0 means strike above spot) too far out
            if z_strike > Z_MAX_MULT * VOL_FACTOR:
                continue

            # Synthetic market price (neutral BSM)
            pm_bsm  = _bsm_yes(spot, strike, sigma_tau)
            pm_ask  = min(0.97, pm_bsm + SPREAD)   # cost to buy YES
            pm_bid  = max(0.03, pm_bsm - SPREAD)   # cost to buy NO = 1 - pm_bid
            no_cost = 1.0 - pm_bid

            # Outcome
            no_wins = (spot_exp < strike)

            for K in K_VALUES:
                p_no = _p_no_logdrift(spot, strike, sigma_tau, p_up, K)
                net_edge = p_no - no_cost

                if net_edge < MIN_EDGE:
                    continue

                pnl = (pm_bsm if no_wins else -no_cost)

                records.append({
                    "T":          T,
                    "K":          K,
                    "spot":       round(spot, 2),
                    "strike":     strike,
                    "z_strike":   round(z_strike, 3),
                    "sigma_tau":  round(sigma_tau, 5),
                    "pm_bsm":     round(pm_bsm, 4),
                    "p_up":       round(p_up, 4),
                    "p_no":       round(p_no, 4),
                    "net_edge":   round(net_edge, 4),
                    "trend":      trend,
                    "rev":        rev,
                    "no_wins":    no_wins,
                    "pnl":        round(pnl, 4),
                })

    print(f"  [sim] Done. {len(records)} trade records, {skipped} ticks skipped.", flush=True)
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("ETH NO log-drift simulation — K sweep results")
    print(f"Period: {SIM_START} → {SIM_END}  |  flat $1 stake per contract")
    print("="*70)

    for K in K_VALUES:
        kdf = df[df["K"] == K]
        if kdf.empty:
            print(f"  K={K:.2f}: no trades")
            continue
        n      = len(kdf)
        wr     = kdf["no_wins"].mean()
        pnl    = kdf["pnl"].sum()
        ppt    = pnl / n
        be     = kdf["pm_bsm"].apply(lambda p: 1.0 - p).mean()  # avg breakeven WR
        print(f"  K={K:.2f}:  n={n:5d}  WR={wr*100:.1f}%  BE={be*100:.1f}%  "
              f"WR-BE={( wr-be)*100:+.1f}pp  PnL=${pnl:+.2f}  $/t=${ppt:+.4f}")

    print()
    print("=== Best K by pm band ===")
    for K in K_VALUES:
        kdf = df[df["K"] == K]
        print(f"\n  K={K:.2f} — pm breakdown:")
        for lo, hi in [(0,0.25),(0.25,0.40),(0.40,0.55),(0.55,1.0)]:
            seg = kdf[(kdf["pm_bsm"]>=lo)&(kdf["pm_bsm"]<hi)]
            if len(seg)<5: continue
            wr  = seg["no_wins"].mean()
            be  = seg["pm_bsm"].apply(lambda p: 1.0-p).mean()
            pnl = seg["pnl"].sum()
            print(f"    pm [{lo:.2f},{hi:.2f}): n={len(seg):4d}  WR={wr*100:.1f}%  "
                  f"BE={be*100:.1f}%  WR-BE={( wr-be)*100:+.1f}pp  PnL=${pnl:+.2f}")


if __name__ == "__main__":
    print("ETH NO K_drift simulation starting …")
    df = run_simulation()

    out_path = RESULTS_DIR / "eth_no_drift_sim.csv"
    df.to_csv(out_path, index=False)
    print(f"  [sim] Raw results saved → {out_path}")

    summarize(df)
