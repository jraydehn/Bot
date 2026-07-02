"""
test_third_no_trend_regime.py
Test whether 3rd NO per expiry performs better in bearish vs neutral/bullish regimes.

Approach:
  - 2yr Binance 1h BTC data (2024-06-01 to present)
  - Simplified composite_trend proxy: EMA20/50/200 alignment + stoch_k_1h + stoch_k_4h
  - Simulate 3 NO tiers per 1h expiry window at fixed strike offsets (+0.3%, +0.5%, +0.8%)
  - Tier 1 = deepest OTM (model picks first, best edge)
  - Tier 3 = least deep OTM (3rd pick, currently blocked by cap=2)
  - Outcome: did BTC close below strike at expiry (1h later)?
  - Split by trend regime → WR + PnL + MCPT
"""

import math, warnings
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIERS = {
    "T1_deep":  0.008,   # +0.8% offset → deepest OTM, model picks 1st
    "T2_mid":   0.005,   # +0.5% offset → 2nd pick
    "T3_near":  0.003,   # +0.3% offset → 3rd pick (blocked by current cap)
}
REGIME_THRESHOLDS = {"strong_bear": -3, "strong_bull": 3}  # trend_score cutoffs
N_PERM = 10_000

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_binance(interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://api.binance.us/api/v3/klines"
    rows, cur = [], start_ms
    while cur < end_ms:
        k = requests.get(url, params={
            "symbol": "BTCUSDT", "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": 1000
        }, timeout=20).json()
        if not k or not isinstance(k, list):
            break
        rows += k
        cur = k[-1][0] + 1
        if len(k) < 1000:
            break
    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ["o","h","l","c","v"]:
        df[col] = df[col].astype(float)
    return df.set_index("ts").sort_index()

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    return (close - lo) / (hi - lo + 1e-9) * 100

def compute_trend_score(h1: pd.DataFrame, h4: pd.DataFrame) -> pd.Series:
    """
    Simplified composite_trend proxy (-5 to +5):
      EMA alignment: close vs EMA20/50/200 → -3 to +3
      Stoch_k_1h:   <40 → -1, >60 → +1
      Stoch_k_4h:   <40 → -1, >60 → +1  (reindexed to 1h)
    """
    c = h1["c"]

    # EMA alignment
    ema20  = c.ewm(span=20,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    ema_score = (
        np.sign(c - ema20).astype(int) +
        np.sign(c - ema50).astype(int) +
        np.sign(c - ema200).astype(int)
    )

    # Stoch 1h
    sk1h = stoch_k(h1["h"], h1["l"], h1["c"], 14)
    stoch_1h = pd.Series(0, index=h1.index)
    stoch_1h[sk1h > 60] =  1
    stoch_1h[sk1h < 40] = -1

    # Stoch 4h — resample 1h → 4h, then forward-fill back to 1h index
    sk4h_raw = stoch_k(h4["h"], h4["l"], h4["c"], 14)
    sk4h = sk4h_raw.reindex(h1.index, method="ffill")
    stoch_4h = pd.Series(0, index=h1.index)
    stoch_4h[sk4h > 60] =  1
    stoch_4h[sk4h < 40] = -1

    score = ema_score + stoch_1h + stoch_4h
    return score.clip(-5, 5)

def regime_label(score: float) -> str:
    if score <= REGIME_THRESHOLDS["strong_bear"]:
        return "strong_bear"
    if score >= REGIME_THRESHOLDS["strong_bull"]:
        return "strong_bull"
    return "neutral"

# ---------------------------------------------------------------------------
# Lognormal pm helper (market-implied YES probability)
# ---------------------------------------------------------------------------
def lognorm_yes_prob(spot: float, strike: float, rv_1h: float, tau_h: float = 1.0) -> float:
    """P(BTC_close > strike) under lognormal with drift=0, vol=rv_1h per bar."""
    if rv_1h <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    sigma = rv_1h * math.sqrt(tau_h)
    d2 = (math.log(spot / strike)) / sigma
    return float(norm.cdf(d2))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Fetching 2yr Binance 1h data...")
    start = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    end   = int(pd.Timestamp("2026-06-22", tz="UTC").timestamp() * 1000)

    h1 = fetch_binance("1h", start, end)
    h4 = fetch_binance("4h", start, end)
    print(f"  1h bars: {len(h1)}  |  4h bars: {len(h4)}")

    # Rolling realized vol (24h window, annualized → per-bar fraction)
    lr = np.log(h1["c"] / h1["c"].shift(1))
    rv1h = lr.rolling(24).std()   # per-1h-bar vol

    # Trend score
    trend = compute_trend_score(h1, h4)
    trend = trend.reindex(h1.index).fillna(0)

    # Need at least 200 bars of warmup for EMA200
    valid_start = h1.index[200]
    h1 = h1[h1.index >= valid_start]
    trend = trend[trend.index >= valid_start]
    rv1h  = rv1h[rv1h.index >= valid_start]

    print(f"  Analysis window: {h1.index[0].date()} → {h1.index[-1].date()}  ({len(h1)} bars)")
    print(f"  Trend score distribution:")
    tc = pd.cut(trend, bins=[-6,-3,-1,1,3,6], labels=["≤-3","-2to-1","neutral","+1to+2","≥+3"])
    print(f"  {tc.value_counts().sort_index().to_dict()}")

    # Build simulation rows
    records = []
    for i in range(len(h1) - 1):
        bar     = h1.iloc[i]
        nxt     = h1.iloc[i + 1]
        ts      = h1.index[i]
        spot    = bar["c"]
        rv      = rv1h.iloc[i]
        t_score = float(trend.iloc[i])
        regime  = regime_label(t_score)

        if math.isnan(rv) or rv <= 0:
            continue

        next_close = nxt["c"]

        for tier, offset in TIERS.items():
            strike = spot * (1 + offset)
            pm_yes = lognorm_yes_prob(spot, strike, rv, tau_h=1.0)
            pm_no  = 1 - pm_yes
            no_wins = int(next_close < strike)   # NO wins if close < strike

            # PnL: bet $1 on NO, collect pm_yes/(1-pm_yes) if win, lose $1 if lose
            # Using lognormal pm as the "market price" (fair value = no edge in expectation)
            # Edge comes from whether realized WR beats breakeven (1 - pm_yes)
            pnl = (pm_yes / pm_no) if no_wins else -1.0

            records.append({
                "ts":      ts,
                "tier":    tier,
                "offset":  offset,
                "strike":  round(strike, 2),
                "pm_yes":  round(pm_yes, 4),
                "rv":      round(rv, 6),
                "trend":   t_score,
                "regime":  regime,
                "no_wins": no_wins,
                "pnl":     round(pnl, 4),
            })

    df = pd.DataFrame(records)
    print(f"\nTotal simulation rows: {len(df)}")

    # ---------------------------------------------------------------------------
    # Results by tier × regime
    # ---------------------------------------------------------------------------
    print("\n" + "="*70)
    print("WIN RATE + PnL by TIER × REGIME")
    print("="*70)
    regime_order = ["strong_bear", "neutral", "strong_bull"]

    summary = {}
    for tier in TIERS:
        sub = df[df["tier"] == tier]
        be_wr = 1 - sub["pm_yes"].mean()   # breakeven WR = 1 - avg_pm_yes
        print(f"\n  {tier}  (offset={TIERS[tier]*100:.1f}%)  breakeven WR≈{be_wr:.1%}  n={len(sub)}")
        for reg in regime_order:
            r = sub[sub["regime"] == reg]
            if len(r) == 0:
                continue
            wr = r["no_wins"].mean()
            pnl_sum = r["pnl"].sum()
            be = 1 - r["pm_yes"].mean()
            print(f"    {reg:15s}: n={len(r):5d}  WR={wr:.1%}  BE={be:.1%}  "
                  f"edge={wr-be:+.1%}  PnL(sim)={pnl_sum:+.1f}")
        summary[tier] = sub

    # ---------------------------------------------------------------------------
    # Key question: Does 3rd NO (T3_near) perform better in strong_bear?
    # ---------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FOCUS: T3_near (3rd NO pick) — is strong_bear significantly better?")
    print("="*70)
    t3 = df[df["tier"] == "T3_near"].copy()
    t3_bear    = t3[t3["regime"] == "strong_bear"]
    t3_neutral = t3[t3["regime"] == "neutral"]
    t3_bull    = t3[t3["regime"] == "strong_bull"]
    t3_nonbear = t3[t3["regime"] != "strong_bear"]

    for grp, lbl in [(t3_bear,"strong_bear"), (t3_neutral,"neutral"),
                     (t3_bull,"strong_bull"), (t3_nonbear,"non-bear")]:
        if len(grp) == 0:
            continue
        wr = grp["no_wins"].mean()
        be = 1 - grp["pm_yes"].mean()
        print(f"  {lbl:15s}: n={len(grp):5d}  WR={wr:.1%}  BE={be:.1%}  edge={wr-be:+.1%}")

    # Year-stability: T3 bear vs non-bear by year
    print("\n  Year-stability (T3_near edge = WR - BE):")
    t3["year"] = t3["ts"].dt.year
    for yr, grp in t3.groupby("year"):
        for reg in ["strong_bear", "neutral", "strong_bull"]:
            r = grp[grp["regime"] == reg]
            if len(r) < 10:
                continue
            wr = r["no_wins"].mean()
            be = 1 - r["pm_yes"].mean()
            print(f"    {yr}  {reg:15s}: n={len(r):4d}  WR={wr:.1%}  BE={be:.1%}  edge={wr-be:+.2%}")

    # MCPT: is T3_bear WR unusually high vs random sample of same size from T3?
    print("\n  MCPT: T3 strong_bear vs null (random T3 sample of same size)")
    observed_wr = t3_bear["no_wins"].mean()
    n_bear      = len(t3_bear)
    rng = np.random.default_rng(42)
    null_wrs = np.array([
        t3["no_wins"].sample(n_bear, random_state=int(rng.integers(1e9))).mean()
        for _ in range(N_PERM)
    ])
    p_val = (null_wrs >= observed_wr).mean()
    z     = (observed_wr - null_wrs.mean()) / null_wrs.std()
    print(f"    Observed WR={observed_wr:.3f}  z={z:+.2f}  p={p_val:.4f}")
    print(f"    {'SIGNIFICANT — bearish regime improves T3 NO' if p_val < 0.05 else 'NOT significant'}")

    # Same for strong_bull (does bullish regime allow 3rd YES — inverse test)
    print("\n  MCPT: T3 WR in strong_bull (low WR = bullish = bad for NO)")
    observed_wr_bull = t3_bull["no_wins"].mean()
    n_bull = len(t3_bull)
    if n_bull > 0:
        null_wrs_b = np.array([
            t3["no_wins"].sample(n_bull, random_state=int(rng.integers(1e9))).mean()
            for _ in range(N_PERM)
        ])
        p_bull = (null_wrs_b <= observed_wr_bull).mean()
        z_bull = (observed_wr_bull - null_wrs_b.mean()) / null_wrs_b.std()
        print(f"    Observed WR={observed_wr_bull:.3f}  z={z_bull:+.2f}  p_low={p_bull:.4f}")
        print(f"    {'SIGNIFICANT — bullish regime hurts T3 NO (block it)' if p_bull < 0.05 else 'NOT significant'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
