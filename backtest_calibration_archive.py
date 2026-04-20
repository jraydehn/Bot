#!/usr/bin/env python3
"""
backtest_calibration_archive.py — P&L-based calibration sweep on REAL Kalshi data.

Replays paper_trade_archive_*.csv scans (real Kalshi p_market, real outcomes)
under candidate (k_drift, no_mult, yes_mult) parameters. Applies the full
decision.py gate stack, sizes via half-Kelly with 5% cap, tracks PnL.

Reports per-asset:
  - $PnL, WR, breakeven WR, max consecutive losses, max drawdown
  - per strike-distance bucket: n, WR, $PnL
  - n_yes / n_no split
"""

import math, sys, glob, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import lookup_p_up
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

RESULTS_DIR = Path(__file__).parent / "results"
BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD

# Per-asset gate parameters (mirrors decision.py)
ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max_otm_no": 0.40, "gate3": 0.01,
            "strike_step": 100.0},
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.005,
            "strike_step": 10.0},
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.01,
            "strike_step": 1.0},
}

GATE_CS_MIN_YES_OTM = 0.55
GATE_CI_MIN_BEARISH = 0.45
RR_MAX_NO = 4.0
RR_MIN_NO = 0.33
RR_MAX_YES = 3.0
RR_EDGE_EXC = 0.08


def load_archive(asset):
    """Load all archive + current CSV for an asset, dedupe by (decision_time, contract_ticker)."""
    if asset == "BTC":
        patterns = ["paper_trades_archive_2026*.csv", "paper_trades_archive_pre_*.csv", "paper_trades.csv"]
    elif asset == "ETH":
        patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    elif asset == "SOL":
        patterns = ["paper_trades_sol_archive_*.csv", "paper_trades_sol.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))
    # Filter out non-asset files (BTC pattern catches eth/sol archives)
    if asset == "BTC":
        files = [f for f in files if "_eth" not in f.name and "_sol" not in f.name]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["_src"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    needed = ["decision_time", "contract_ticker", "spot", "strike", "p_market",
              "vol_eff", "tau_minutes", "composite_trend", "composite_rev",
              "composite_p_up", "resolved_yes"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        print(f"  [{asset}] missing cols: {missing}")
        return pd.DataFrame()
    # Filter rows that have everything we need to evaluate + an outcome
    raw = raw.dropna(subset=needed)
    # Numeric coercions
    for c in ["spot", "strike", "p_market", "vol_eff", "tau_minutes",
              "composite_trend", "composite_rev", "composite_p_up", "resolved_yes"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=needed)
    # Dedup by (decision_time, contract_ticker) — keep last (most recent archive wins)
    raw = raw.drop_duplicates(subset=["decision_time", "contract_ticker"], keep="last")
    raw = raw.sort_values("decision_time").reset_index(drop=True)
    return raw


def compute_pmodel(spot, strike, vol_eff, tau, p_up, k_drift):
    """Recompute p_model with override drift multiplier."""
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k_drift
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def evaluate_row(row, k_drift, no_mult, yes_mult, params):
    """Apply gate stack to one scan row; return dict of best (yes/no) trade or None."""
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    vol_eff = row["vol_eff"]; tau = row["tau_minutes"]
    p_up = row["composite_p_up"]
    if vol_eff <= 0 or tau <= 0 or pm <= 0 or pm >= 1:
        return None
    offset = (strike - spot) / spot if spot > 0 else 0
    pm_raw = compute_pmodel(spot, strike, vol_eff, tau, p_up, k_drift)
    candidates = []
    for side in ("yes", "no"):
        if side == "yes":
            pm_use = float(np.clip(pm_raw * yes_mult, 0.01, 0.99))
        else:
            pm_use = float(np.clip(pm_raw * no_mult, 0.01, 0.99))
        # Gate 0 saturation
        if not (params["pm_min"] <= pm_use <= params["pm_max"]): continue
        if not (0.04 <= pm <= 0.96): continue
        # Gate CS / CI / NS (composite path)
        if side == "yes":
            if offset > 0 and p_up < GATE_CS_MIN_YES_OTM: continue
            if offset <= 0 and p_up < GATE_CI_MIN_BEARISH: continue
        if side == "no":
            if offset < 0 and p_up > params["ns_max_otm_no"]: continue
        # Edge
        fee = kalshi_fee(pm)
        if side == "yes":
            raw = pm_use - pm
            net = raw - fee - SLIPPAGE - SPREAD
            rr = pm / (1 - pm) if pm < 1 else 999
            if rr > RR_MAX_YES: continue
            # Gate OTM tier
            if pm < 0.15: tier_min = 0.04
            elif pm < 0.25: tier_min = 0.03
            elif pm < 0.35: tier_min = 0.02
            else: tier_min = 0.0
        else:
            raw = pm - pm_use
            net = raw - fee - SLIPPAGE - SPREAD
            rr = (1 - pm) / pm if pm > 0 else 999
            if (rr < RR_MIN_NO or rr > RR_MAX_NO) and net < RR_EDGE_EXC: continue
            tier_min = 0.0
        if net < max(params["gate3"], tier_min): continue
        candidates.append({"side": side, "pm_use": pm_use, "pm": pm, "net": net,
                           "offset": offset, "strike": strike})
    if not candidates: return None
    return max(candidates, key=lambda x: x["net"])


def kelly_bet(pm_use, pm, side, bankroll):
    if side == "yes":
        b = (1 - pm) / pm if pm > 0 else 0
        p, q = pm_use, 1 - pm_use
    else:
        b = pm / (1 - pm) if pm < 1 else 0
        p_no = 1 - pm_use
        p, q = p_no, 1 - p_no
    if b <= 0: return 0
    kf = max(0.0, (b*p - q)/b)
    bf = min(kf * KELLY_MULT, KELLY_CAP)
    return round(bankroll * bf, 2)


def trade_pnl(bet, side, pm, won):
    fee_rate = kalshi_fee(pm)
    if bet <= 0: return 0
    if side == "yes":
        if won:
            n_ct = bet / pm if pm > 0 else 0
            return bet * (1 - pm) / pm - fee_rate * n_ct
        return -bet
    else:
        if won:
            n_ct = bet / (1 - pm) if pm < 1 else 0
            return bet * pm / (1 - pm) - fee_rate * n_ct
        return -bet


def run_backtest(asset, scans_df, k_drift, no_mult, yes_mult):
    """Group by decision_time, pick best across scanned contracts, simulate."""
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []; offsets_strike = []; bankrolls = [BANKROLL_0]
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None
        best_row = None
        for _, row in group.iterrows():
            cand = evaluate_row(row, k_drift, no_mult, yes_mult, params)
            if cand is None: continue
            if best is None or cand["net"] > best["net"]:
                best = cand; best_row = row
        if best is None: continue
        bet = kelly_bet(best["pm_use"], best["pm"], best["side"], bankroll)
        if bet <= 0: continue
        actual_yes = int(best_row["resolved_yes"])
        won = (actual_yes == 1 and best["side"] == "yes") or (actual_yes == 0 and best["side"] == "no")
        pnl = trade_pnl(bet, best["side"], best["pm"], won)
        bankroll = max(1.0, bankroll + pnl)
        pnls.append(pnl); wins.append(won); sides.append(best["side"])
        # strike distance in increments
        sd = round(best["offset"] * best_row["spot"] / params["strike_step"])
        offsets_strike.append(sd)
        bankrolls.append(bankroll)
    if not pnls:
        return dict(n=0, wr=0, pnl=0, final=BANKROLL_0, max_loss_streak=0,
                    max_dd=0, n_yes=0, n_no=0, breakeven_wr=0, sd_buckets={})
    n = len(pnls); wr = sum(wins)/n
    streak = 0; max_streak = 0
    for w in wins:
        if not w:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0
    peak = BANKROLL_0; max_dd = 0
    for b in bankrolls:
        peak = max(peak, b); max_dd = max(max_dd, (peak - b) / peak)
    avg_win = sum(p for p, w in zip(pnls, wins) if w) / max(1, sum(wins))
    avg_loss = -sum(p for p, w in zip(pnls, wins) if not w) / max(1, n - sum(wins))
    breakeven_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0
    # Per strike-distance bucket
    sd_buckets = {}
    for sd, p, w in zip(offsets_strike, pnls, wins):
        if sd not in sd_buckets: sd_buckets[sd] = {"n": 0, "wins": 0, "pnl": 0.0}
        sd_buckets[sd]["n"] += 1
        sd_buckets[sd]["wins"] += int(w)
        sd_buckets[sd]["pnl"] += p
    return dict(n=n, wr=wr, pnl=sum(pnls), final=bankrolls[-1],
                max_loss_streak=max_streak, max_dd=max_dd,
                n_yes=sum(1 for s in sides if s == "yes"),
                n_no=sum(1 for s in sides if s == "no"),
                breakeven_wr=breakeven_wr,
                sd_buckets=sd_buckets)


def main():
    print(f"\n{'='*78}\n  CALIBRATION SWEEP — real Kalshi archive data, full gate stack\n{'='*78}\n", flush=True)

    sweeps = {
        "BTC": [(k, nm, ym) for k in [0.7, 1.0, 1.2, 1.4, 1.7]
                            for nm in [0.65, 0.80, 1.0]
                            for ym in [0.80, 0.90, 1.0]],
        "ETH": [(k, 1.0, 1.0) for k in [0.6, 0.8, 1.0, 1.2]],
        "SOL": [(k, 1.0, 1.0) for k in [0.1, 0.2, 0.5, 1.0]],
    }

    for asset in ("BTC", "ETH", "SOL"):
        t0 = time.time()
        print(f"[{asset}] loading archives...", flush=True)
        df = load_archive(asset)
        if df.empty:
            print(f"  [{asset}] NO DATA"); continue
        n_scans = len(df); n_hours = df["decision_time"].nunique()
        print(f"[{asset}] loaded {n_scans:,} scans across {n_hours:,} unique hours ({df['decision_time'].min()} → {df['decision_time'].max()})", flush=True)
        print(f"[{asset}] running {len(sweeps[asset])} param combos...", flush=True)
        results = []
        for k, nm, ym in sweeps[asset]:
            r = run_backtest(asset, df, k, nm, ym)
            r["k"], r["no_m"], r["yes_m"] = k, nm, ym
            results.append(r)
            print(f"  {asset} k={k:.2f} no_m={nm:.2f} yes_m={ym:.2f}: "
                  f"n={r['n']:4d} WR={r['wr']:.1%} (be={r['breakeven_wr']:.1%}) "
                  f"pnl=${r['pnl']:+8.2f} streak={r['max_loss_streak']:2d} maxDD={r['max_dd']:.1%} "
                  f"({r['n_yes']}y/{r['n_no']}n)", flush=True)
        print(f"[{asset}] done in {time.time()-t0:.1f}s", flush=True)
        # Top 5 by PnL
        results.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"\n[{asset}] TOP 5 by P&L:", flush=True)
        for r in results[:5]:
            print(f"  k={r['k']:.2f} no_m={r['no_m']:.2f} yes_m={r['yes_m']:.2f}  "
                  f"n={r['n']:4d} WR={r['wr']:.1%} (be={r['breakeven_wr']:.1%}) "
                  f"pnl=${r['pnl']:+8.2f} streak={r['max_loss_streak']:2d} maxDD={r['max_dd']:.1%}", flush=True)
        # Strike-distance bucket profile of best
        best = results[0]
        print(f"\n[{asset}] BEST combo strike-distance buckets:", flush=True)
        step = ASSET_PARAMS[asset]["strike_step"]
        for sd in sorted(best["sd_buckets"].keys()):
            b = best["sd_buckets"][sd]
            wr = b["wins"] / b["n"]
            print(f"  sd={sd:+3d} (${sd*step:+.0f})  n={b['n']:4d}  WR={wr:.1%}  PnL=${b['pnl']:+7.2f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
