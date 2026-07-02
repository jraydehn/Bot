#!/usr/bin/env python3
"""
simulate_pmarket_base.py

Compare two 15m model variants on resolved paper trade data:
  CURRENT: z_base = log(spot/strike) / (vol * sqrt(tau))  — log-normal base
  PMARKET: z_base = norm.ppf(p_market)                    — Kalshi market base

Same drift signals (bp, body, stoch) and tau decay are applied to both.
Gates are NOT applied — this isolates the base model change only.

Run: python3 simulate_pmarket_base.py
"""

import math
import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")

EDGE_THRESHOLD = 0.04
KELLY_MULT     = 0.25
MAX_BET_FRAC   = 0.03
BANKROLL       = 1000.0
SEP = "=" * 72

CSVS = {
    "BTC": "results/paper_trades_btc15m.csv",
    "ETH": "results/paper_trades_eth15m.csv",
    "SOL": "results/paper_trades_sol15m.csv",
}


# ── model replication ──────────────────────────────────────────────────────────

def drift_btc(bp: float, body: float, dir15: int, tau: float) -> tuple[float, float]:
    """Returns (z_drift, delta) for BTC branch."""
    z_bp = math.tanh((bp - 0.5) * 3.0) * math.sqrt(tau / 5.0)
    delta = 0.0
    if body >= 0.50:
        delta += dir15 * 0.05
    elif body >= 0.30:
        delta += dir15 * 0.02
    return z_bp, delta


def drift_eth(bp: float, stoch: float, body: float, dir15: int, tau: float) -> tuple[float, float]:
    """Returns (z_drift, delta) for ETH branch."""
    stoch5 = stoch / 100.0
    p5_up  = (bp + stoch5) / 2.0
    z_bp   = (p5_up - 0.5) * 2.0 * math.sqrt(tau / 5.0)
    z_body = 0.0
    if body >= 0.7:
        z_body = -float(norm.ppf(max(0.05, min(0.95, body)))) * dir15 * math.sqrt(tau / 5.0)
    return z_bp + z_body, 0.0


def drift_sol(bp: float, body: float, dir15: int, tau: float,
              composite_p_up: float) -> tuple[float, float]:
    """Returns (z_drift, delta) for SOL branch (pre-reform: composite_p_up + step deltas)."""
    z_drift = 0.0
    if composite_p_up is not None and 0.05 <= composite_p_up <= 0.95:
        z_drift = float(norm.ppf(composite_p_up)) * math.sqrt(tau / 60.0)
    delta = 0.0
    if bp > 0.65:    delta += 0.07
    elif bp > 0.55:  delta += 0.03
    elif bp < 0.35:  delta -= 0.07
    elif bp < 0.45:  delta -= 0.03
    if body >= 0.50:   delta += dir15 * 0.05
    elif body >= 0.30: delta += dir15 * 0.02
    return z_drift, delta


def compute_pmodel(z_base: float, z_drift: float, delta: float) -> float:
    p_base = float(norm.cdf(z_base + z_drift))
    return max(0.05, min(0.96, p_base + delta))


def best_side_edge(p_model: float, p_market: float, tau: float):
    edge_yes = p_model - p_market
    edge_no  = p_market - p_model
    if edge_yes >= edge_no:
        side, edge = "yes", edge_yes
    else:
        side, edge = "no", edge_no
    # tau confidence decay below 5 min
    tau_conf = (tau / 5.0) ** 2 if tau < 5.0 else 1.0
    return side, edge, edge * tau_conf


# ── Kelly flat sizing (approximate) ───────────────────────────────────────────

def kelly_pnl(side: str, edge: float, p_market: float, bankroll: float,
              would_pnl: float) -> float:
    """Re-size would_pnl to flat-$1000 Kelly."""
    if edge <= 0:
        return 0.0
    # Kelly fraction based on edge and market price
    if side == "yes":
        kf = (edge / (1 - p_market)) * KELLY_MULT
    else:
        kf = (edge / p_market) * KELLY_MULT
    kf = min(kf, MAX_BET_FRAC)
    bet = bankroll * kf
    # Scale would_pnl proportionally to the new bet size
    # would_pnl was sized at original bet; we don't have original bet, so use as-is
    return would_pnl


# ── simulation ────────────────────────────────────────────────────────────────

def run_asset(asset: str, df: pd.DataFrame):
    results = []

    for _, row in df.iterrows():
        pm    = float(row["p_market"])
        tau   = float(row["tau_minutes"])
        bp    = float(row.get("bp_5m", 0.5) or 0.5)
        body  = float(row.get("body_15m", 0.0) or 0.0)
        dir15 = int(row.get("dir_15m", 0) or 0)
        stoch = float(row.get("stoch_k_5m", 50.0) or 50.0)
        p_up  = row.get("composite_p_up")
        p_up  = float(p_up) if pd.notna(p_up) else None

        side_actual  = str(row["side"]).lower()
        would_win    = bool(row["would_win"])
        would_pnl    = float(row["would_pnl"])
        p_model_live = float(row["p_model_15m"]) if pd.notna(row.get("p_model_15m")) else None

        if tau <= 0 or pm <= 0 or pm >= 1:
            continue

        # ── get drift for this asset ────────────────────────────────────────
        if asset == "BTC":
            z_drift, delta = drift_btc(bp, body, dir15, tau)
        elif asset == "ETH":
            z_drift, delta = drift_eth(bp, stoch, body, dir15, tau)
        else:
            z_drift, delta = drift_sol(bp, body, dir15, tau, p_up)

        # ── CURRENT model: log-normal z_base (use recorded p_model_live) ───
        if p_model_live is not None:
            pm_cur = p_model_live
        else:
            pm_cur = None  # can't reconstruct without spot/strike/vol

        # ── PMARKET model: norm.ppf(p_market) as z_base ────────────────────
        pm_clamped = max(0.03, min(0.97, pm))
        z_base_pm  = float(norm.ppf(pm_clamped))
        pm_new     = compute_pmodel(z_base_pm, z_drift, delta)

        # Determine which side/edge each model picks
        side_new, edge_new, adj_edge_new = best_side_edge(pm_new, pm, tau)

        results.append({
            "pm": pm, "tau": tau,
            "p_model_cur": pm_cur,
            "p_model_new": pm_new,
            "side_actual": side_actual,
            "would_win": would_win,
            "would_pnl": would_pnl,
            "side_new": side_new,
            "edge_new": edge_new,
            "adj_edge_new": adj_edge_new,
        })

    return pd.DataFrame(results)


# ── reporting ──────────────────────────────────────────────────────────────────

def report(asset: str, df: pd.DataFrame, res: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  {asset} 15m  —  {len(res)} resolved trades")
    print(SEP)

    # ── CURRENT model performance (from live CSV) ─────────────────────────
    traded = df[df["would_pnl"].notna()].copy()
    traded["won"] = traded["would_win"].astype(bool)
    n_cur  = len(traded)
    wr_cur = traded["won"].mean()
    pnl_cur = traded["would_pnl"].sum()
    print(f"\n  CURRENT model (live trades, all resolved):")
    print(f"    N={n_cur}  WR={wr_cur:.1%}  PnL=${pnl_cur:+,.0f}")

    # ── PMARKET model: trades that pass EDGE_THRESHOLD ─────────────────────
    qual = res[res["adj_edge_new"] >= EDGE_THRESHOLD].copy()
    # Among qualifying new-model trades, check if actual side matches new model's pick
    # and whether they would have won
    qual["won_new"] = (
        ((qual["side_new"] == "yes") & (qual["side_actual"] == "yes") & qual["would_win"]) |
        ((qual["side_new"] == "yes") & (qual["side_actual"] == "no")  & ~qual["would_win"]) |
        ((qual["side_new"] == "no")  & (qual["side_actual"] == "no")  & qual["would_win"]) |
        ((qual["side_new"] == "no")  & (qual["side_actual"] == "yes") & ~qual["would_win"])
    )
    # Use would_pnl only when model agrees with actual side taken
    same_side = qual["side_new"] == qual["side_actual"]
    qual_same = qual[same_side]
    qual_diff = qual[~same_side]

    n_new = len(qual)
    n_same = len(qual_same)
    n_diff = len(qual_diff)

    print(f"\n  PMARKET model (norm.ppf(pm) base, same drift + tau decay):")
    print(f"    Qualifying trades (edge>={EDGE_THRESHOLD}): {n_new}")
    print(f"      Same side as live trade:     {n_same}  "
          f"WR={qual_same['would_win'].mean():.1%}  PnL=${qual_same['would_pnl'].sum():+,.0f}")
    if n_diff > 0:
        print(f"      Different side (would flip): {n_diff}")

    # Edge distribution
    print(f"\n  Edge distribution (new model, qualifying only):")
    bins = [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 1.0]
    labels = ["0.04-0.06", "0.06-0.08", "0.08-0.10", "0.10-0.15", "0.15-0.20", ">0.20"]
    qual_same["edge_bucket"] = pd.cut(qual_same["adj_edge_new"], bins=bins, labels=labels)
    for lbl in labels:
        sub = qual_same[qual_same["edge_bucket"] == lbl]
        if len(sub) == 0:
            continue
        wr = sub["would_win"].mean()
        pnl = sub["would_pnl"].sum()
        be = sub["pm"].mean()
        print(f"    edge={lbl:<12}  N={len(sub):>3}  WR={wr:.1%}  BE={be:.1%}  PnL=${pnl:+,.0f}")

    # ── Compare: current model vs pmarket model trade overlap ───────────────
    # How many live trades would be DROPPED or ADDED by the new model?
    all_res = res.copy()
    all_res["same_side"] = all_res["side_new"] == all_res["side_actual"]
    all_res["qualifies_new"] = all_res["adj_edge_new"] >= EDGE_THRESHOLD

    # All live trades are "qualifies_current" = True (they were taken)
    would_drop = all_res[~all_res["qualifies_new"] & all_res["same_side"]]
    would_flip = all_res[all_res["qualifies_new"] & ~all_res["same_side"]]
    would_keep = all_res[all_res["qualifies_new"] & all_res["same_side"]]

    print(f"\n  Trade routing vs current live trades:")
    print(f"    Keep same side & still qualifies:  {len(would_keep):>4}  "
          f"WR={would_keep['would_win'].mean():.1%}  PnL=${would_keep['would_pnl'].sum():+,.0f}")
    print(f"    Drop (below threshold in new):     {len(would_drop):>4}  "
          f"WR={would_drop['would_win'].mean():.1%}  PnL=${would_drop['would_pnl'].sum():+,.0f}")
    if len(would_flip):
        print(f"    Flip side in new model:            {len(would_flip):>4}")

    # Simulated PnL if we'd only taken would_keep trades
    n_keep = len(would_keep)
    wr_keep = would_keep["would_win"].mean() if n_keep else float("nan")
    pnl_keep = would_keep["would_pnl"].sum()
    pnl_drop = would_drop["would_pnl"].sum()

    print(f"\n  Summary:")
    print(f"    Current model:  N={n_cur}  WR={wr_cur:.1%}  PnL=${pnl_cur:+,.0f}")
    print(f"    Pmarket filter: N={n_keep}  WR={wr_keep:.1%}  PnL=${pnl_keep:+,.0f}  "
          f"(dropped {len(would_drop)} trades worth ${pnl_drop:+,.0f})")
    print(f"    PnL delta: ${pnl_keep - pnl_cur:+,.0f}")

    # ── p_model calibration comparison ────────────────────────────────────
    has_cur = res["p_model_cur"].notna()
    if has_cur.sum() > 20:
        print(f"\n  p_model calibration (YES trades, same-side only):")
        yes_same = qual_same[qual_same["side_new"] == "yes"]
        if len(yes_same) > 5:
            err_new = (yes_same["p_model_new"] - yes_same["pm"]).abs().mean()
            print(f"    Mean |p_model_new - p_market| = {err_new:.3f}  (should be small = model close to market)")
            print(f"    Mean p_model_new = {yes_same['p_model_new'].mean():.3f}  "
                  f"vs mean p_market = {yes_same['pm'].mean():.3f}")
        no_same = qual_same[qual_same["side_new"] == "no"]
        if len(no_same) > 5:
            print(f"    NO bets: mean p_model_new = {no_same['p_model_new'].mean():.3f}  "
                  f"vs mean p_market = {no_same['pm'].mean():.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  p_market base model vs log-normal base — 15m models")
    print("  Same drift signals (bp, body, stoch) + tau decay on both")
    print(SEP)

    for asset, csv_path in CSVS.items():
        df = pd.read_csv(csv_path)
        resolved = df[df["would_pnl"].notna()].copy()
        for col in ["p_market", "bp_5m", "body_15m", "dir_15m", "stoch_k_5m",
                    "tau_minutes", "composite_p_up", "p_model_15m", "would_win", "would_pnl"]:
            if col in resolved.columns:
                resolved[col] = pd.to_numeric(resolved[col], errors="coerce")
        resolved = resolved.dropna(subset=["p_market", "tau_minutes", "would_pnl"])
        res = run_asset(asset, resolved)
        report(asset, resolved, res)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
