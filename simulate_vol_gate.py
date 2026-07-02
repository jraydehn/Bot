#!/usr/bin/env python3
"""
simulate_vol_gate.py

Simulate a volatility-based gate on existing 15m paper trade data.

Gate logic:
  p_market gate: block YES when p_market < thresh; block NO when p_market > (1-thresh)
  sigma gate:    block when |offset_pct| > N × sigma_15m_pct

Both gates target deep-OTM contracts where empirical move probability
is near zero and model edge is illusory.

Columns used:
  p_market, side, offset_pct, realized_vol_annual,
  would_win, would_pnl, bet_amount, asset

Run: python3 simulate_vol_gate.py
"""

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CSVS = {
    "BTC": "results/paper_trades_btc15m.csv",
    "ETH": "results/paper_trades_eth15m.csv",
    "SOL": "results/paper_trades_sol15m.csv",
}
MINS_PER_YEAR = 525_600.0
SEP  = "=" * 78
SEP2 = "-" * 78

P_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
SIGMA_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def load_resolved() -> pd.DataFrame:
    dfs = []
    for asset, path in CSVS.items():
        d = pd.read_csv(path, low_memory=False)
        d["asset"] = asset
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    trades   = df[df["decision"] == "trade"].copy()
    resolved = trades[trades["would_pnl"].notna()].copy()

    # sigma_15m_pct from realized vol
    resolved["sigma_15m_pct"] = (
        resolved["realized_vol_annual"].astype(float)
        * math.sqrt(15.0 / MINS_PER_YEAR)
        * 100.0
    )
    resolved["n_sigma"] = (
        resolved["offset_pct"].abs() / resolved["sigma_15m_pct"].replace(0, np.nan)
    )
    resolved["p_market"]  = resolved["p_market"].astype(float)
    resolved["would_win"] = resolved["would_win"].astype(float)
    resolved["would_pnl"] = resolved["would_pnl"].astype(float)
    resolved["bet_amount"] = resolved["bet_amount"].astype(float)

    # YES side: p_market is the bet side price  (we want YES, market says p_market)
    # NO side:  price of NO = 1 - p_market      (we want NO, market says 1-p_market)
    # For the gate: "deep OTM YES" = p_market << 0.50; "deep OTM NO" = p_market >> 0.50
    return resolved


def gate_mask_p(df: pd.DataFrame, thresh: float) -> pd.Series:
    """Returns True where the p_market gate would BLOCK the trade."""
    block_yes = (df["side"] == "yes") & (df["p_market"] < thresh)
    block_no  = (df["side"] == "no")  & (df["p_market"] > 1.0 - thresh)
    return block_yes | block_no


def gate_mask_sigma(df: pd.DataFrame, n: float) -> pd.Series:
    """Returns True where the sigma gate would BLOCK the trade (|offset| > N×sigma)."""
    return df["n_sigma"] > n


def report_gate(df: pd.DataFrame, blocked: pd.Series, label: str):
    kept    = df[~blocked]
    blocked_ = df[blocked]

    n_total  = len(df)
    n_blocked = blocked.sum()
    n_kept    = n_total - n_blocked

    wr_total  = df["would_win"].mean()
    wr_blocked = blocked_["would_win"].mean() if n_blocked else float("nan")
    wr_kept    = kept["would_win"].mean() if n_kept else float("nan")

    pnl_total  = df["would_pnl"].sum()
    pnl_blocked = blocked_["would_pnl"].sum() if n_blocked else 0.0
    pnl_kept   = kept["would_pnl"].sum()
    pnl_delta  = pnl_kept - pnl_total   # positive = saved money by blocking

    wins_blocked  = int(blocked_["would_win"].sum()) if n_blocked else 0
    losses_blocked = n_blocked - wins_blocked

    print(f"  {label:<22}  "
          f"blocked={n_blocked:>3}  "
          f"(W{wins_blocked}/L{losses_blocked})  "
          f"WR_blocked={wr_blocked:>5.1%}  "
          f"WR_kept={wr_kept:>5.1%}  "
          f"PnL_kept=${pnl_kept:>+7.2f}  "
          f"delta=${pnl_delta:>+7.2f}")


def section_p_gate(df: pd.DataFrame, title: str):
    print(f"\n  {title}")
    print(f"  {'Gate':<22}  {'blocked':>8}  {'(W/L)':>7}  {'WR_blk':>9}  "
          f"{'WR_kept':>8}  {'PnL_kept':>10}  {'delta':>8}")
    print(f"  {SEP2}")
    print(f"  {'BASELINE (no gate)':<22}  "
          f"blocked={0:>3}  "
          f"( n/a  )  "
          f"WR_blocked=  n/a  "
          f"WR_kept={df['would_win'].mean():>5.1%}  "
          f"PnL_kept=${df['would_pnl'].sum():>+7.2f}  "
          f"delta=${0:>+7.2f}")

    for thresh in P_THRESHOLDS:
        blocked = gate_mask_p(df, thresh)
        label = f"p_mkt < {thresh:.2f} / > {1-thresh:.2f}"
        report_gate(df, blocked, label)


def section_sigma_gate(df: pd.DataFrame, title: str):
    print(f"\n  {title}")
    print(f"  {'Gate':<22}  {'blocked':>8}  {'(W/L)':>7}  {'WR_blk':>9}  "
          f"{'WR_kept':>8}  {'PnL_kept':>10}  {'delta':>8}")
    print(f"  {SEP2}")
    print(f"  {'BASELINE (no gate)':<22}  "
          f"blocked={0:>3}  "
          f"( n/a  )  "
          f"WR_blocked=  n/a  "
          f"WR_kept={df['would_win'].mean():>5.1%}  "
          f"PnL_kept=${df['would_pnl'].sum():>+7.2f}  "
          f"delta=${0:>+7.2f}")

    for n in SIGMA_THRESHOLDS:
        blocked = gate_mask_sigma(df, n)
        label = f"|offset| > {n:.1f}×sigma"
        report_gate(df, blocked, label)


def side_breakdown(df: pd.DataFrame, blocked: pd.Series, thresh: float):
    """Print YES vs NO breakdown for a given threshold."""
    for side_val, side_label in [("yes", "YES"), ("no", "NO")]:
        sub     = df[df["side"] == side_val]
        blk_sub = blocked[df["side"] == side_val]
        if len(sub) == 0:
            continue
        kept = sub[~blk_sub]
        blk_ = sub[blk_sub]
        n_blk = blk_sub.sum()
        wr_k = kept["would_win"].mean() if len(kept) else float("nan")
        pnl_k = kept["would_pnl"].sum() if len(kept) else 0.0
        wr_b = blk_["would_win"].mean() if n_blk else float("nan")
        w_b = int(blk_["would_win"].sum()) if n_blk else 0
        l_b = n_blk - w_b
        print(f"      {side_label}: blocked={n_blk}(W{w_b}/L{l_b}) "
              f"WR_blk={wr_b:.1%}  WR_kept={wr_k:.1%}  PnL_kept=${pnl_k:+.2f}")


def section_detail_by_asset(df: pd.DataFrame):
    print(f"\n  DETAIL: recommended gate (p_market < 0.10 / > 0.90) by asset + side")
    print(f"  {SEP2}")
    thresh = 0.10
    for asset in ["BTC", "ETH", "SOL"]:
        sub = df[df["asset"] == asset].copy()
        if sub.empty:
            continue
        blocked = gate_mask_p(sub.reset_index(drop=True), thresh)
        blocked.index = sub.index
        kept = sub[~blocked]
        blk_ = sub[blocked]
        n_blk = blocked.sum()
        pnl_base = sub["would_pnl"].sum()
        pnl_kept = kept["would_pnl"].sum()
        delta = pnl_kept - pnl_base
        wr_base = sub["would_win"].mean()
        wr_kept = kept["would_win"].mean() if len(kept) else float("nan")
        wr_blk  = blk_["would_win"].mean() if n_blk else float("nan")
        print(f"\n  {asset}  (N={len(sub)}, baseline WR={wr_base:.1%}, PnL=${pnl_base:+.2f})")
        print(f"    blocked={n_blk}  WR_blocked={wr_blk:.1%}  "
              f"WR_kept={wr_kept:.1%}  PnL_kept=${pnl_kept:+.2f}  delta=${delta:+.2f}")
        side_breakdown(sub, blocked.reindex(sub.index), thresh)


def section_blocked_trades(df: pd.DataFrame):
    """Show every trade that would be blocked at 0.10 threshold."""
    thresh = 0.10
    blocked = gate_mask_p(df, thresh)
    blk = df[blocked][["asset","side","p_market","offset_pct","sigma_15m_pct",
                        "n_sigma","would_win","would_pnl","bet_amount"]].copy()
    if blk.empty:
        return
    blk = blk.sort_values(["asset","p_market"])
    print(f"\n  ALL BLOCKED TRADES at p_market < 0.10 / > 0.90 ({len(blk)} trades)")
    print(f"  {SEP2}")
    print(f"  {'asset':<5} {'side':<4} {'p_mkt':>6} {'offset':>8} {'σ_15m':>7} "
          f"{'n_σ':>5} {'win':>4} {'pnl':>8} {'bet':>7}")
    print(f"  {'-'*5} {'-'*4} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*4} {'-'*8} {'-'*7}")
    for _, r in blk.iterrows():
        win_str = "WIN" if r["would_win"] == 1 else "LOSS"
        print(f"  {r['asset']:<5} {r['side']:<4} {r['p_market']:>6.3f} "
              f"{r['offset_pct']:>+7.3f}% {r['sigma_15m_pct']:>6.3f}% "
              f"{r['n_sigma']:>5.1f}  {win_str:>4} ${r['would_pnl']:>+7.2f} "
              f"${r['bet_amount']:>6.2f}")
    wins  = (blk["would_win"] == 1).sum()
    losses = len(blk) - wins
    print(f"\n  Summary: {wins}W / {losses}L  WR={wins/len(blk):.1%}  "
          f"PnL=${blk['would_pnl'].sum():+.2f}")


def main():
    print(SEP)
    print("  15m vol gate simulation — p_market gate + sigma gate")
    print("  Source: paper_trades_btc15m/eth15m/sol15m.csv (resolved trades only)")
    print(SEP)

    df = load_resolved()
    print(f"\n  Total resolved trades: {len(df)}")
    print(f"  BTC: {(df['asset']=='BTC').sum()}  "
          f"ETH: {(df['asset']=='ETH').sum()}  "
          f"SOL: {(df['asset']=='SOL').sum()}")
    print(f"  Baseline WR: {df['would_win'].mean():.1%}  "
          f"PnL: ${df['would_pnl'].sum():+.2f}")
    print(f"  sigma_15m_pct: median={df['sigma_15m_pct'].median():.3f}%  "
          f"mean={df['sigma_15m_pct'].mean():.3f}%")
    print(f"  |offset| in sigma: median={df['n_sigma'].median():.2f}σ  "
          f"mean={df['n_sigma'].mean():.2f}σ  max={df['n_sigma'].max():.2f}σ")

    # --- p_market gate: all assets combined ---
    section_p_gate(df, "P_MARKET GATE (all assets combined)")

    # --- sigma-based gate: all assets combined ---
    section_sigma_gate(df, "SIGMA GATE |offset| > N×sigma_15m (all assets combined)")

    # --- detail: recommended threshold by asset + side ---
    section_detail_by_asset(df)

    # --- show every blocked trade at the recommended level ---
    section_blocked_trades(df)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
