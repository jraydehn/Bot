"""
simulate_smc_gate_reform.py

Tests modifying smc_gate for the BTC 1h paper trader:
  Current:  block ALL BTC YES when bos_4h=bearish (or choch_4h + bos_1h=bearish)
  Proposed: only block YES when pm < 0.35 — allow near-ATM YES (pm >= 0.35) through

Data sources:
  - results/blocked_trades.csv  (gate audit log)
  - results/paper_trades.csv    (actual paper trades)

Simulation logic:
  1. Filter blocked_trades → smc_gate + BTC YES + resolved_yes not null + would_pnl not null
  2. Group by scan minute (floor logged_at to minute)
  3. For each scan minute, pick the best YES block (highest net_edge)
  4. OTM (pm < 0.35): keep blocking — no change
  5. Near-ATM (pm >= 0.35): allow through; competes vs actual paper_trade that minute
       - If no actual trade that minute → YES fires, P&L = would_pnl
       - If actual trade exists AND blocked YES net_edge > actual net_edge → YES replaces it
         (approx: treat as additive — actual still fires, blocked YES also fires since
          the system can bet on multiple contracts per scan; the block prevented the YES
          but didn't affect the other side trade)
       - Simplification: since smc_gate blocks YES on a different contract/side than
         what's actually traded (actual trades are mostly NO-side), both can co-exist.
         We add would_pnl for each near-ATM YES that would now be unblocked.

  Simple version: raw delta = sum(would_pnl) for near-ATM YES blocks that fire.
  We note that would_pnl=0 rows (Kelly assigned 0) are excluded from P&L but counted.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results")

# ── Load data ────────────────────────────────────────────────────────────────

print("Loading blocked_trades.csv ...")
bt = pd.read_csv(RESULTS_DIR / "blocked_trades.csv", low_memory=False)

print("Loading paper_trades.csv ...")
pt = pd.read_csv(RESULTS_DIR / "paper_trades.csv", low_memory=False)

# ── Step 1: Filter smc_gate BTC YES ─────────────────────────────────────────

smc_all = bt[
    (bt["gate_name"] == "smc_gate") &
    (bt["asset"] == "BTC") &
    (bt["side"] == "yes")
].copy()

print(f"\nTotal smc_gate BTC YES rows: {len(smc_all):,}")

# Rows with resolved_yes null (unresolved / still open)
null_resolved = smc_all["resolved_yes"].isna()
print(f"  Excluded (unresolved, would_pnl=null): {null_resolved.sum():,}")

smc = smc_all[~null_resolved & smc_all["would_pnl"].notna()].copy()
print(f"  After dropping unresolved: {len(smc):,}")

# Rows with would_pnl=0 (Kelly assigned 0 bet size — net_edge too low)
zero_pnl = (smc["would_pnl"] == 0)
print(f"  would_pnl=0 rows (Kelly size=0, net_edge below threshold): {zero_pnl.sum():,}")
print(f"    → Excluded from P&L calculations, noted in output")

smc_nonzero = smc[~zero_pnl].copy()
print(f"  Rows with nonzero would_pnl: {len(smc_nonzero):,}")

# ── Step 2: Compute scan minute ──────────────────────────────────────────────

smc_nonzero["logged_at_dt"] = pd.to_datetime(smc_nonzero["logged_at"])
smc_nonzero["scan_min"] = smc_nonzero["logged_at_dt"].dt.floor("min")
smc_nonzero["date"] = smc_nonzero["logged_at_dt"].dt.date

# ── Step 3: Split into OTM vs Near-ATM ──────────────────────────────────────

otm = smc_nonzero[smc_nonzero["pm"] < 0.35].copy()
near_atm = smc_nonzero[smc_nonzero["pm"] >= 0.35].copy()

print(f"\nSplit (nonzero would_pnl):")
print(f"  OTM  (pm < 0.35) — keep blocking:  {len(otm):,} rows")
print(f"  Near-ATM (pm >= 0.35) — allow through: {len(near_atm):,} rows")

# ── Step 4: Best YES per scan minute for near-ATM ───────────────────────────
# Per spec: pick the trade with highest net_edge per scan minute.
# This represents the one trade that would have been selected.

best_per_min = (
    near_atm
    .sort_values("net_edge", ascending=False)
    .drop_duplicates(subset="scan_min", keep="first")
    .copy()
)
print(f"\nUnique scan minutes (near-ATM best-per-minute): {len(best_per_min):,}")

# ── Step 5: Check competition vs paper_trades ────────────────────────────────
# Paper trades for BTC

pt["logged_at_dt"] = pd.to_datetime(pt["logged_at"])
pt["scan_min"] = pt["logged_at_dt"].dt.floor("min")
btc_pt = pt[pt["contract_ticker"].str.startswith("KXBTCD", na=False)].copy()
btc_actual_trades = btc_pt[btc_pt["decision"] == "trade"].copy()

# Index actual trades by scan_min → best (highest net_edge) actual trade per minute
best_actual_per_min = (
    btc_actual_trades
    .sort_values("net_edge", ascending=False)
    .drop_duplicates(subset="scan_min", keep="first")
    .set_index("scan_min")[["net_edge", "would_pnl", "side"]]
    .rename(columns={"net_edge": "actual_net_edge", "would_pnl": "actual_pnl", "side": "actual_side"})
)

# Join
best_per_min = best_per_min.join(
    best_actual_per_min, on="scan_min", how="left"
)

# Classification:
# - no_trade minute: no actual BTC trade at that scan minute → YES fires
# - actual trade exists: smc-blocked YES is a DIFFERENT contract/side (YES vs NO)
#   Both can fire simultaneously. We add would_pnl (YES fires in addition to existing NO trade).
#   Exception: if actual trade is also YES on same contract → the YES would replace it
#   (but looking at the data, actuals are almost all NO-side, so treat as additive).

has_actual = best_per_min["actual_net_edge"].notna()
n_no_trade_minutes = (~has_actual).sum()
n_has_actual_minutes = has_actual.sum()

print(f"\nOf {len(best_per_min):,} near-ATM unique scan minutes:")
print(f"  No actual BTC trade that minute: {n_no_trade_minutes:,}  → YES fires freely")
print(f"  Had actual BTC trade that minute: {n_has_actual_minutes:,} → YES fires additively (different side)")

# All near-ATM best-per-min would now fire under the proposed change
# would_pnl = flat dollar outcome already computed by Kelly at blocking time
firing_trades = best_per_min.copy()

# ── Step 6: P&L computation ──────────────────────────────────────────────────

delta_pnl = firing_trades["would_pnl"].sum()
n_wins = (firing_trades["resolved_yes"] == True).sum()
n_losses = (firing_trades["resolved_yes"] == False).sum()
n_total = len(firing_trades)

wr = n_wins / n_total if n_total > 0 else 0

# Breakeven win rate: for YES bets, at pm=p, you win (1-pm)/pm × bet on win,
# lose bet on loss. Breakeven WR = pm for YES contracts on Kalshi.
# But since would_pnl already encodes Kelly sizing, use dollar-based breakeven.
total_win_pnl = firing_trades.loc[firing_trades["resolved_yes"] == True, "would_pnl"].sum()
total_loss_pnl = firing_trades.loc[firing_trades["resolved_yes"] == False, "would_pnl"].sum()
# Breakeven WR: wins × avg_win_pnl_per_trade = losses × avg_loss_per_trade (at break-even)
# Simpler: breakeven WR = |avg_loss| / (avg_win + |avg_loss|)
avg_win = total_win_pnl / n_wins if n_wins > 0 else 0
avg_loss = abs(total_loss_pnl / n_losses) if n_losses > 0 else 0
breakeven_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0

# ── Bucket breakdown ─────────────────────────────────────────────────────────

BUCKETS = [
    (0.35, 0.50, "0.35–0.50"),
    (0.50, 0.65, "0.50–0.65"),
    (0.65, 0.80, "0.65–0.80"),
    (0.80, 1.01, "0.80+   "),
]

bucket_rows = []
for lo, hi, label in BUCKETS:
    b = firing_trades[(firing_trades["pm"] >= lo) & (firing_trades["pm"] < hi)]
    if len(b) == 0:
        bucket_rows.append({
            "bucket": label, "n": 0, "wins": 0, "losses": 0,
            "wr": float("nan"), "breakeven_wr": float("nan"), "pnl": 0.0
        })
        continue
    b_wins = (b["resolved_yes"] == True).sum()
    b_losses = (b["resolved_yes"] == False).sum()
    b_wr = b_wins / len(b)
    b_pnl = b["would_pnl"].sum()
    b_avg_win = b.loc[b["resolved_yes"] == True, "would_pnl"].mean() if b_wins > 0 else 0
    b_avg_loss = abs(b.loc[b["resolved_yes"] == False, "would_pnl"].mean()) if b_losses > 0 else 0
    b_bkeven = b_avg_loss / (b_avg_win + b_avg_loss) if (b_avg_win + b_avg_loss) > 0 else 0
    bucket_rows.append({
        "bucket": label, "n": len(b), "wins": b_wins, "losses": b_losses,
        "wr": b_wr, "breakeven_wr": b_bkeven, "pnl": b_pnl
    })

# Also: all near-ATM (including multi-per-minute, before best-per-min selection)
# for raw unfiltered bucket stats
raw_near_atm_buckets = []
for lo, hi, label in BUCKETS:
    b = near_atm[(near_atm["pm"] >= lo) & (near_atm["pm"] < hi)]
    raw_near_atm_buckets.append((label, len(b), b["would_pnl"].sum()))

# ── Daily P&L delta ──────────────────────────────────────────────────────────

daily = (
    firing_trades
    .groupby("date")["would_pnl"]
    .agg(["sum", "count", lambda x: (x > 0).sum()])
    .rename(columns={"sum": "delta_pnl", "count": "n_trades", "<lambda_0>": "wins"})
    .reset_index()
)
daily["wr"] = daily["wins"] / daily["n_trades"]

# ── Print results ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SMC GATE REFORM SIMULATION — BTC YES pm >= 0.35 unblocked")
print("=" * 70)

print(f"""
DATA COVERAGE
  Blocked trades date range: {smc_nonzero['logged_at_dt'].min().date()} → {smc_nonzero['logged_at_dt'].max().date()}
  All-time smc_gate BTC YES rows: {len(smc_all):,}
  Excluded (unresolved/null would_pnl): {null_resolved.sum():,}
  Excluded (would_pnl=0, Kelly size=0): {zero_pnl.sum():,}
  Eligible (nonzero would_pnl): {len(smc_nonzero):,}
    → OTM (pm < 0.35, keep blocking):     {len(otm):,}
    → Near-ATM (pm >= 0.35, unblock):     {len(near_atm):,}
  After best-per-scan-minute selection:  {len(best_per_min):,} unique trade events
""")

print("─" * 70)
print("OVERALL SUMMARY (best-per-minute near-ATM YES unblocked)")
print("─" * 70)
print(f"  Trades affected (scan minutes): {n_total:,}")
print(f"  Wins:  {n_wins:,}  |  Losses: {n_losses:,}")
print(f"  Win rate:        {wr*100:.1f}%")
print(f"  Breakeven WR:    {breakeven_wr*100:.1f}%")
print(f"  Total P&L delta: ${delta_pnl:+,.2f}")
print(f"  Avg win  / trade: ${avg_win:+.2f}")
print(f"  Avg loss / trade: ${-avg_loss:.2f}")
print()

print("─" * 70)
print("BUCKET BREAKDOWN (best-per-minute, proposed unblocking)")
print("─" * 70)
print(f"  {'Bucket':<12} {'N':>5} {'Wins':>5} {'Losses':>7} {'WR':>7} {'B/E WR':>8} {'P&L Delta':>12}")
print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*12}")
for r in bucket_rows:
    wr_str = f"{r['wr']*100:.1f}%" if not np.isnan(r['wr']) else "N/A"
    be_str = f"{r['breakeven_wr']*100:.1f}%" if not np.isnan(r['breakeven_wr']) else "N/A"
    print(f"  {r['bucket']:<12} {r['n']:>5} {r['wins']:>5} {r['losses']:>7} {wr_str:>7} {be_str:>8} ${r['pnl']:>+10,.2f}")
print()

print("─" * 70)
print("RAW (all near-ATM rows, before best-per-minute selection)")
print("─" * 70)
print(f"  {'Bucket':<12} {'N':>6} {'Raw P&L':>12}")
for label, n, pnl in raw_near_atm_buckets:
    print(f"  {label:<12} {n:>6} ${pnl:>+10,.2f}")
print()

print("─" * 70)
print("DAILY P&L DELTA (best-per-minute, near-ATM unblocked)")
print("─" * 70)
print(f"  {'Date':<12} {'N':>5} {'Wins':>5} {'WR':>7} {'Delta P&L':>12}")
print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*7} {'─'*12}")
running_total = 0
for _, row in daily.iterrows():
    running_total += row["delta_pnl"]
    print(f"  {str(row['date']):<12} {int(row['n_trades']):>5} {int(row['wins']):>5} {row['wr']*100:>6.1f}% ${row['delta_pnl']:>+10,.2f}   (cumul: ${running_total:+,.2f})")

print()
print("─" * 70)
print("INTERPRETATION")
print("─" * 70)
print(f"""
  Current behavior: smc_gate blocks ALL BTC YES → $0 from these trades.
  Proposed change:  allow YES through when pm >= 0.35.

  Best-per-minute analysis (one trade per scan cycle):
    → {n_total:,} new YES trades would fire
    → Net P&L delta: ${delta_pnl:+,.2f}
    → Win rate {wr*100:.1f}% vs breakeven {breakeven_wr*100:.1f}%
    → {'POSITIVE EDGE — unblocking increases P&L' if delta_pnl > 0 else 'NEGATIVE EDGE — unblocking reduces P&L'}

  Caveats:
    - would_pnl uses flat Kelly sizing from blocked_trades.csv (no recompute).
    - Competition check: {n_has_actual_minutes:,} scan minutes had a concurrent actual trade;
      since actual trades are predominantly NO-side, both fire (additive, not replacing).
    - {zero_pnl.sum():,} rows with would_pnl=0 excluded (Kelly below min threshold).
    - Data window: {smc_nonzero['logged_at_dt'].min().date()} → {smc_nonzero['logged_at_dt'].max().date()}
      ({(smc_nonzero['logged_at_dt'].max() - smc_nonzero['logged_at_dt'].min()).days + 1} days)
""")
