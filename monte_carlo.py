"""
Monte Carlo backtest: 20 seeds, dynamic p_market, full dataset.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_point import load_data
from backtest import run_backtest

N_SEEDS    = 20
OFFSET     = 0.001
BANKROLL   = 10_000

print("Loading data...", flush=True)
df_1m, df_1h, df_4h = load_data()
print(f"  1m : {len(df_1m):,} rows", flush=True)
print(f"  1h : {len(df_1h):,} rows", flush=True)
print(f"  4h : {len(df_4h):,} rows", flush=True)
print(flush=True)

print(f"  Decision points total (1h rows): ~{len(df_1h):,}", flush=True)
print(f"  Strike offset   : {OFFSET:.3%} above spot", flush=True)
print(f"  p_market        : dynamic (simulated per step)", flush=True)
print(f"  Starting bankroll: ${BANKROLL:,.2f}", flush=True)
print(flush=True)

finals = []
drawdowns = []
trades_list = []
wins_list = []

for seed in range(N_SEEDS):
    random.seed(seed)
    df = run_backtest(df_1m, df_1h, df_4h,
                      strike_offset=OFFSET,
                      p_market=None,
                      bankroll=BANKROLL)

    if df.empty:
        print(f"  seed={seed:<2} NO RESULTS", flush=True)
        continue

    # Build equity curve
    bankroll = BANKROLL
    peak = bankroll
    max_dd = 0.0
    trade_count = 0
    win_count = 0

    for _, row in df.iterrows():
        if row["decision"] == "trade":
            bankroll = row["bankroll_after"]
            trade_count += 1
            # win = YES trade resolved yes, or NO trade resolved no
            side = row.get("side", "yes")
            resolved = row.get("resolved_yes", False)
            if (side == "yes" and resolved) or (side == "no" and not resolved):
                win_count += 1
            if bankroll > peak:
                peak = bankroll
            dd = (bankroll - peak) / peak
            if dd < max_dd:
                max_dd = dd

    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    finals.append(bankroll)
    drawdowns.append(max_dd)
    trades_list.append(trade_count)
    wins_list.append(win_rate)

    if (seed + 1) % 5 == 0 or seed == N_SEEDS - 1:
        print(f"  [ {seed+1:>2}/{N_SEEDS}]  "
              f"seed={seed:<2}  final=${bankroll:,.0f}  "
              f"max_dd={max_dd:.1%}  trades={trade_count}  win={win_rate:.1%}",
              flush=True)

# Summary
finals_sorted = sorted(finals)
median_final = finals_sorted[len(finals_sorted) // 2]
mean_final   = sum(finals) / len(finals)
drawdowns_sorted = sorted(drawdowns)
median_dd  = drawdowns_sorted[len(drawdowns_sorted) // 2]
worst_dd   = min(drawdowns)
profitable = sum(1 for f in finals if f > BANKROLL)

print(flush=True)
print("=" * 55, flush=True)
print(f"  MONTE CARLO  ({N_SEEDS} seeds, offset={OFFSET:.1%}, ${BANKROLL/1000:.0f}k start)", flush=True)
print("=" * 55, flush=True)
print(f"  Median final bankroll : ${median_final:,.0f}  ({100*(median_final/BANKROLL-1):+.1f}%)", flush=True)
print(f"  Mean   final bankroll : ${mean_final:,.0f}  ({100*(mean_final/BANKROLL-1):+.1f}%)", flush=True)
print(f"  Min    final bankroll : ${min(finals):,.0f}", flush=True)
print(f"  Max    final bankroll : ${max(finals):,.0f}", flush=True)
print(f"  Profitable runs       : {profitable}/{N_SEEDS}", flush=True)
print(f"  Median max drawdown   : {median_dd:.1%}", flush=True)
print(f"  Worst  max drawdown   : {worst_dd:.1%}", flush=True)
print(flush=True)

# Per-seed table
print(f"  {'seed':<5} {'final':>10} {'return':>8} {'max_dd':>8} {'trades':>7} {'win%':>7}", flush=True)
print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7}", flush=True)
for i, (f, dd, t, w) in enumerate(zip(finals, drawdowns, trades_list, wins_list)):
    ret = 100 * (f / BANKROLL - 1)
    print(f"  {i:<5} ${f:>9,.0f} {ret:>+7.1f}% {dd:>7.1%} {t:>7} {w:>6.1%}", flush=True)
