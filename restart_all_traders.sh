#!/bin/bash
# restart_all_traders.sh
# Clean restart of all watchdogs and traders. Kills watchdogs FIRST so they
# cannot respawn children while we're killing them. Then restarts watchdogs,
# which in turn spawn exactly one child per asset.

set -e
cd "$(dirname "$0")"

export COINGLASS_API_KEY="8f0a30c29a5e424ba2641f649051786b"

echo "=== Stopping watchdogs ==="
pkill -f "run_all_assets.py" 2>/dev/null && echo "  killed run_all_assets.py" || echo "  run_all_assets.py not running"
pkill -f "run_all_15m.py"   2>/dev/null && echo "  killed run_all_15m.py"   || echo "  run_all_15m.py not running"

# Brief wait for watchdog threads to acknowledge termination
sleep 2

echo "=== Stopping all trader subprocesses ==="
pkill -f "paper_trade_runner_15m.py" 2>/dev/null && echo "  killed 15m traders" || echo "  no 15m traders running"
pkill -f "paper_trade_runner.py"     2>/dev/null && echo "  killed 1h traders"  || echo "  no 1h traders running"

sleep 3

echo "=== Verifying clean stop ==="
REMAINING=$(pgrep -f "paper_trade_runner" 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING" -gt 0 ]; then
    echo "  WARNING: $REMAINING process(es) still alive — force killing..."
    pkill -9 -f "paper_trade_runner_15m.py" 2>/dev/null
    pkill -9 -f "paper_trade_runner.py"     2>/dev/null
    pkill -9 -f "run_all_assets.py"         2>/dev/null
    pkill -9 -f "run_all_15m.py"            2>/dev/null
    sleep 2
fi
echo "  All clear."

echo ""
echo "=== Starting 1h watchdog (BTC PAPER / ETH paper / SOL paper) ==="
# 2026-07-05: SOL live/dual STOPPED (user decision — degrading performance).
# SOL continues in PAPER mode for data collection per feedback_paper_always_on.
# 2026-07-10: BTC hourly converted to PAPER-ONLY (user decision): slow grind-down
# in live PnL, user wants to move toward a high-conviction/lower-volume gating
# approach before risking live capital again. Live money paused; paper collection
# continues per feedback_paper_always_on. See project_btc_hourly_paper_20260710.md.
nohup python3 run_all_assets.py \
    --btc-bankroll 2500 --btc-loss-limit 350 \
    --eth-bankroll 2500 --eth-loss-limit 20 \
    --sol-bankroll 2500 \
    >> logs/run_all_assets.log 2>&1 &
echo "  PID=$!"

echo ""
echo "=== Starting 15m watchdog (BTC PAPER / ETH paper / SOL live) ==="
# BTC 15m converted to PAPER-ONLY 2026-07-10 (user decision): the YES-bias
# investigation found the live decision path has been running on a chronically
# bullish-saturated fallback since ~06-26 (see project_pup15m_20260710.md).
# Live money paused pending a full model revamp; paper collection continues
# per feedback_paper_always_on, now also shadow-logging the corrected p_up_v2
# -> K_YES/K_NO model (p_model_yes_v2/no_v2/best_side_v2/best_edge_v2 in
# btc_scan_archive_15m.csv) for comparison against the existing model.
# SOL 15m live restarted 2026-07-06 (user decision): $2,500 bankroll, $150 stop
# (was stopped 2026-07-05 for degrading performance — no signal-side changes
# since then, see project_sol_live_stop_20260705.md).
nohup python3 run_all_15m.py \
    --btc-bankroll 2500 \
    --eth-bankroll 2500 \
    --sol-bankroll 2500 \
    --sol-live --sol-loss-limit 150 \
    >> logs/run_all_15m.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Done. Traders start staggered (~60s for all to appear)."
echo "Verify with:  ps aux | grep -E '(run_all|paper_trade_runner)' | grep -v grep"
