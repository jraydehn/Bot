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
echo "=== Starting 1h watchdog (BTC/ETH/SOL) ==="
# BTC/SOL run paper-only; ETH 15m live runner is started separately via restart_15m_traders.sh
nohup python3 run_all_assets.py \
    --btc-bankroll 2000 --btc-loss-limit 300 --max-contracts 1000 \
    --eth-bankroll 2000 --eth-loss-limit 240 \
    --sol-bankroll 2000 --sol-loss-limit 240 \
    > logs/watchdog_1h.log 2>&1 &
echo "  PID=$!"

echo ""
echo "=== Starting 15m watchdog (BTC/ETH/SOL) ==="
nohup python3 run_all_15m.py \
    --btc-bankroll 2000 \
    --eth-bankroll 2000 \
    --sol-bankroll 2000 \
    > logs/watchdog_15m.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Done. Traders start staggered (~60s for all to appear)."
echo "Verify with:  ps aux | grep -E '(run_all|paper_trade_runner)' | grep -v grep"
