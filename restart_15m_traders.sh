#!/bin/bash
# restart_15m_traders.sh
# Restart ONLY the 15m watchdog and its children, leaving 1h traders running.
# Kills the 15m watchdog FIRST so it cannot respawn children mid-kill.

cd "$(dirname "$0")"

echo "=== Stopping 15m watchdog ==="
pkill -f "run_all_15m.py" 2>/dev/null && echo "  killed run_all_15m.py" || echo "  run_all_15m.py not running"

sleep 2

echo "=== Stopping 15m trader subprocesses ==="
pkill -f "paper_trade_runner_15m.py" 2>/dev/null && echo "  killed 15m traders" || echo "  no 15m traders running"

sleep 3

REMAINING=$(pgrep -f "paper_trade_runner_15m" 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING" -gt 0 ]; then
    echo "  WARNING: $REMAINING still alive — force killing..."
    pkill -9 -f "paper_trade_runner_15m.py" 2>/dev/null
    sleep 2
fi
echo "  All 15m traders stopped."

echo ""
echo "=== Starting 15m watchdog ==="
nohup python3 run_all_15m.py \
    --btc-bankroll 1000 \
    --eth-bankroll 1000 \
    --sol-bankroll 1000 \
    > logs/watchdog_15m.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Done. 15m traders start within ~10s."
echo "Verify with:  ps aux | grep paper_trade_runner_15m | grep -v grep"
