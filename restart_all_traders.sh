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
echo "=== Starting 1h watchdog (BTC dual / ETH paper / SOL dual) ==="
# Config as of 2026-07-02: bankrolls 2500; stop-loss BTC 350 / SOL 250 (shared
# SOL pool with 15m via live_trades_sol.csv) per stop-level sweep — stops <$300
# BTC / <$225 SOL were cutting mean-reverting days (-$955 / -$33 historical).
nohup python3 run_all_assets.py \
    --btc-dual --btc-bankroll 2500 --btc-loss-limit 350 \
    --eth-bankroll 2500 --eth-loss-limit 20 \
    --sol-dual --sol-bankroll 2500 --sol-loss-limit 250 \
    >> logs/run_all_assets.log 2>&1 &
echo "  PID=$!"

echo ""
echo "=== Starting 15m watchdog (BTC live / ETH paper / SOL live) ==="
# BTC 15m live since 2026-07-03 (user go-live decision): $2,500 bankroll, $250 stop.
# Stops are PER-RUNNER since 2026-07-03: each runner's daily limit counts only its
# own contract series (15M vs hourly tickers) even though both write live_trades.csv.
nohup python3 run_all_15m.py \
    --btc-bankroll 2500 \
    --btc-live --btc-loss-limit 250 \
    --eth-bankroll 2500 \
    --sol-bankroll 2500 \
    --sol-live --sol-loss-limit 250 \
    >> logs/run_all_15m.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Done. Traders start staggered (~60s for all to appear)."
echo "Verify with:  ps aux | grep -E '(run_all|paper_trade_runner)' | grep -v grep"
