#!/usr/bin/env bash
# Hourly paper trading runner.
# Refreshes OHLCV data then logs a signal + checks outcomes.
#
# Run manually:
#   bash run_paper_trade.sh
#
# Schedule with cron (top of each hour):
#   crontab -e
#   0 * * * * /path/to/kalshi_btc/run_paper_trade.sh >> /path/to/kalshi_btc/results/cron.log 2>&1
#
# Required env vars (set in your shell profile or crontab):
#   export KALSHI_KEY_ID=your-key-id
#   export KALSHI_KEY_PATH=~/kalshi_key_fixed.pem

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/results/cron.log"

echo ""
echo "========================================"
echo "  PAPER TRADE RUN  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# 1. Refresh OHLCV data
echo ""
echo "-- fetch_data.py --"
python3 "$DIR/fetch_data.py"

# 2. Log a new signal
echo ""
echo "-- paper_trade_runner.py --"
python3 "$DIR/paper_trade_runner.py"

# 3. Check outcomes for expired contracts
echo ""
echo "-- outcome_checker.py --"
python3 "$DIR/outcome_checker.py"

echo ""
echo "Done."
