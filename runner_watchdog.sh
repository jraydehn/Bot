#!/bin/bash
# runner_watchdog.sh — [2026-08-12] restart runners whose LOGS go stale.
#
# Why: the 08-08..08-11 fleet outage — processes alive but loops hung in
# unbounded DNS/network syscalls that requests' timeout does not cover
# (see feedback_runner_restart_isolation memory). pgrep cannot detect
# this; log mtime can. Installed proactively after that incident. Run from cron every 10 minutes:
#   */10 * * * * /Users/justindehn/Documents/ClaudeCode/kalshi_btc/runner_watchdog.sh
#
# Scope: 15m paper runners + hourly challenger runners. The hourly
# PRODUCTION runners live under run_all_assets.py's own watchdog and are
# left alone. To disable: remove the crontab line.
cd /Users/justindehn/Documents/ClaudeCode/kalshi_btc || exit 1
STALE_MIN=30
WLOG=logs/runner_watchdog.log

check() {  # $1 pattern  $2 logfile  $3 restart-cmd
    local pat="$1" log="$2" cmd="$3"
    if [ ! -f "$log" ]; then return; fi
    if [ -z "$(find "$log" -mmin -$STALE_MIN 2>/dev/null)" ]; then
        echo "$(date -u '+%F %T') STALE ($pat): log >${STALE_MIN}min old — restarting" >> "$WLOG"
        pkill -f "$pat" 2>/dev/null
        sleep 3
        eval "nohup $cmd >> $log 2>&1 &"
        disown 2>/dev/null
    fi
}

check "paper_trade_runner_15m.py --asset BTC" logs/btc15m_paper_twin.log \
      "python3 -u paper_trade_runner_15m.py --asset BTC --bankroll 2500.0 --loop"
check "paper_trade_runner_15m.py --asset ETH" logs/eth15m_paper_twin.log \
      "python3 -u paper_trade_runner_15m.py --asset ETH --bankroll 2500.0 --loop"
check "paper_trade_runner_15m.py --asset SOL" logs/paper_sol.log \
      "python3 -u paper_trade_runner_15m.py --asset SOL --bankroll 2500.0 --loop"
check "btc_hourly_voltail_runner.py" results/btc_hourly_voltail_runner.log \
      "python3 -u btc_hourly_voltail_runner.py"
check "eth_hourly_voltail_runner.py" results/eth_hourly_voltail_runner.log \
      "python3 -u eth_hourly_voltail_runner.py"
check "sol_hourly_v7_runner.py" results/sol_hourly_v7_runner.log \
      "python3 -u sol_hourly_v7_runner.py"
check "sol_hourly_v8_runner.py" results/sol_hourly_v8_runner.log \
      "python3 -u sol_hourly_v8_runner.py"
check "btc_hourly_bookdyn_runner.py" results/btc_hourly_bookdyn_runner.log \
      "python3 -u btc_hourly_bookdyn_runner.py"
check "eth_hourly_bookdyn_runner.py" results/eth_hourly_bookdyn_runner.log \
      "python3 -u eth_hourly_bookdyn_runner.py"
