#!/bin/bash
# runner_watchdog.sh — [2026-08-12] restart runners whose LOGS go stale.
#
# Why: the 08-08..08-11 fleet outage — processes alive but loops hung in
# unbounded DNS/network syscalls that requests' timeout does not cover
# (see feedback_runner_restart_isolation memory). pgrep cannot detect
# this; log mtime can. Installed proactively after that incident. Run from cron every 10 minutes:
#   */10 * * * * /Users/justindehn/Documents/ClaudeCode/kalshi_btc/runner_watchdog.sh
#
# Scope: 15m paper runners + hourly voltail challengers.
# [2026-08-12] bookdyn x2 + v7/v8 RETIRED (books -$6.1k/-$1.2k/-$5.4k/-$4.0k) — removed from watch. The hourly
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
# [2026-08-14] niche runners added — BTC v1 froze-alive ~30h and v2 ~14h
# unnoticed BECAUSE they were never in this watchdog (the exact failure
# mode it exists for). Hourly cadence: logs should tick every few min.
check "btc_hourly_niche_runner.py" logs/btc_hourly_niche.log \
      "python3 -u btc_hourly_niche_runner.py"
# [2026-08-18 READ] BTC niche v2 RETIRED (final +$178/136 vs v1 +$1,506/81;
# no witness/hybrid value). ETH v2 stays — its book feeds the CORE referee.
check "hourly_niche_runner_v2.py --asset ETH" logs/eth_hourly_niche_v2.log \
      "python3 -u hourly_niche_runner_v2.py --asset ETH"
# [2026-08-18] ETH hourly YES-favorite paper book (model-free bias,
# 3/3-window validated; tracker 08-18). Archive-tail pattern; [hb] lines.
check "eth_hourly_fav_runner.py" logs/eth_hourly_fav.log \
      "python3 -u eth_hourly_fav_runner.py"
# [2026-08-19] ETH hourly fav-RESCUES paper book (bands B/C outside the
# fav range, sweep-validated w/ Aug confirm + 1c fill stress; A/D
# rejected). First read ~09-01.
check "eth_hourly_fav_rescues_runner.py" logs/eth_hourly_fav_rescues.log \
      "python3 -u eth_hourly_fav_rescues_runner.py"
check "btc_hourly_voltail_runner.py" results/btc_hourly_voltail_runner.log \
      "python3 -u btc_hourly_voltail_runner.py"
check "eth_hourly_voltail_runner.py" results/eth_hourly_voltail_runner.log \
      "python3 -u eth_hourly_voltail_runner.py"
