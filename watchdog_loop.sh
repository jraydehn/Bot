#!/bin/bash
# [2026-08-15] Watchdog LOOP daemon. Why: macOS TCC blocks cron from
# ~/Documents ("Operation not permitted" in /var/mail — cron fired but was
# refused at the folder since 08-12 17:44, so every runner freeze went
# uncaught). This loop is spawned from an interactive session that HAS
# Documents access and inherits it. Survives until reboot; after a reboot,
# re-spawn it (or grant /usr/sbin/cron Full Disk Access in System Settings
# for a durable cron path). Local file ops only — no network, no hang risk.
cd /Users/justindehn/Documents/ClaudeCode/kalshi_btc || exit 1
while true; do
  /bin/bash runner_watchdog.sh
  sleep 600
done
