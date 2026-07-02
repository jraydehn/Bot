#!/usr/bin/env python3
"""
run_all_15m.py

Launch paper_trade_runner_15m.py for BTC, ETH, and SOL simultaneously.
Each asset runs as a separate subprocess with prefixed output.

Usage:
    python3 run_all_15m.py
    python3 run_all_15m.py --btc-bankroll 1000 --eth-bankroll 500 --sol-bankroll 500
"""

import argparse
import fcntl as _fcntl
import subprocess
import sys
import threading
import time
from pathlib import Path

RESTART_DELAY = 5


def _is_locked(asset: str) -> bool:
    """Return True if another process holds the 15m flock for this asset."""
    lock_path = Path(__file__).parent / f".paper_trade_15m_{asset}.lock"
    try:
        with open(lock_path, "w") as fd:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        return False
    except (BlockingIOError, OSError):
        return True


def stream_output(proc, prefix: str) -> None:
    for line in proc.stdout:
        print(f"  [{prefix}] {line.rstrip()}", flush=True)


def run_asset(asset: str, bankroll: float, is_live: bool = False, loss_limit: float = 150.0) -> None:
    script = Path(__file__).parent / "paper_trade_runner_15m.py"
    cmd = [sys.executable, "-u", str(script), "--asset", asset,
           "--bankroll", str(bankroll), "--loop"]
    if is_live:
        cmd += ["--live", "--daily-loss-limit", str(loss_limit)]
    while True:
        print(f"  [{asset}15M] Starting...", flush=True)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stream_output(proc, f"{asset}15M")
            proc.wait()
            code = proc.returncode
        except Exception as exc:
            print(f"  [{asset}15M] Launch error: {exc}", flush=True)
            code = -1

        if code == 1 and _is_locked(asset):
            print(f"  [{asset}15M] Another process is running — watchdog standing by.", flush=True)
            while _is_locked(asset):
                time.sleep(30)
            print(f"  [{asset}15M] Lock released — restarting.", flush=True)
        else:
            print(f"  [{asset}15M] Exited (code={code}). Restarting in {RESTART_DELAY}s...", flush=True)
            time.sleep(RESTART_DELAY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 15m paper traders for BTC, ETH, and SOL.")
    parser.add_argument("--btc-bankroll", type=float, default=1000.0)
    parser.add_argument("--eth-bankroll", type=float, default=1000.0)
    parser.add_argument("--sol-bankroll", type=float, default=1000.0)
    parser.add_argument("--sol-live", action="store_true",
                        help="Run SOL 15m in live mode (place real orders)")
    parser.add_argument("--sol-loss-limit", type=float, default=150.0,
                        help="Daily loss limit for SOL live 15m runner")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated assets to NOT run here (e.g. BTC when a standalone "
                             "live 15m runner owns BTC — avoids two processes logging the same paper book)")
    args = parser.parse_args()

    _skip = {a.strip().upper() for a in args.skip.split(",") if a.strip()}
    assets = [
        (a, br, live, ll) for a, br, live, ll in [
            ("BTC", args.btc_bankroll, False, 150.0),
            ("ETH", args.eth_bankroll, False, 150.0),
            ("SOL", args.sol_bankroll, args.sol_live, args.sol_loss_limit),
        ] if a not in _skip
    ]

    print("=" * 60)
    mode_str = "LIVE" if args.sol_live else "PAPER"
    print(f"  15M TRADERS — BTC(paper) / ETH(paper) / SOL({mode_str})")
    print("=" * 60)
    for asset, br, live, ll in assets:
        live_tag = f"  LIVE loss-limit=${ll:.0f}" if live else ""
        print(f"  {asset}: bankroll=${br:,.0f}{live_tag}")
    print()

    threads = []
    for asset, bankroll, is_live, loss_limit in assets:
        t = threading.Thread(target=run_asset, args=(asset, bankroll, is_live, loss_limit), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")


if __name__ == "__main__":
    main()
