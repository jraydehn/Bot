"""
Run paper_trade_runner.py for BTC, ETH, and SOL simultaneously.

Each asset runs as a separate subprocess. Output is prefixed with the asset
name so all three streams are readable in one terminal. Crashed subprocesses
are automatically restarted after a 5-second delay.

Usage:
    # Paper only
    python3 run_all_assets.py

    # Live with per-asset bankrolls and loss limits
    python3 run_all_assets.py --live \
        --btc-bankroll 250 --btc-loss-limit 50 \
        --eth-bankroll 100 --eth-loss-limit 20 \
        --sol-bankroll 100 --sol-loss-limit 20
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ASSETS = ["BTC", "ETH", "SOL"]
RESTART_DELAY = 5  # seconds before restarting a crashed subprocess


def stream_output(proc, prefix: str) -> None:
    """Read lines from proc.stdout and print with asset prefix."""
    for line in proc.stdout:
        print(f"  [{prefix}] {line.rstrip()}", flush=True)


def run_asset(asset: str, extra_args: list) -> None:
    """Launch paper_trade_runner for one asset; restart on crash."""
    script = Path(__file__).parent / "paper_trade_runner.py"
    cmd = [sys.executable, "-u", str(script), "--asset", asset] + extra_args

    while True:
        print(f"  [{asset}] Starting...", flush=True)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stream_output(proc, asset)
            proc.wait()
            exit_code = proc.returncode
        except Exception as exc:
            print(f"  [{asset}] Launch error: {exc}", flush=True)
            exit_code = -1

        print(f"  [{asset}] Exited (code={exit_code}). Restarting in {RESTART_DELAY}s...", flush=True)
        time.sleep(RESTART_DELAY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all asset traders simultaneously")
    parser.add_argument("--live", action="store_true",
                        help="Place real orders on Kalshi (default: paper only)")
    parser.add_argument("--max-contracts", type=int, default=50)

    # Per-asset bankroll and loss limit
    parser.add_argument("--btc-bankroll",   type=float, default=250.0)
    parser.add_argument("--btc-loss-limit", type=float, default=50.0)
    parser.add_argument("--eth-bankroll",   type=float, default=100.0)
    parser.add_argument("--eth-loss-limit", type=float, default=20.0)
    parser.add_argument("--sol-bankroll",   type=float, default=100.0)
    parser.add_argument("--sol-loss-limit", type=float, default=20.0)
    args = parser.parse_args()

    asset_args = {
        "BTC": ["--bankroll", str(args.btc_bankroll), "--daily-loss-limit", str(args.btc_loss_limit)],
        "ETH": ["--bankroll", str(args.eth_bankroll), "--daily-loss-limit", str(args.eth_loss_limit)],
        "SOL": ["--bankroll", str(args.sol_bankroll), "--daily-loss-limit", str(args.sol_loss_limit)],
    }
    if args.live:
        for a in ASSETS:
            asset_args[a] += ["--live", "--max-contracts", str(args.max_contracts)]

    mode = "LIVE" if args.live else "PAPER"
    print(f"  Mode: {mode}")
    for a in ASSETS:
        print(f"  {a}: {' '.join(asset_args[a])}")
    print()

    threads = []
    for i, asset in enumerate(ASSETS):
        t = threading.Thread(target=run_asset, args=(asset, asset_args[asset]), daemon=True)
        t.start()
        threads.append(t)
        if i < len(ASSETS) - 1:
            time.sleep(20)  # stagger starts to avoid simultaneous API/data load contention

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n  Shutting down all asset runners.")
        sys.exit(0)


if __name__ == "__main__":
    main()
