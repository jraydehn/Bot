"""
Run paper_trade_runner.py for BTC, ETH, and SOL simultaneously.

Each asset runs as a separate subprocess. Output is prefixed with the asset
name so all three streams are readable in one terminal. Crashed subprocesses
are automatically restarted after a 5-second delay.

Usage:
    python3 run_all_assets.py
    python3 run_all_assets.py --bankroll 1000
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
    parser = argparse.ArgumentParser(description="Run all asset paper traders simultaneously")
    parser.add_argument("--bankroll", type=float, default=None,
                        help="Bankroll to pass to each runner (default: runner default)")
    args = parser.parse_args()

    extra_args = []
    if args.bankroll is not None:
        extra_args += ["--bankroll", str(args.bankroll)]

    print(f"  Starting paper traders for: {', '.join(ASSETS)}")
    print(f"  Extra args: {extra_args or 'none'}\n")

    threads = []
    for i, asset in enumerate(ASSETS):
        t = threading.Thread(target=run_asset, args=(asset, extra_args), daemon=True)
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
