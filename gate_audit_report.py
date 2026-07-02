"""
gate_audit_report.py — Per-gate accuracy report from blocked_trades.csv.

Run:
    python gate_audit_report.py              # all gates, all assets
    python gate_audit_report.py --asset BTC  # BTC only
    python gate_audit_report.py --gate btc_exhaustion_gate

For each gate prints:
  n_blocked | n_resolved | WR_if_taken | $ blocked right | $ blocked wrong | net_impact
  correct block = gate prevented a loss (gate was right)
  false positive = gate prevented a win (gate was wrong, we missed money)
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

BLOCKED_CSV = Path(__file__).parent / "results" / "blocked_trades.csv"


def _pct(n, d):
    return f"{100*n/d:.1f}%" if d else "—"


def _fmt_pnl(v):
    if v >= 0:
        return f"+${v:.2f}"
    return f"-${abs(v):.2f}"


def run_report(asset_filter=None, gate_filter=None):
    if not BLOCKED_CSV.exists():
        print("blocked_trades.csv not found — no blocks logged yet.")
        return

    with open(BLOCKED_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if asset_filter:
        rows = [r for r in rows if r.get("asset", "").upper() == asset_filter.upper()]
    if gate_filter:
        rows = [r for r in rows if r.get("gate_name", "") == gate_filter]

    if not rows:
        print("No rows match filter.")
        return

    # Bucket by gate
    by_gate = defaultdict(list)
    for r in rows:
        by_gate[r.get("gate_name", "unknown")].append(r)

    # Summary header
    total_blocked  = len(rows)
    total_resolved = sum(1 for r in rows if (r.get("resolved_yes") or "").strip())
    print(f"\n{'='*72}")
    print(f"  Gate Audit Report — {BLOCKED_CSV.name}")
    if asset_filter: print(f"  Asset filter: {asset_filter}")
    print(f"  Total blocked: {total_blocked}  |  Resolved: {total_resolved}  |  Pending: {total_blocked-total_resolved}")
    print(f"{'='*72}\n")

    gate_summaries = []

    for gate_name in sorted(by_gate.keys()):
        gate_rows = by_gate[gate_name]
        resolved  = [r for r in gate_rows if (r.get("resolved_yes") or "").strip()]
        n_total   = len(gate_rows)
        n_res     = len(resolved)

        if n_res == 0:
            gate_summaries.append({
                "name": gate_name, "n": n_total, "n_res": 0,
                "wr_if_taken": None, "net": None,
                "correct": 0, "fp": 0,
                "pnl_correct": 0.0, "pnl_fp": 0.0,
            })
            continue

        # Parse would_pnl and resolved_yes
        wins_if_taken = 0
        pnl_correct   = 0.0   # money saved (gate blocked a loss)
        pnl_fp        = 0.0   # money missed (gate blocked a win)

        for r in resolved:
            try:
                res_yes = r["resolved_yes"].strip().lower() in ("true", "1", "yes")
                side    = r.get("side", "yes").strip().lower()
                wpnl    = float(r["would_pnl"]) if r.get("would_pnl", "").strip() else None

                # Did the blocked bet win?
                would_have_won = (side == "yes" and res_yes) or (side == "no" and not res_yes)
                if would_have_won:
                    wins_if_taken += 1

                if wpnl is not None:
                    if would_have_won:
                        pnl_fp += wpnl        # false positive: we missed this gain
                    else:
                        pnl_correct += abs(wpnl)  # correct block: we saved this loss
            except Exception:
                continue

        wr_if_taken = wins_if_taken / n_res
        net_impact  = pnl_correct - pnl_fp   # positive = gate saved money net

        gate_summaries.append({
            "name": gate_name, "n": n_total, "n_res": n_res,
            "wr_if_taken": wr_if_taken,
            "net": net_impact,
            "correct": n_res - wins_if_taken,
            "fp": wins_if_taken,
            "pnl_correct": pnl_correct,
            "pnl_fp": pnl_fp,
        })

    # Print per-gate table
    print(f"  {'Gate':<30} {'n':>5} {'res':>5} {'WR-if-taken':>12} {'correct':>8} {'FP':>5} {'$saved':>8} {'$missed':>8} {'net':>9}")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*12} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*9}")

    total_net = 0.0
    for g in gate_summaries:
        if g["n_res"] == 0:
            print(f"  {g['name']:<30} {g['n']:>5} {'0':>5} {'(pending)':>12}")
            continue
        be = _breakeven_for_gate(by_gate[g["name"]])
        wr_str = f"{g['wr_if_taken']*100:.1f}% (BE:{be*100:.0f}%)"
        net_str = _fmt_pnl(g["net"])
        print(
            f"  {g['name']:<30} {g['n']:>5} {g['n_res']:>5} "
            f"{wr_str:>12} {g['correct']:>8} {g['fp']:>5} "
            f"{_fmt_pnl(g['pnl_correct']):>8} {_fmt_pnl(g['pnl_fp']):>8} {net_str:>9}"
        )
        total_net += g["net"] if g["net"] is not None else 0

    print(f"\n  {'TOTAL NET IMPACT':>59} {_fmt_pnl(total_net):>9}")
    print()

    # Detailed breakdown per gate (if resolved trades exist)
    for g in gate_summaries:
        if g["n_res"] < 3:
            continue
        gate_rows_res = [r for r in by_gate[g["name"]] if (r.get("resolved_yes") or "").strip()]
        print(f"  [{g['name']}]  n_blocked={g['n']}  n_resolved={g['n_res']}")
        print(f"    WR if taken: {g['wr_if_taken']*100:.1f}%  |  "
              f"Correct blocks: {g['correct']}  |  False positives: {g['fp']}")
        print(f"    $ saved from correct blocks: {_fmt_pnl(g['pnl_correct'])}")
        print(f"    $ missed from false positives: {_fmt_pnl(g['pnl_fp'])}")
        print(f"    Net gate impact: {_fmt_pnl(g['net'])}")

        # Side breakdown
        sides = set(r.get("side","") for r in gate_rows_res)
        for side in sorted(sides):
            s_rows = [r for r in gate_rows_res if r.get("side","") == side]
            s_wins = sum(1 for r in s_rows if _would_have_won(r))
            print(f"    {side.upper()}: n={len(s_rows)} WR={_pct(s_wins, len(s_rows))}")
        print()


def _breakeven_for_gate(gate_rows) -> float:
    """Rough average breakeven WR across blocked trades in this gate."""
    bes = []
    for r in gate_rows:
        try:
            pm   = float(r.get("pm") or 0)
            side = r.get("side", "yes").strip().lower()
            if pm <= 0 or pm >= 1:
                continue
            be = (1 - pm) if side == "yes" else pm
            bes.append(be)
        except Exception:
            continue
    return sum(bes) / len(bes) if bes else 0.5


def _would_have_won(r) -> bool:
    try:
        res_yes = r["resolved_yes"].strip().lower() in ("true", "1", "yes")
        side    = r.get("side", "yes").strip().lower()
        return (side == "yes" and res_yes) or (side == "no" and not res_yes)
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate audit report")
    parser.add_argument("--asset", help="Filter to one asset (BTC/ETH/SOL)")
    parser.add_argument("--gate",  help="Filter to one gate name")
    args = parser.parse_args()
    run_report(asset_filter=args.asset, gate_filter=args.gate)
