"""
fix_archive_logged_at.py  — one-time repair of btc_scan_archive.csv

~95k rows (close_ts 2026-06-03..06-13) lost `logged_at` due to a since-fixed
bug in an old fill_scan_outcomes() rewrite. All other columns are intact.
Reconstruct logged_at = close_ts - tau_minutes (validated <2min vs actual on
good rows). Atomic write via temp + os.replace. Backs up first.

Run AFTER pausing the BTC hourly runner to avoid a lost-append race.
"""
import csv, os, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

P = Path(__file__).parent / "results" / "btc_scan_archive.csv"
BAK = P.with_name("btc_scan_archive_pre_logged_at_fix.csv")

def parse_ts(s):
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def main():
    with open(P, newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)
    print(f"loaded {len(rows)} rows, {len(cols)} cols")

    fixed = skipped = 0
    for row in rows:
        if (row.get("logged_at") or "").strip():
            continue
        cts = parse_ts(row.get("close_ts"))
        try:
            tau = float(row.get("tau_minutes") or "nan")
        except ValueError:
            tau = float("nan")
        if cts is None or tau != tau:  # NaN check
            skipped += 1
            continue
        la = cts - timedelta(minutes=tau)
        row["logged_at"] = la.strftime("%Y-%m-%d %H:%M:%S")
        fixed += 1

    print(f"would fill {fixed} blank rows; {skipped} unrecoverable (no close_ts/tau)")
    blanks_remaining = sum(1 for r in rows if not (r.get("logged_at") or "").strip())
    print(f"blank logged_at remaining after fill: {blanks_remaining}")

    import sys
    if "--apply" not in sys.argv:
        print("DRY RUN — pass --apply to write (will back up to "
              f"{BAK.name} first)")
        return

    shutil.copy2(P, BAK)
    print(f"backed up -> {BAK.name}")
    tmp = P.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, P)
    print(f"wrote {len(rows)} rows -> {P.name}")

if __name__ == "__main__":
    main()
