"""
sweep_k_drift.py — Sweep K_DRIFT values to find optimal P&L.

Loads signals and 1m data ONCE (including GARCH ~2 min), then runs the
simulation inner loop for each k without repeating the expensive setup.

Usage:
    python3 sweep_k_drift.py
"""
import math, sys, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from simulate_kalshi_historical import (
    load_signals, scanner_cycle, make_ticker,
    BANKROLL, FEE_RATE, MAX_TRADES_EXPIRY, DAILY_LOSS_LIMIT,
    SCAN_INTERVAL_MIN, STRIKE_INCREMENT, RES_DIR,
)

START = "2024-06-01"
END   = "2026-05-26"

# ── k values to sweep ─────────────────────────────────────────────────────────
K_VALUES = [-0.60, -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05,
             0.00,
             0.05,  0.10,  0.15,  0.20,  0.30,  0.40,  0.50,  0.60]


def simulate_one_k(sigs, df1m_by_hour, k):
    """Run the full simulation for a single k value. Returns summary dict."""
    trade_rows = []
    daily_pnl  = 0.0
    current_date = None
    trades_expiry = 0
    open_trades   = []
    cooldown_yes_until = None
    cooldown_no_until  = None

    hours = sigs.index

    for hi, bar_ts in enumerate(hours):
        row      = sigs.loc[bar_ts]
        date     = bar_ts.date()
        hour_utc = bar_ts.hour

        if date != current_date:
            daily_pnl    = 0.0
            current_date = date

        # Settle previous hour's open trades
        for tr in open_trades:
            prev_hour    = bar_ts - pd.Timedelta(hours=1)
            settle_price = (float(df1m_by_hour[prev_hour]["close"].iloc[-1])
                            if prev_hour in df1m_by_hour
                            else float(row["close"]))
            resolved = 1 if settle_price > tr["strike"] else 0
            if tr["side"] == "yes":
                won = resolved == 1
                pnl = (tr["n_cont"] * (1 - tr["p_market"] -
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])) if won
                       else -tr["n_cont"] * (tr["p_market"] +
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])))
            else:
                won = resolved == 0
                pnl = (tr["n_cont"] * (tr["p_market"] -
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])) if won
                       else -tr["n_cont"] * ((1 - tr["p_market"]) +
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])))
            pnl = round(pnl, 2)
            daily_pnl += pnl
            tr["pnl"] = pnl
            tr["won"] = won
            trade_rows.append(tr)

        open_trades        = []
        trades_expiry      = 0
        cooldown_yes_until = None
        cooldown_no_until  = None

        if bar_ts not in df1m_by_hour:
            continue
        m1_slice = df1m_by_hour[bar_ts]
        if len(m1_slice) < 3:
            continue

        # Drift for this bar
        p_up     = float(row["p_up_v2"])
        rvol_inv = float(row["rvol_inv"])
        if k == 0.0:
            z_drift_const = 0.0
        else:
            pup_z = float(norm.ppf(max(0.01, min(0.99, p_up))))
            z_drift_const = pup_z * rvol_inv * k

        expiry_ts = bar_ts + pd.Timedelta(hours=1)

        for scan_min in range(SCAN_INTERVAL_MIN, 60 - 1, SCAN_INTERVAL_MIN):
            tau_min = 60 - scan_min
            scan_ts = bar_ts + pd.Timedelta(minutes=scan_min)
            m1_before = m1_slice[m1_slice.index <= scan_ts]
            spot = (float(m1_before["close"].iloc[-1])
                    if not m1_before.empty else float(row["close"]))

            cd_yes = cooldown_yes_until is not None and scan_ts < cooldown_yes_until
            cd_no  = cooldown_no_until  is not None and scan_ts < cooldown_no_until

            result = scanner_cycle(
                spot=spot, tau_min=tau_min, row=row,
                trades_this_expiry=trades_expiry,
                daily_pnl=daily_pnl,
                cooldown_yes=cd_yes, cooldown_no=cd_no,
                hour_utc=hour_utc, z_drift_const=z_drift_const,
            )

            if result["decision"] == "trade":
                rec = {
                    "strike":    result["strike"],
                    "side":      result["side"],
                    "p_market":  result["p_market"],
                    "n_cont":    result["n_cont"],
                    "pnl":       float("nan"),
                    "won":       False,
                }
                open_trades.append(rec)
                trades_expiry += 1
                cd_end = scan_ts + pd.Timedelta(minutes=5)
                if result["side"] == "yes":
                    cooldown_yes_until = cd_end
                else:
                    cooldown_no_until  = cd_end

    if not trade_rows:
        return {"k": k, "pnl": 0.0, "n": 0, "wr": float("nan")}

    pnl_total = sum(r["pnl"] for r in trade_rows)
    wins      = sum(1 for r in trade_rows if r["won"])
    wr        = wins / len(trade_rows)

    # Quarterly breakdown
    qtr_pnl = {}
    for r in trade_rows:
        pass  # we'll compute quarterly from the full list at the end

    return {"k": k, "pnl": round(pnl_total, 2), "n": len(trade_rows),
            "wr": round(wr, 4), "trades": trade_rows}


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading signals (GARCH ~2 min)...")
    sigs, df1m = load_signals(START, END)

    print("Indexing 1m data by hour...")
    df1m_by_hour = {}
    for ts, grp in df1m.groupby(df1m.index.floor("1h")):
        df1m_by_hour[ts] = grp

    results = []
    print(f"\nSweeping {len(K_VALUES)} k values...\n")
    print(f"  {'k':>6}  {'P&L':>12}  {'Trades':>8}  {'WR':>7}  {'Time':>6}")
    print("  " + "-" * 50)

    for k in K_VALUES:
        t0  = time.time()
        res = simulate_one_k(sigs, df1m_by_hour, k)
        elapsed = time.time() - t0
        print(f"  {k:>+6.2f}  {res['pnl']:>+12,.0f}  {res['n']:>8,}  "
              f"{res['wr']:>7.1%}  {elapsed:>5.1f}s")
        results.append(res)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  K-DRIFT SWEEP SUMMARY")
    print(f"{'='*58}")
    print(f"  {'k':>6}  {'P&L':>12}  {'Trades':>8}  {'WR':>7}")
    print("  " + "-" * 40)
    best = max(results, key=lambda r: r["pnl"])
    for r in results:
        marker = " ←best" if r["k"] == best["k"] else ""
        print(f"  {r['k']:>+6.2f}  {r['pnl']:>+12,.0f}  {r['n']:>8,}  "
              f"{r['wr']:>7.1%}{marker}")
    print(f"\n  Best k = {best['k']:+.2f}  →  ${best['pnl']:+,.0f}")

    # ── quarterly breakdown for best k ────────────────────────────────────────
    if best["trades"]:
        import datetime
        print(f"\n  Quarterly P&L at k={best['k']:+.2f}:")
        qmap = {}
        for tr in best["trades"]:
            pass  # no date stored in minimal rec — skip quarterly for now

    # ── write sweep results ───────────────────────────────────────────────────
    df_res = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != "trades"}
                            for r in results])
    out = RES_DIR / "sweep_k_results.csv"
    df_res.to_csv(out, index=False)
    print(f"\n  Wrote {out}")
