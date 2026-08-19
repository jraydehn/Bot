"""SOL hourly fav-RESCUES paper runner — 2026-08-19.

SOL replication of the ETH rescue-band process (same scripts, asset arg:
analysis/eth_hourly_rescue_sweep.py sol + inline validation). Selection
on May-Jun+July only, Aug untouched confirm, mechanical features only.
SOL picked its own conditions (asset-specific doctrine — ETH's liq_bias
rule is NOT the SOL survivor):

  B: YES, pm in [0.70, 0.80), vwap_stretch_score >= 0
     (favorite holding at/above VWAP) +939/+1,221/+901; WR 78.3%;
     mcP 0.0034; natural threshold boundary (>=-1 collapses to +423);
     survives 1c fill stress incl. Aug (+483).
  C: NO, pm in [0.20, 0.40), tau_minutes <= 21.5
     (short-tau theta play; coheres w/ SOL short-tau structure)
     +1,190/+995/+454; mcP 0.011; neighbor-stable tau 15-25;
     WEAKEST FLAG: Aug at 1c slip only +55 (stoch/stretch variants
     failed that bar outright and were rejected).
  M: YES, pm in [0.50, 0.70), oi_chg_pct <= -0.1178
     (deep OI drop + mid favorite; band ETH had no survivor in)
     +514/+569/+943 (Aug = untouched confirm = strongest window,
     +691 slipped); mcP 0.034; monotone deepening to -0.2 but cliff
     at -0.06 — flagged, the forward read adjudicates.

  REJECTED: NO 0.03-0.20 (193 sweep passes, nearly ALL Aug-negative =
  regime shift, no rescue); YES 0.97+ (ETH tick-mirage lesson, not
  extended). MID-price caveat as usual.

Design: identical pipeline to eth_hourly_fav_rescues_runner.py — archive
tail, flat $100, one bet per contract PER BAND, own book
results/paper_trades_sol_hourly_fav_rescues.csv (band+side cols).
Pre-registered first read ~2026-09-01. NOT on the dashboard.
"""
import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "sol_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_sol_hourly_fav_rescues.csv"
STATE = BASE / "results" / ".sol_hourly_fav_rescues_state.json"

STAKE = 100.0
TAIL_BYTES = 2_000_000
LOOP_SEC = 120

# band: (side, pm_lo, pm_hi, condition_col, condition(pd.Series) -> mask)
BANDS = {
    "B": ("yes", 0.70, 0.80, "vwap_stretch_score", lambda s: s >= 0.0),
    "C": ("no", 0.20, 0.40, "tau_minutes", lambda s: s <= 21.5),
    "M": ("yes", 0.50, 0.70, "oi_chg_pct", lambda s: s <= -0.1178),
}

BOOK_COLS = ["logged_at", "band", "side", "contract_ticker", "close_ts",
             "spot", "strike", "p_market", "tau_minutes", "stake",
             "resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net"]


def read_archive_tail() -> pd.DataFrame:
    with open(ARCHIVE, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - TAIL_BYTES))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    df = pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                              errors="coerce", utc=True, format="mixed")
    for c in ["p_market", "tau_minutes", "strike", "spot", "resolved_yes",
              "oi_chg_pct", "vwap_stretch_score"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.dropna(subset=["dt"])


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_ts": "1970-01-01T00:00:00+00:00",
            "traded": {b: [] for b in BANDS}}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st))


def ensure_book() -> None:
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")


def append_trade(row: dict) -> None:
    import csv
    with open(BOOK, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=BOOK_COLS, extrasaction="ignore").writerow(row)


def resolve_pending(arch: pd.DataFrame) -> None:
    book = pd.read_csv(BOOK, low_memory=False)
    if book.empty:
        return
    pend = book["resolved_yes"].isna()
    if not pend.any():
        return
    res_map = (arch.dropna(subset=["resolved_yes"])
               .drop_duplicates("contract_ticker", keep="last")
               .set_index("contract_ticker")["resolved_yes"].to_dict())
    changed = 0
    for i in book[pend].index:
        rv = res_map.get(book.at[i, "contract_ticker"])
        if rv is None or (isinstance(rv, float) and rv != rv):
            continue
        rv = int(float(rv))
        pm = float(book.at[i, "p_market"])
        stake = float(book.at[i, "stake"])
        side = str(book.at[i, "side"])
        win = (rv == 1) if side == "yes" else (rv == 0)
        cost = pm if side == "yes" else 1.0 - pm
        gross = round(stake * (1 - cost) / cost, 2) if win else -stake
        fee = round((stake / cost) * 0.07 * pm * (1 - pm), 2)
        book.at[i, "resolved_yes"] = rv
        book.at[i, "would_win"] = int(win)
        book.at[i, "would_pnl"] = gross
        book.at[i, "fee_est"] = fee
        book.at[i, "would_pnl_net"] = round(gross - fee, 2)
        changed += 1
    if changed:
        tmp = BOOK.with_suffix(".csv.tmp")
        book.to_csv(tmp, index=False)
        os.replace(tmp, BOOK)
        net = pd.to_numeric(book["would_pnl_net"], errors="coerce")
        by_band = book.assign(_n=net).groupby("band")["_n"].sum().to_dict()
        print(f"  [resolve] filled {changed} outcome(s); book net so far: "
              f"{net.sum():+.2f} ({', '.join(f'{b} {v:+.0f}' for b, v in sorted(by_band.items()))})")


def main() -> None:
    ensure_book()
    st = load_state()
    traded = {b: set(st["traded"].get(b, [])) for b in BANDS}
    print(f"[rescues] SOL hourly fav-rescues paper runner up. bands: "
          f"B=YES[0.70,0.80) stretch>=0, C=NO[0.20,0.40) tau<=21.5, "
          f"M=YES[0.50,0.70) oi_chg<=-0.1178, "
          f"flat ${STAKE:.0f}. already traded: "
          + ", ".join(f"{b}={len(s)}" for b, s in traded.items()))
    last_hb = 0.0
    while True:
        try:
            arch = read_archive_tail()
            new = arch[arch["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                for bname, (side, lo, hi, ccol, cfn) in BANDS.items():
                    m = new["p_market"].ge(lo) & new["p_market"].lt(hi) & cfn(new[ccol])
                    hits = new[m & ~new["contract_ticker"].isin(traded[bname])]
                    hits = hits.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
                    for _, r in hits.iterrows():
                        append_trade({
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "band": bname, "side": side,
                            "contract_ticker": r["contract_ticker"],
                            "close_ts": r.get("close_ts", ""),
                            "spot": r["spot"], "strike": r["strike"],
                            "p_market": round(float(r["p_market"]), 4),
                            "tau_minutes": r["tau_minutes"], "stake": STAKE,
                            "resolved_yes": "", "would_win": "", "would_pnl": "",
                            "fee_est": "", "would_pnl_net": "",
                        })
                        traded[bname].add(r["contract_ticker"])
                        print(f"  [TRADE:{bname}] {side.upper()} {r['contract_ticker']} "
                              f"pm={r['p_market']:.3f} {ccol}={r[ccol]:.3g} "
                              f"tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = {b: sorted(s)[-3000:] for b, s in traded.items()}
                save_state(st)
            resolve_pending(arch)
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        if time.time() - last_hb >= 600:
            print("[hb]", flush=True)
            last_hb = time.time()
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
