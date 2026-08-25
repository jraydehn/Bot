"""SOL hourly YES-favorite paper runner — 2026-08-19.

SOL replication of the ETH hourly favorite bias (eth_hourly_fav_runner.py,
tracker 08-18): band-map on sol_scan_archive (579k scans, 05-21..08-19)
shows the same structural underpricing of YES favorites, and it is the
ONLY unconditionally positive band on either side below 0.97:

    side = YES only (no model, no edge computation)
    p_market in [0.80, 0.97]
    flat $100, one bet per contract (first scan entering the band)

    May-Jun +$3,281/882 | July +$3,510/1,930 | Aug-fwd +$5,670/1,441
    WR 93.5% vs BE 90.8% | full mcP 0.00000 | 11/13 weeks green
    TICK-fill stress (buy at next cent up): +$9,206, all windows green
    (retains 74% — same profile as ETH's 77%; the 0.97+ band was NOT
    extended per the ETH tick-mirage lesson).

Context: SOL hourly model campaign = 6 retrain nulls + v7/v8 retired
08-12; this model-free book is the strongest known SOL hourly strategy.
Consistent with the asymmetry doctrine: SOL 15m mkt-fav failed (-$4,063)
— the HOURLY market carries the bias, as on ETH.

Same MID-price fill caveat as the ETH book (disclosed, survives tick
stress). Same-expiry strikes correlate — expect clustered losses.

Design: byte-identical pipeline to eth_hourly_fav_runner.py — archive
tail, own book results/paper_trades_sol_hourly_fav.csv, resolution from
the archive's resolved_yes backfill. First read ~2026-09-01.
"""
import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from hourly_live_fill import load_auth_or_die, fill_for

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "sol_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_sol_hourly_fav.csv"
STATE = BASE / "results" / ".sol_hourly_fav_state.json"

PM_LO, PM_HI = 0.80, 0.97
STAKE = 100.0
TAIL_BYTES = 2_000_000
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "contract_ticker", "close_ts", "spot", "strike",
             "p_market", "fill_price", "filled", "tau_minutes", "stake",
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
    for c in ["p_market", "tau_minutes", "strike", "spot", "resolved_yes"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.dropna(subset=["dt"])


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_ts": "1970-01-01T00:00:00+00:00", "traded": []}


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
    # [2026-08-25 STALE-PENDING FALLBACK (user-caught: 4 filled ETH fav
    # trades stuck >12h): a resolution landing while the runner is down or
    # after the row scrolls out of the archive TAIL window was unreachable
    # forever (2MB tail ~= a few hours on the fast archives). Targeted
    # chunked full-archive lookup for ONLY the still-missing pending
    # tickers; self-clears once the book is clean.]
    _missing = {book.at[_i, "contract_ticker"] for _i in book[pend].index
                if book.at[_i, "contract_ticker"] not in res_map}
    if _missing:
        try:
            for _fch in pd.read_csv(ARCHIVE,
                                    usecols=["contract_ticker",
                                             "resolved_yes"],
                                    chunksize=500_000, low_memory=False,
                                    on_bad_lines="skip"):
                _fhit = _fch[_fch["contract_ticker"].isin(_missing)
                             & _fch["resolved_yes"].notna()]
                for _ft, _frv in zip(_fhit["contract_ticker"],
                                     _fhit["resolved_yes"]):
                    res_map[_ft] = _frv
        except Exception:
            pass
    changed = 0
    for i in book[pend].index:
        rv = res_map.get(book.at[i, "contract_ticker"])
        if rv is None or (isinstance(rv, float) and rv != rv):
            continue
        rv = int(float(rv))
        stake = float(book.at[i, "stake"])
        filled = pd.to_numeric(book.at[i, "filled"], errors="coerce")
        fill = pd.to_numeric(book.at[i, "fill_price"], errors="coerce")
        win = rv == 1  # YES-only book
        if filled == 1 and fill == fill and 0 < fill < 1:
            gross = round(stake * (1 - fill) / fill, 2) if win else -stake
            fee = round((stake / fill) * 0.07 * fill * (1 - fill), 2)
        else:  # signal fired but no executable fill — zero PnL row
            gross = 0.0
            fee = 0.0
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
        print(f"  [resolve] filled {changed} outcome(s); book net so far: "
              f"{pd.to_numeric(book['would_pnl_net'], errors='coerce').sum():+.2f}")


def main() -> None:
    ensure_book()
    auth = load_auth_or_die("[fav]")
    st = load_state()
    traded = set(st["traded"])
    print(f"[fav] SOL hourly YES-favorite paper runner up. rule: YES, "
          f"pm in [{PM_LO},{PM_HI}], flat ${STAKE:.0f}, model-free. "
          f"{len(traded)} tickers already traded.")
    last_hb = 0.0
    while True:
        try:
            arch = read_archive_tail()
            new = arch[arch["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                hits = new[new["p_market"].between(PM_LO, PM_HI)].copy()
                hits = hits[~hits["contract_ticker"].isin(traded)]
                hits = hits.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
                for _, r in hits.iterrows():
                    fill, ok = fill_for(auth, r["contract_ticker"], "yes",
                                        PM_HI + 0.01)
                    append_trade({
                        "logged_at": datetime.now(timezone.utc).isoformat(),
                        "contract_ticker": r["contract_ticker"],
                        "close_ts": r.get("close_ts", ""),
                        "spot": r["spot"], "strike": r["strike"],
                        "p_market": round(float(r["p_market"]), 4),
                        "fill_price": round(fill, 4) if fill is not None else "",
                        "filled": int(ok),
                        "tau_minutes": r["tau_minutes"], "stake": STAKE,
                        "resolved_yes": "", "would_win": "", "would_pnl": "",
                        "fee_est": "", "would_pnl_net": "",
                    })
                    traded.add(r["contract_ticker"])
                    tag = "TRADE" if ok else "NOFILL"
                    fs = f"{fill:.3f}" if fill is not None else "none"
                    print(f"  [{tag}] YES {r['contract_ticker']} pm={r['p_market']:.3f} "
                          f"ask={fs} tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-3000:]
                save_state(st)
            resolve_pending(arch)
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        # heartbeat — same rationale as the niche runners (silent-idle
        # runners triggered watchdog churn-restarts; [hb] keeps log mtime
        # honest so silence means FROZEN).
        if time.time() - last_hb >= 600:
            print("[hb]", flush=True)
            last_hb = time.time()
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
