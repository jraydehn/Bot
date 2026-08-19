"""ETH hourly fav-RESCUES paper runner — 2026-08-19.

Paper-trades the two validated rescue bands OUTSIDE the live fav book's
YES 0.80-0.97 range (analysis/eth_hourly_rescue_sweep.py + _validate.py,
user-approved 08-19). Selection on May-Jun+July only; Aug was the
untouched confirm; mechanical features only (no model-derived signals per
feedback_derived_signal_needs_model_oos):

  B: YES, pm in [0.70, 0.80), liq_bias <= 0
     (buy-the-dip favorite — replicates BTC niche thesis on ETH)
     May-Jun +$1,049/256 | July +$1,336/499 | Aug +$2,588/311
     WR 79.6%, full mcP=0.00015, threshold-robust liq_bias -1..+0.5,
     survives flat 1c adverse fill in every window (+$3,499 total).

  C: NO, pm in [0.20, 0.40), vwap_stretch_score <= -1
     (price stretched below VWAP — momentum-confirmed NO favorite)
     May-Jun +$972/114 | July +$1,612/359 | Aug +$2,352/246
     WR 76.8%, full mcP=0.00010, monotone in stretch (<=-2 +$1.3k,
     <=-1 +$4.2k, <=0 -$67), survives 1c fill stress (+$3,852).

  REJECTED: A (YES 0.97-0.995) — mid-price mirage; biggest pocket (0.99+)
  untradeable at tick, rest dies on one cent of crossing (July -$87 at
  tick fills). D (NO 0.03-0.20 stretch<=-1) — Aug mcP=0.15, negative
  under slip. YES 0.50-0.70 — no rule beat +$244 selection-min (noise).

  Same MID-price fill caveat as the fav book (B/C margins ~4.7/6.9 $/trade
  survive it; disclosed, not modeled). Same-expiry strikes correlate.

Design: byte-identical pipeline to eth_hourly_fav_runner.py — archive
tail, flat $100, one bet per contract PER BAND (first qualifying scan),
own book results/paper_trades_eth_hourly_fav_rescues.csv (band+side
columns), resolution joined from the archive's resolved_yes backfill.
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

from hourly_live_fill import load_auth_or_die, fill_for

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "eth_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_eth_hourly_fav_rescues.csv"
STATE = BASE / "results" / ".eth_hourly_fav_rescues_state.json"

STAKE = 100.0
TAIL_BYTES = 2_000_000
LOOP_SEC = 120

# band: (side, pm_lo, pm_hi, condition_col, condition(pd.Series) -> mask)
BANDS = {
    "B": ("yes", 0.70, 0.80, "liq_bias", lambda s: s <= 0.0),
    "C": ("no", 0.20, 0.40, "vwap_stretch_score", lambda s: s <= -1.0),
}

BOOK_COLS = ["logged_at", "band", "side", "contract_ticker", "close_ts",
             "spot", "strike", "p_market", "fill_price", "filled", "tau_minutes", "stake",
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
              "liq_bias", "vwap_stretch_score"]:
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
        stake = float(book.at[i, "stake"])
        side = str(book.at[i, "side"])
        win = (rv == 1) if side == "yes" else (rv == 0)
        filled = pd.to_numeric(book.at[i, "filled"], errors="coerce")
        fill = pd.to_numeric(book.at[i, "fill_price"], errors="coerce")
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
        net = pd.to_numeric(book["would_pnl_net"], errors="coerce")
        by_band = book.assign(_n=net).groupby("band")["_n"].sum().to_dict()
        print(f"  [resolve] filled {changed} outcome(s); book net so far: "
              f"{net.sum():+.2f} ({', '.join(f'{b} {v:+.0f}' for b, v in sorted(by_band.items()))})")


def main() -> None:
    ensure_book()
    auth = load_auth_or_die("[rescues]")
    st = load_state()
    traded = {b: set(st["traded"].get(b, [])) for b in BANDS}
    print(f"[rescues] ETH hourly fav-rescues paper runner up. bands: "
          f"B=YES[0.70,0.80) liq_bias<=0, C=NO[0.20,0.40) stretch<=-1, "
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
                        # [2026-08-19] fee-aware fill cap (user directive):
                        # anchor to the SIGNAL cost + the validated 1c slip
                        # stress, not the band edge — B/C fee margins (~3.6/4pp)
                        # clear fees at 1c but a band-edge cap could admit
                        # multi-cent slippage on early-band signals.
                        pm_sig = float(r["p_market"])
                        cost_sig = pm_sig if side == "yes" else 1.0 - pm_sig
                        fill, ok = fill_for(auth, r["contract_ticker"], side,
                                            cost_sig + 0.01)
                        append_trade({
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "band": bname, "side": side,
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
                        traded[bname].add(r["contract_ticker"])
                        tag = "TRADE" if ok else "NOFILL"
                        fs = f"{fill:.3f}" if fill is not None else "none"
                        print(f"  [{tag}:{bname}] {side.upper()} {r['contract_ticker']} "
                              f"pm={r['p_market']:.3f} cost={fs} {ccol}={r[ccol]:.3g} "
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
