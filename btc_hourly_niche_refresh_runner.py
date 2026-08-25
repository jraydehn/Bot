"""BTC hourly niche REFRESH challenger runner — 2026-08-20.

Races the frozen niche model (btc_hourly_niche_runner.py), which went
silent 08-13 when BTC broke out of its 05-18..06-19 training range (zero
fires in 8 days vs validated ~6.7/day = the staleness tell; refused band
population +$2,854/338).

Model: btc_hourly_lgbm_niche_refresh_20260820.pkl — IDENTICAL recipe,
training extended to 07-31, early-stop val 08-01..09, seed 5 of 6
(median-val selection). UNTOUCHED 08-10+ holdout: all 6 seeds positive
(band +923..+2,743, median +1,310, WR 60-68%); fires in the drought
window every seed (+598..+2,246). DISCLOSED CAVEAT: the walk-forward
July fold is NEGATIVE (-$1,282/487) — recipe is regime-sensitive; this
paper race is the arbiter, not the holdout.

Rule: mirrors the incumbent's CURRENT config exactly (NICHE-CAL band
pm 0.32-0.45, edge 0.06-0.20, YES only, flat $100) so the model is the
only difference. If the incumbent's band reverts at its 08-25 referee
read, mirror the change here.

LIVE-FILL accounting per feedback_live_fill_paper_books: fills at the
executable ask (cap = signal cost + 1c), filled=0 + zero PnL when the
price ran away. First read ~2026-09-03.
"""
import json
import os
import pickle
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from hourly_live_fill import load_auth_or_die, fill_for

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "btc_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_btc_hourly_niche_refresh.csv"
STATE = BASE / "results" / ".btc_hourly_niche_refresh_state.json"
MODEL = BASE / "models" / "btc_hourly_lgbm_niche_refresh_20260820.pkl"

EDGE_MIN = 0.06
# [2026-08-18 NICHE-CAL wired per user: band floor was misplaced
# (0.32-0.35 = +13% margin/3wks excluded), band top dead (0.45-0.65),
# edge>0.20 over-claims (-28% margin). Referee book: n=72, 54% WR vs
# 38% BE, +$2,523, 3/3 wks. Revert at 08-25 if the live rule's referee
# stream out-ranks it. Prior: PM 0.35-0.65, no edge cap.]
PM_LO, PM_HI = 0.32, 0.45
EDGE_MAX = 0.20
STAKE = 100.0
TAIL_BYTES = 2_000_000  # ~6k archive rows, >> one loop of new scans
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "contract_ticker", "close_ts", "spot", "strike",
             "p_market", "fill_price", "filled", "p_model", "fee_adj_edge", "tau_minutes", "stake",
             "resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net",
             # [2026-08-22] appended at END (column-order lesson 08-21)
             "markov_daily_regime"]

with open(MODEL, "rb") as f:
    _art = pickle.load(f)
model, FEATS = _art["model"], _art["features"]


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
    return df.dropna(subset=["dt"])


def prep(df: pd.DataFrame) -> pd.DataFrame:
    for c in set(FEATS + ["p_market", "tau_minutes", "strike", "spot", "resolved_yes"]) - {"z_moneyness"}:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(df["tau_minutes"].clip(lower=1))
    return df


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
    """Fill outcomes for expired trades from the archive's resolved_yes backfill."""
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
        win = rv == 1  # YES-only book
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
        print(f"  [resolve] filled {changed} outcome(s); book net so far: "
              f"{pd.to_numeric(book['would_pnl_net'], errors='coerce').sum():+.2f}")


def main() -> None:
    ensure_book()
    auth = load_auth_or_die("[niche-refresh]")
    st = load_state()
    traded = set(st["traded"])
    print(f"[niche] BTC hourly YES-niche paper runner up. rule: YES, pm∈[{PM_LO},{PM_HI}], "
          f"fee-adj edge>={EDGE_MIN}, flat ${STAKE:.0f}. {len(traded)} tickers already traded.")
    while True:
        try:
            arch = prep(read_archive_tail())
            new = arch[arch["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                cand = new[new["p_market"].between(PM_LO, PM_HI)].copy()
                cand = cand[~cand["contract_ticker"].isin(traded)]
                if len(cand):
                    p = model.predict_proba(cand[FEATS])[:, 1]
                    fee = 0.07 * cand["p_market"] * (1 - cand["p_market"])
                    edge = p - cand["p_market"] - fee
                    hits = cand[(edge >= EDGE_MIN) & (edge < EDGE_MAX)].copy()
                    hits["p_model"] = p[(edge >= EDGE_MIN) & (edge < EDGE_MAX)]
                    hits["fee_adj_edge"] = edge[(edge >= EDGE_MIN) & (edge < EDGE_MAX)]
                    hits = hits.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
                    # [2026-08-22 MKV-SIDEWAYS BLOCK — see v1 runner comment]
                    _mkv = hits.get("markov_daily_regime")
                    if _mkv is not None:
                        _blk = hits[_mkv.astype(str) == "Sideways"]
                        for _, _br in _blk.iterrows():
                            traded.add(_br["contract_ticker"])
                            print(f"  [mkv-block] {_br['contract_ticker']} "
                                  f"daily=Sideways (consumed)")
                        hits = hits[_mkv.astype(str) != "Sideways"]
                    for _, r in hits.iterrows():
                        pm_sig = float(r["p_market"])
                        fill, ok = fill_for(auth, r["contract_ticker"],
                                            "yes", pm_sig + 0.01)
                        append_trade({
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "contract_ticker": r["contract_ticker"],
                            "close_ts": r.get("close_ts", ""),
                            "spot": r["spot"], "strike": r["strike"],
                            "p_market": round(pm_sig, 4),
                            "fill_price": round(fill, 4) if fill is not None else "",
                            "filled": int(ok),
                            "p_model": round(float(r["p_model"]), 4),
                            "fee_adj_edge": round(float(r["fee_adj_edge"]), 4),
                            "tau_minutes": r["tau_minutes"], "stake": STAKE,
                            "markov_daily_regime": r.get("markov_daily_regime", ""),
                            "resolved_yes": "", "would_win": "", "would_pnl": "",
                            "fee_est": "", "would_pnl_net": "",
                        })
                        traded.add(r["contract_ticker"])
                        tag = "TRADE" if ok else "NOFILL"
                        fs = f"{fill:.3f}" if fill is not None else "none"
                        print(f"  [{tag}] YES {r['contract_ticker']} pm={pm_sig:.3f} "
                              f"ask={fs} p_model={r['p_model']:.3f} "
                              f"edge={r['fee_adj_edge']:.3f} tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-3000:]
                save_state(st)
            resolve_pending(arch)
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        # [2026-08-15] heartbeat: these runners are silent on uneventful
        # cycles, so the log-mtime watchdog (30min) churn-restarted them
        # while idle. One [hb] line per ~10min keeps mtime honest —
        # silence now means FROZEN, not quiet.
        global _last_hb
        try:
            _last_hb
        except NameError:
            _last_hb = 0
        if time.time() - _last_hb >= 600:
            print("[hb]", flush=True)
            _last_hb = time.time()
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
