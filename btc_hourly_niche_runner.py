"""BTC hourly YES-niche paper runner — 2026-07-28.

Paper-trades the ONE surviving BTC hourly edge found by the fee-audit retrain
(project_hourly_fee_audit_phase0_20260728.md): retrained LGBM
(models/btc_hourly_lgbm_niche_20260728.pkl, trained on btc_scan_archive
05-18..06-19, early-stop 06-20..30) applied to mid-pm YES only.

Rule (validated on the untouched July holdout, flat $100/contract, one bet
per contract, net of Kalshi fees):
    side = YES only
    p_market in [0.35, 0.65]
    fee-adjusted model edge = p_model - pm - 0.07*pm*(1-pm) >= 0.06
Holdout: +$5,043 over 261 trades (~6.7/day), WR 55.2% vs BE 45.1%,
positive in ALL 6 weeks (incl. weeks 29-30 that lost -$15.7k broad-book).
Controls: ALL mid-pm YES (no model) = -$8,113; model-says-avoid = -$9,126
=> genuine model separation, not regime beta. NO-side model signal remains
anti-predictive -- deliberately NOT traded.

Design: reads the tail of results/btc_scan_archive.csv (written by the main
hourly runner every scan with all features) so the live pipeline is byte-
identical to the validation data. No edits to paper_trade_runner.py; own
book results/paper_trades_btc_hourly_niche.csv; flat $100 stakes (matches
validation -- NOT Kelly). Resolution joined back from the archive's own
resolved_yes backfill.
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

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "btc_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_btc_hourly_niche.csv"
STATE = BASE / "results" / ".btc_hourly_niche_state.json"
MODEL = BASE / "models" / "btc_hourly_lgbm_niche_20260728.pkl"

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
             "p_market", "p_model", "fee_adj_edge", "tau_minutes", "stake",
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
    changed = 0
    for i in book[pend].index:
        rv = res_map.get(book.at[i, "contract_ticker"])
        if rv is None or (isinstance(rv, float) and rv != rv):
            continue
        rv = int(float(rv))
        pm = float(book.at[i, "p_market"])
        stake = float(book.at[i, "stake"])
        win = rv == 1  # YES-only book
        gross = round(stake * (1 - pm) / pm, 2) if win else -stake
        fee = round((stake / pm) * 0.07 * pm * (1 - pm), 2)
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
                    # [2026-08-22 MKV-SIDEWAYS BLOCK, user-approved] Block
                    # + consume when markov_daily_regime == "Sideways".
                    # Cross-asset frozen rule (SOL 08-06 -$1,041/48 P=.07,
                    # ETH -$807/42 P=.04, pre-registered universal
                    # candidate awaiting BTC testability); BTC's first
                    # Sideways regime = the 08-22 run, gate fires 35/35,
                    # blocks the entire -$1,177; pre-run BTC cost +$147/6.
                    # The niche thesis needs a trending tape.
                    _mkv = hits.get("markov_daily_regime")
                    if _mkv is not None:
                        _blk = hits[_mkv.astype(str) == "Sideways"]
                        for _, _br in _blk.iterrows():
                            traded.add(_br["contract_ticker"])
                            print(f"  [mkv-block] {_br['contract_ticker']} "
                                  f"daily=Sideways (consumed)")
                        hits = hits[_mkv.astype(str) != "Sideways"]
                    for _, r in hits.iterrows():
                        append_trade({
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "contract_ticker": r["contract_ticker"],
                            "close_ts": r.get("close_ts", ""),
                            "spot": r["spot"], "strike": r["strike"],
                            "p_market": round(float(r["p_market"]), 4),
                            "p_model": round(float(r["p_model"]), 4),
                            "fee_adj_edge": round(float(r["fee_adj_edge"]), 4),
                            "tau_minutes": r["tau_minutes"], "stake": STAKE,
                            "markov_daily_regime": r.get("markov_daily_regime", ""),
                            "resolved_yes": "", "would_win": "", "would_pnl": "",
                            "fee_est": "", "would_pnl_net": "",
                        })
                        traded.add(r["contract_ticker"])
                        print(f"  [TRADE] YES {r['contract_ticker']} pm={r['p_market']:.3f} "
                              f"p_model={r['p_model']:.3f} edge={r['fee_adj_edge']:.3f} "
                              f"tau={r['tau_minutes']:.0f}m")
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
