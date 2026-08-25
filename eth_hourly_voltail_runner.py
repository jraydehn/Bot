"""ETH hourly VOL-TAIL paper runner — single-leg variant, low-frequency
collector. 2026-08-04.

ETH's bookdyn replacement. Same axis as the BTC vol-tail book (predicted
width from the bookdyn tail quantiles vs ladder implied width) with the
pre-declared DESIGN adaptation for ETH's sparse ladder: single tail legs
allowed (buy whichever qualifying tail rung exists), since requiring both
rungs starved the book (~5 straddles/month).

Evidence status (disclosed): the signal's RELATIVE pattern is real on ETH
(screen: +$10,820 on 30 legs vs controls −$9k/−$18k; same ordering as
BTC's two-window result) but n never reached the backtest evidence bar —
THIS RUNNER EXISTS TO ACCRUE THAT SAMPLE. Expect ~2-4 events/week; first
meaningful read ~mid-September, not the 08-18 cycle.

PRE-REGISTERED RULE: vratio >= 1.2 → on the event's first qualifying
loop, buy ANY tail rung available: YES leg at ASK in [0.03,0.15] and/or
NO leg with yes_bid in [0.85,0.97]. $100/leg at the ASK, fees on top,
mid logged per leg for slippage measurement.
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

import sol_hourly_banked_signals as bank
import sol_hourly_crossasset_flow as xa
from live_signal import load_auth, kalshi_get
from train_eth_hourly_bookdyn import assemble

BASE = Path(__file__).parent
BOOK = BASE / "results" / "paper_trades_eth_hourly_voltail.csv"
STATE = BASE / "results" / ".eth_hourly_voltail_state.json"
MODEL_P = BASE / "models" / "eth_hourly_bookdyn_20260731.pkl"

VRATIO_MIN = 1.2
STAKE = 100.0
ETH_TAIL = 14_000_000
BTC_TAIL = 6_000_000
LOOP_SEC = 120
Q_LO, Q_HI = 0.15, 0.85

BOOK_COLS = ["logged_at", "event", "side", "contract_ticker", "close_ts",
             "spot", "strike", "vratio", "pred_width", "imp_width",
             "cost_ask", "cost_mid", "stake", "resolved_yes", "would_win",
             "would_pnl", "fee_est", "would_pnl_net"]


def read_tail(asset: str, nbytes: int) -> pd.DataFrame:
    path = BASE / "results" / f"{asset}_scan_archive.csv"
    with open(path, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - nbytes))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    return pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    for c in ["p_market", "spot", "strike", "tau_minutes", "resolved_yes",
              "price_move_pct", "offset_pct", "composite_p_up"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_ts"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["p_market"])
    return df[df["p_market"].between(0.02, 0.98)].reset_index(drop=True)


def build_liq():
    liq = bank.fetch_liq_bars_live(asset="ETH").copy()
    tot = liq["liq_long"] + liq["liq_short"]
    liq["liq_total_z"] = tot.rolling(168).rank(pct=True)
    liq["liq_long_z"] = liq["liq_long"].rolling(168).rank(pct=True)
    liq["liq_short_z"] = liq["liq_short"].rolling(168).rank(pct=True)
    liq["liq_imbalance"] = (liq["liq_long"] - liq["liq_short"]) / tot.replace(0, np.nan)
    liq["liq_imbalance_trend6"] = (liq["liq_imbalance"] - liq["liq_imbalance"].shift(6)) / 6
    return liq.drop(columns=["liq_long", "liq_short"])


def scale_price(v):
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def main() -> None:
    auth = load_auth()
    with open(MODEL_P, "rb") as f:
        art = pickle.load(f)
    models = art["models"]
    FEATS = art["features"]
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else {"traded": []}
    traded = set(st["traded"])
    print(f"[eth-voltail] up. vratio>={VRATIO_MIN}, single-leg allowed, "
          f"${STAKE:.0f}/leg at ASK. {len(traded)} events already traded.")

    import csv
    while True:
        try:
            eth_tail = prep(read_tail("eth", ETH_TAIL))
            btc_s = xa.build_book_series("btc", df=read_tail("btc", BTC_TAIL))
            eth_s = xa.build_book_series("eth", df=eth_tail)
            d = assemble(eth_tail, build_liq(), btc_s, eth_s)
            recent = d[d["dt"] >= d["dt"].max() - pd.Timedelta(minutes=6)].copy()
            if len(recent):
                lo = models[Q_LO].predict(recent[FEATS])
                hi = models[Q_HI].predict(recent[FEATS])
                recent["pred_width"] = (hi - lo) * np.sqrt(recent["tau_h"])
                recent["event"] = recent["contract_ticker"].astype(str).str.rsplit(
                    "-T", n=1).str[0]
                ev = recent.groupby("event").agg(
                    pred=("pred_width", "median"),
                    impw=("imp_width_pct", "median")).dropna()
                ev["vratio"] = ev["pred"] / ev["impw"]
                if len(ev):
                    _vmax = float(ev["vratio"].max())
                    print(f"  [hb] events={len(ev)} vratio_max={_vmax:.2f} "
                          f"impw_cov={recent['imp_width_pct'].notna().mean():.0%}")
                fires = ev[(ev["vratio"] >= VRATIO_MIN) & ~ev.index.isin(traded)]
                if len(fires):
                    for evname, row in fires.iterrows():
                        # [2026-08-05] per-event fetch + DOLLAR quote fields
                        # (yes_ask_dollars/yes_bid_dollars — the old yes_ask/
                        # yes_bid keys don't exist in /markets responses and
                        # silently defaulted to 0: no leg ever qualified).
                        mk = kalshi_get("/markets", {"event_ticker": evname,
                                                     "limit": 200}, auth)
                        legs = []
                        n_mk = 0
                        for m in mk.get("markets", []):
                            t = m["ticker"]; q = m; n_mk += 1
                            try:
                                ask = scale_price(q.get("yes_ask_dollars")
                                                  or q.get("yes_ask") or 0)
                                bid = scale_price(q.get("yes_bid_dollars")
                                                  or q.get("yes_bid") or 0)
                            except (TypeError, ValueError):
                                continue
                            if 0.03 <= ask <= 0.15:
                                legs.append(("yes", ask, (ask + bid) / 2, t, q))
                            if 0.85 <= bid <= 0.97 and (1 - bid) >= 0.03:
                                legs.append(("no", 1 - bid, 1 - (ask + bid) / 2, t, q))
                        if not legs:
                            print(f"  [no-legs] {evname} vratio={row['vratio']:.2f} "
                                  f"markets={n_mk} — no qualifying tail rung")
                            continue
                        best = {}
                        for side, cost, mid, t, q in legs:
                            if side not in best or cost < best[side][1]:
                                best[side] = (side, cost, mid, t, q)
                        with open(BOOK, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=BOOK_COLS,
                                               extrasaction="ignore")
                            for side, cost, mid, t, q in best.values():
                                w.writerow({
                                    "logged_at": datetime.now(timezone.utc).isoformat(),
                                    "event": evname, "side": side,
                                    "contract_ticker": t,
                                    "close_ts": q.get("close_time", ""),
                                    "spot": float(recent["spot"].iloc[-1]),
                                    "strike": q.get("floor_strike", ""),
                                    "vratio": round(float(row["vratio"]), 3),
                                    "pred_width": round(float(row["pred"]), 4),
                                    "imp_width": round(float(row["impw"]), 4),
                                    "cost_ask": round(cost, 4),
                                    "cost_mid": round(mid, 4),
                                    "stake": STAKE,
                                })
                        traded.add(evname)
                        print(f"  [TAIL] {evname} vratio={row['vratio']:.2f} "
                              f"legs={[(s, round(c,2)) for s, c, *_ in best.values()]}")
                    st["traded"] = sorted(traded)[-500:]
                    STATE.write_text(json.dumps(st))
            bk = pd.read_csv(BOOK, low_memory=False)
            pend = bk["resolved_yes"].isna() if len(bk) else pd.Series(dtype=bool)
            if len(bk) and pend.any():
                res = (d.dropna(subset=["resolved_yes"])
                       .drop_duplicates("contract_ticker", keep="last")
                       .set_index("contract_ticker")["resolved_yes"].to_dict())
                _missing = {bk.at[_i, "contract_ticker"]
                            for _i in bk[pend].index
                            if bk.at[_i, "contract_ticker"] not in res}
                if _missing:  # [2026-08-25 stale-pending fallback — see fav runners]
                    try:
                        for _fch in pd.read_csv(BASE / "results" / "eth_scan_archive.csv",
                                                usecols=["contract_ticker",
                                                         "resolved_yes"],
                                                chunksize=500_000,
                                                low_memory=False,
                                                on_bad_lines="skip"):
                            _fhit = _fch[
                                _fch["contract_ticker"].isin(_missing)
                                & _fch["resolved_yes"].notna()]
                            for _ft, _frv in zip(_fhit["contract_ticker"],
                                                 _fhit["resolved_yes"]):
                                res[_ft] = _frv
                    except Exception:
                        pass
                ch = 0
                for i in bk[pend].index:
                    rv = res.get(bk.at[i, "contract_ticker"])
                    if rv is None or (isinstance(rv, float) and rv != rv):
                        continue
                    rv = int(float(rv))
                    cost = float(bk.at[i, "cost_ask"])
                    stk = float(bk.at[i, "stake"])
                    yes_side = bk.at[i, "side"] == "yes"
                    win = (rv == 1) == yes_side
                    gross = round(stk * (1 - cost) / cost, 2) if win else -stk
                    pm_eff = cost if yes_side else 1 - cost
                    feev = round((stk / cost) * 0.07 * pm_eff * (1 - pm_eff), 2)
                    bk.loc[i, ["resolved_yes", "would_win", "would_pnl",
                               "fee_est", "would_pnl_net"]] = \
                        [rv, int(win), gross, feev, round(gross - feev, 2)]
                    ch += 1
                if ch:
                    tmp = BOOK.with_suffix(".csv.tmp")
                    bk.to_csv(tmp, index=False)
                    os.replace(tmp, BOOK)
                    net = pd.to_numeric(bk["would_pnl_net"], errors="coerce").sum()
                    print(f"  [resolve] {ch} legs; book net: {net:+.2f}")
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
