"""BTC hourly VOL-TAIL paper runner — direction-neutral straddle book.
2026-08-04.

The replacement direction for the retired bookdyn approach: trades the ONE
axis the campaign proved learnable (vol) against the price the ladder
quotes for it (implied width). Signal: vratio = bookdyn-tail predicted
width (q85−q15, the dimension those models actually learned) / ladder
imp_width_pct. Backtest (hourly_voltail_backtest.py): screen <07-09
+$130k-at-mid (inflated by mid-fills on wide tail spreads — DISCLOSED),
frozen test +$7,212 vs controls −$28k/−$35k — the signal RANKS regimes
correctly in both windows; absolute edge is what THIS runner measures
honestly.

PRE-REGISTERED RULE: vratio >= 1.2 → ONE straddle per event-hour:
  YES leg: cheapest tail rung with ASK in [0.03, 0.15]
  NO leg:  rung with yes_bid in [0.85, 0.97] (NO-ask = 1−bid in [.03,.15])
  $100/leg, fills at the ASK (honest taker), fees on top, hold to settle.
  Both legs required, else skip. Mid-price also logged per leg so forward
  slippage (ask vs mid) is measurable — the exact quantity the backtest
  couldn't see.
ETH: tested, does NOT clear the bar (sparse ladder, ~5 straddles/month) —
no ETH runner, by design not omission.
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
from live_signal import load_auth, kalshi_get
from train_btc_hourly_bookdyn import assemble, FEATS

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "btc_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_btc_hourly_voltail.csv"
STATE = BASE / "results" / ".btc_hourly_voltail_state.json"
MODEL_P = BASE / "models" / "btc_hourly_bookdyn_20260731.pkl"

VRATIO_MIN = 1.2
STAKE = 100.0
TAIL_BYTES = 18_000_000
LOOP_SEC = 120
Q_LO, Q_HI = 0.15, 0.85

BOOK_COLS = ["logged_at", "event", "side", "contract_ticker", "close_ts",
             "spot", "strike", "vratio", "pred_width", "imp_width",
             "cost_ask", "cost_mid", "stake", "resolved_yes", "would_win",
             "would_pnl", "fee_est", "would_pnl_net"]


def read_tail() -> pd.DataFrame:
    with open(ARCHIVE, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - TAIL_BYTES))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    df = pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")
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


def scale_price(v):
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def main() -> None:
    auth = load_auth()
    with open(MODEL_P, "rb") as f:
        art = pickle.load(f)
    models = art["models"]
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else {"traded": []}
    traded = set(st["traded"])
    print(f"[btc-voltail] up. vratio>={VRATIO_MIN}, ${STAKE:.0f}/leg at ASK, "
          f"one straddle/event. {len(traded)} events already traded.")

    import csv
    while True:
        try:
            liq = bank.fetch_liq_bars_live(asset="BTC")
            tot = liq["liq_long"] + liq["liq_short"]
            lf = liq.copy()
            lf["liq_total_z"] = tot.rolling(168).rank(pct=True)
            lf["liq_long_z"] = lf["liq_long"].rolling(168).rank(pct=True)
            lf["liq_short_z"] = lf["liq_short"].rolling(168).rank(pct=True)
            lf["liq_imbalance"] = (lf["liq_long"] - lf["liq_short"]) / tot.replace(0, np.nan)
            for w in (6, 12):
                lf[f"liq_imbalance_trend{w}"] = (lf["liq_imbalance"]
                                                 - lf["liq_imbalance"].shift(w)) / w
            lf = lf.drop(columns=["liq_long", "liq_short"])
            d = assemble(read_tail(), lf)
            recent = d[d["dt"] >= d["dt"].max() - pd.Timedelta(minutes=6)].copy()
            if len(recent):
                lo = models[Q_LO].predict(recent[FEATS])
                hi = models[Q_HI].predict(recent[FEATS])
                recent["pred_width"] = (hi - lo) * np.sqrt(recent["tau_h"])
                recent["event"] = recent["contract_ticker"].astype(str).str.rsplit(
                    "-T", n=1).str[0]
                ev_stats = recent.groupby("event").agg(
                    pred=("pred_width", "median"),
                    impw=("imp_width_pct", "median")).dropna()
                ev_stats["vratio"] = ev_stats["pred"] / ev_stats["impw"]
                fires = ev_stats[(ev_stats["vratio"] >= VRATIO_MIN)
                                 & ~ev_stats.index.isin(traded)]
                if len(fires):
                    mk = kalshi_get("/markets", {"series_ticker": "KXBTCD",
                                                 "status": "open", "limit": 200}, auth)
                    quotes = {m["ticker"]: m for m in mk.get("markets", [])}
                    for ev, row in fires.iterrows():
                        evq = [(t, q) for t, q in quotes.items() if t.startswith(ev + "-")]
                        ylegs, nlegs = [], []
                        for t, q in evq:
                            try:
                                ask = scale_price(q.get("yes_ask", 0))
                                bid = scale_price(q.get("yes_bid", 0))
                            except (TypeError, ValueError):
                                continue
                            if 0.03 <= ask <= 0.15:
                                ylegs.append((ask, bid, t, q))
                            if 0.85 <= bid <= 0.97 and (1 - bid) >= 0.03:
                                nlegs.append((ask, bid, t, q))
                        if not ylegs or not nlegs:
                            continue
                        ylegs.sort(key=lambda x: x[0])          # cheapest YES ask
                        nlegs.sort(key=lambda x: -x[1])         # highest bid → cheapest NO
                        ya, yb, yt, yq = ylegs[0]
                        na, nb, nt, nq = nlegs[0]
                        rows = [
                            {"side": "yes", "contract_ticker": yt,
                             "cost_ask": round(ya, 4), "cost_mid": round((ya + yb) / 2, 4),
                             "strike": yq.get("floor_strike", ""),
                             "close_ts": yq.get("close_time", "")},
                            {"side": "no", "contract_ticker": nt,
                             "cost_ask": round(1 - nb, 4),
                             "cost_mid": round(1 - (na + nb) / 2, 4),
                             "strike": nq.get("floor_strike", ""),
                             "close_ts": nq.get("close_time", "")},
                        ]
                        with open(BOOK, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=BOOK_COLS,
                                               extrasaction="ignore")
                            for r in rows:
                                r.update({"logged_at": datetime.now(timezone.utc).isoformat(),
                                          "event": ev, "spot": float(recent["spot"].iloc[-1]),
                                          "vratio": round(float(row["vratio"]), 3),
                                          "pred_width": round(float(row["pred"]), 4),
                                          "imp_width": round(float(row["impw"]), 4),
                                          "stake": STAKE})
                                w.writerow(r)
                        traded.add(ev)
                        print(f"  [STRADDLE] {ev} vratio={row['vratio']:.2f} "
                              f"YES {yt}@{ya:.2f} + NO {nt}@{1-nb:.2f}")
                    st["traded"] = sorted(traded)[-500:]
                    STATE.write_text(json.dumps(st))
            # resolve from archive backfill
            bk = pd.read_csv(BOOK, low_memory=False)
            pend = bk["resolved_yes"].isna() if len(bk) else pd.Series(dtype=bool)
            if len(bk) and pend.any():
                res = (d.dropna(subset=["resolved_yes"])
                       .drop_duplicates("contract_ticker", keep="last")
                       .set_index("contract_ticker")["resolved_yes"].to_dict())
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
