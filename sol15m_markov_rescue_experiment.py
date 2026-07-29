"""SOL 15m markov-rescue EXPERIMENT book — 2026-07-29.

PRE-REGISTERED forward test of the stabilization rescue that failed the
backtest bar (honest OOS permP=0.1252 on mined data; June-negative when
composed with band gates — see project_hourly_phase2_eth_sol_nulls_20260728
"Regime×slope rescue search"). Rule FROZEN at deploy time; whatever the next
weeks show is a true forward test, immune to the multiple-testing concern.

Trades (flat $100 NO, one bet per contract, paper only, own book):
  population: scans the main SOL runner evaluated where the would-be side is
              NO (p_model < pm, |edge| >= 0.04) AND sol_markov_gate blocked it
              (reconstructed: [6h=Bull & offset>-0.006] or [4h=Sideways &
              stoch_k_1h<90], minus its own rescues)
  rescue:     dprice_120 >= 0  (2h price stabilized/risen — fade the bounce;
              knife-still-falling stays untraded)
  band filters (same as deployed package): NOT pm>0.80;
              NOT (pm in [0.50,0.65] unless slope120_stoch_k_15m>=40)

Reads the main SOL book's own scan tail (features logged by the live runner —
byte-identical pipeline, no lookahead). Main runner untouched: its markov
discipline continues; this book exists purely to settle the 08-11 three-way
markov decision with forward evidence.
Decision rule at review: net-positive -> promote rescue into main runner;
negative -> markov gate keeps its job, question closed.
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
SRC = BASE / "results" / "paper_trades_sol15m.csv"
BOOK = BASE / "results" / "paper_trades_sol15m_markov_rescue_exp.csv"
STATE = BASE / "results" / ".sol15m_markov_rescue_exp_state.json"

STAKE = 100.0
TAIL_BYTES = 1_500_000
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "contract_ticker", "close_time", "spot", "floor_strike",
             "p_market", "p_model_15m", "raw_edge", "dprice_120",
             "slope120_stoch_k_15m", "markov_sol_6h", "markov_sol_4h",
             "stoch_k_1h", "offset_pct", "tau_minutes", "stake",
             "resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net"]


def read_tail() -> pd.DataFrame:
    with open(SRC, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - TAIL_BYTES))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    df = pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    return df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)


def main() -> None:
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"last_ts": datetime.now(timezone.utc).isoformat(), "traded": []}
    # NOTE: last_ts initialized to NOW (not epoch): forward-only by design —
    # no backfill, this is a pre-registered forward test.
    traded = set(st["traded"])
    print(f"[markov-rescue-exp] up. NO-only, flat ${STAKE:.0f}; forward-only from {st['last_ts']}.")
    import csv
    while True:
        try:
            df = read_tail()
            for c in ["p_market", "p_model_15m", "resolved_yes", "stoch_k_1h", "offset_pct",
                      "dprice_120", "slope120_stoch_k_15m", "spot", "floor_strike", "tau_minutes"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            new = df[df["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                n = new
                side_no = n["p_model_15m"] < n["p_market"]
                edge_ok = (n["p_model_15m"] - n["p_market"]).abs() >= 0.04
                m6, m4 = n["markov_sol_6h"].astype(str), n["markov_sol_4h"].astype(str)
                sk = n["stoch_k_1h"].fillna(50.0)
                off = n["offset_pct"]
                gate_no = ((m6 == "Bull") & (off > -0.006)) | ((m4 == "Sideways") & (sk < 90))
                resc_no = ((m6 == "Bull") & (off <= -0.006)) | ((m4 == "Sideways") & (sk >= 90))
                blocked = side_no & edge_ok & gate_no & ~resc_no
                stab = (n["dprice_120"] >= 0.0).fillna(False)
                band_ok = ~(n["p_market"] > 0.80) & \
                    (~n["p_market"].between(0.50, 0.65) | (n["slope120_stoch_k_15m"] >= 40).fillna(False))
                hits = n[blocked & stab & band_ok & ~n["contract_ticker"].isin(traded)]
                hits = hits.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
                for _, r in hits.iterrows():
                    with open(BOOK, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=BOOK_COLS, extrasaction="ignore").writerow({
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "contract_ticker": r["contract_ticker"],
                            "close_time": r.get("close_time", ""),
                            "spot": r["spot"], "floor_strike": r.get("floor_strike", ""),
                            "p_market": round(float(r["p_market"]), 4),
                            "p_model_15m": round(float(r["p_model_15m"]), 4),
                            "raw_edge": round(float(r["p_model_15m"] - r["p_market"]), 4),
                            "dprice_120": round(float(r["dprice_120"]), 4),
                            "slope120_stoch_k_15m": r.get("slope120_stoch_k_15m", ""),
                            "markov_sol_6h": r["markov_sol_6h"], "markov_sol_4h": r["markov_sol_4h"],
                            "stoch_k_1h": r.get("stoch_k_1h", ""), "offset_pct": r.get("offset_pct", ""),
                            "tau_minutes": r.get("tau_minutes", ""), "stake": STAKE,
                        })
                    traded.add(r["contract_ticker"])
                    print(f"  [TRADE] NO {r['contract_ticker']} pm={r['p_market']:.3f} "
                          f"dprice120={r['dprice_120']:+.3f} ({r['markov_sol_6h']}/{r['markov_sol_4h']})")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-2000:]
                STATE.write_text(json.dumps(st))
            # resolve from the source book's backfill
            bk = pd.read_csv(BOOK, low_memory=False)
            if len(bk):
                pend = bk["resolved_yes"].isna()
                if pend.any():
                    res = (df.dropna(subset=["resolved_yes"])
                           .drop_duplicates("contract_ticker", keep="last")
                           .set_index("contract_ticker")["resolved_yes"].to_dict())
                    ch = 0
                    for i in bk[pend].index:
                        rv = res.get(bk.at[i, "contract_ticker"])
                        if rv is None or (isinstance(rv, float) and rv != rv):
                            continue
                        rv = int(float(rv)); pm = float(bk.at[i, "p_market"]); stk = float(bk.at[i, "stake"])
                        win = rv == 0  # NO-only book
                        cost = 1 - pm
                        gross = round(stk * pm / cost, 2) if win else -stk
                        feev = round((stk / cost) * 0.07 * pm * (1 - pm), 2)
                        bk.loc[i, ["resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net"]] = \
                            [rv, int(win), gross, feev, round(gross - feev, 2)]
                        ch += 1
                    if ch:
                        tmp = BOOK.with_suffix(".csv.tmp")
                        bk.to_csv(tmp, index=False)
                        os.replace(tmp, BOOK)
                        print(f"  [resolve] {ch} filled; exp book net: "
                              f"{pd.to_numeric(bk['would_pnl_net'], errors='coerce').sum():+.2f}")
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
