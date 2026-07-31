"""ETH hourly book-dynamics challenger paper runner. 2026-07-31.

Frozen eth_hourly_bookdyn_20260731.pkl on live ETH hourly scans as a
standalone flat-$100 book (paper_trades_eth_hourly_bookdyn.csv), alongside
— never touching — the production ETH hourly runner and the niche v2
runner. Family protocol: PRIMARY YES pm∈[0.20,0.80] fee-adj edge>=0.05,
NO secondary, ctx_gates tagged (BTC/ETH variant, rr_exception=0.08).
First read ~08-14 with the SOL/BTC challenger books.
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
from sol_hourly_ctx_gates import ctx_gate_fails_btc
from train_sol_hourly_v7_quantile import derive_p, QUANTILES
from train_eth_hourly_bookdyn import assemble, FEATS

BASE = Path(__file__).parent
BOOK = BASE / "results" / "paper_trades_eth_hourly_bookdyn.csv"
STATE = BASE / "results" / ".eth_hourly_bookdyn_state.json"
MODEL_P = BASE / "models" / "eth_hourly_bookdyn_20260731.pkl"

PM_LO, PM_HI = 0.20, 0.80
EDGE_MIN = 0.05
STAKE = 100.0
ETH_TAIL = 14_000_000
BTC_TAIL = 6_000_000
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "side", "contract_ticker", "close_ts", "spot",
             "strike", "p_market", "p_model", "fee_adj_edge", "tau_minutes",
             "stake", "resolved_yes", "would_win", "would_pnl", "fee_est",
             "would_pnl_net", "ctx_gates"]


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


def build_liq_feats(liq: pd.DataFrame) -> pd.DataFrame:
    liq = liq.copy()
    tot = liq["liq_long"] + liq["liq_short"]
    liq["liq_total_z"] = tot.rolling(168).rank(pct=True)
    liq["liq_long_z"] = liq["liq_long"].rolling(168).rank(pct=True)
    liq["liq_short_z"] = liq["liq_short"].rolling(168).rank(pct=True)
    liq["liq_imbalance"] = (liq["liq_long"] - liq["liq_short"]) / tot.replace(0, np.nan)
    liq["liq_imbalance_trend6"] = (liq["liq_imbalance"] - liq["liq_imbalance"].shift(6)) / 6
    return liq.drop(columns=["liq_long", "liq_short"])


def main() -> None:
    with open(MODEL_P, "rb") as f:
        art = pickle.load(f)
    models = art["models"]
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"last_ts": datetime.now(timezone.utc).isoformat(), "traded": []}
    traded = set(st["traded"])
    print(f"[eth-hourly-bookdyn] up. PRIMARY yes pm[{PM_LO},{PM_HI}] "
          f"edge>={EDGE_MIN} flat ${STAKE:.0f}; {len(FEATS)} feats. "
          f"{len(traded)} already traded.")

    import csv
    while True:
        try:
            eth_tail = prep(read_tail("eth", ETH_TAIL))
            btc_s = xa.build_book_series("btc", df=read_tail("btc", BTC_TAIL))
            eth_s = xa.build_book_series("eth", df=eth_tail)
            liq = build_liq_feats(bank.fetch_liq_bars_live(asset="ETH"))
            df = assemble(eth_tail, liq, btc_s, eth_s)
            new = df[df["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                cand = new[new["p_market"].between(PM_LO, PM_HI)].copy()
                cand = cand[~cand["contract_ticker"].isin(traded)]
                if len(cand):
                    qp = np.column_stack([models[q].predict(cand[FEATS])
                                          for q in QUANTILES])
                    p = derive_p(qp, cand["needed_scaled"].values)
                    fee = 0.07 * cand["p_market"] * (1 - cand["p_market"])
                    for side in ["yes", "no"]:
                        edge = np.asarray((p - cand["p_market"] - fee) if side == "yes"
                                          else (cand["p_market"] - p - fee))
                        hits = cand[edge >= EDGE_MIN].copy()
                        hits["p_model"] = p[edge >= EDGE_MIN]
                        hits["edge_v"] = edge[edge >= EDGE_MIN]
                        hits = hits.sort_values("dt").drop_duplicates(
                            "contract_ticker", keep="first")
                        for _, r in hits.iterrows():
                            if r["contract_ticker"] in traded:
                                continue
                            with open(BOOK, "a", newline="") as f:
                                csv.DictWriter(f, fieldnames=BOOK_COLS,
                                               extrasaction="ignore").writerow({
                                    "logged_at": datetime.now(timezone.utc).isoformat(),
                                    "side": side,
                                    "contract_ticker": r["contract_ticker"],
                                    "close_ts": r.get("close_ts", ""),
                                    "spot": r["spot"], "strike": r["strike"],
                                    "p_market": round(float(r["p_market"]), 4),
                                    "p_model": round(float(r["p_model"]), 4),
                                    "fee_adj_edge": round(float(r["edge_v"]), 4),
                                    "tau_minutes": r["tau_minutes"],
                                    "stake": STAKE,
                                    "ctx_gates": ctx_gate_fails_btc(
                                        side, float(r["p_market"]),
                                        r.get("offset_pct", float("nan")),
                                        r.get("composite_p_up", float("nan")),
                                        float(r["edge_v"]), rr_exception=0.08),
                                })
                            traded.add(r["contract_ticker"])
                            print(f"  [TRADE:{side}] {r['contract_ticker']} "
                                  f"pm={r['p_market']:.3f} p={r['p_model']:.3f} "
                                  f"edge={r['edge_v']:.3f} tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-4000:]
                STATE.write_text(json.dumps(st))
            bk = pd.read_csv(BOOK, low_memory=False)
            pend = bk["resolved_yes"].isna() if len(bk) else pd.Series(dtype=bool)
            if len(bk) and pend.any():
                res = (df.dropna(subset=["resolved_yes"])
                       .drop_duplicates("contract_ticker", keep="last")
                       .set_index("contract_ticker")["resolved_yes"].to_dict())
                ch = 0
                for i in bk[pend].index:
                    rv = res.get(bk.at[i, "contract_ticker"])
                    if rv is None or (isinstance(rv, float) and rv != rv):
                        continue
                    rv = int(float(rv))
                    pm = float(bk.at[i, "p_market"])
                    stk = float(bk.at[i, "stake"])
                    yes_side = bk.at[i, "side"] == "yes"
                    cost = pm if yes_side else 1 - pm
                    win = (rv == 1) == yes_side
                    gross = round(stk * (1 - cost) / cost, 2) if win else -stk
                    feev = round((stk / cost) * 0.07 * pm * (1 - pm), 2)
                    bk.loc[i, ["resolved_yes", "would_win", "would_pnl",
                               "fee_est", "would_pnl_net"]] = \
                        [rv, int(win), gross, feev, round(gross - feev, 2)]
                    ch += 1
                if ch:
                    tmp = BOOK.with_suffix(".csv.tmp")
                    bk.to_csv(tmp, index=False)
                    os.replace(tmp, BOOK)
                    net = pd.to_numeric(bk["would_pnl_net"], errors="coerce").sum()
                    print(f"  [resolve] {ch} filled; book net: {net:+.2f}")
        except Exception as e:
            print(f"  [error] loop failed (continuing): {e}")
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
