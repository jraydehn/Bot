"""SOL hourly v8 paper runner — third arm of the forward A/B. 2026-07-30.

Runs the frozen compact survivor-core quantile model (sol_hourly_v8_20260730
.pkl) as a standalone flat-$100 book (paper_trades_sol_hourly_v8.csv)
alongside production and the v7 runner. Identical pre-registered rule as v7
for comparability: PRIMARY YES pm∈[0.20,0.80] fee-adj edge>=0.05; NO side
logged secondary. Review together with v7 at the late-Aug read.
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
from train_sol_hourly_v7_quantile import derive_p, QUANTILES
from train_sol_hourly_v8 import assemble, FEATS

BASE = Path(__file__).parent
BOOK = BASE / "results" / "paper_trades_sol_hourly_v8.csv"
STATE = BASE / "results" / ".sol_hourly_v8_state.json"
MODEL_P = BASE / "models" / "sol_hourly_v8_20260730.pkl"

PM_LO, PM_HI = 0.20, 0.80
EDGE_MIN = 0.05
STAKE = 100.0
SOL_TAIL = 18_000_000     # ~4 days (24h recent-YES + slopes warmup)
NBR_TAIL = 6_000_000      # BTC/ETH: only need ~2h for book chg15
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "side", "contract_ticker", "close_ts", "spot",
             "strike", "p_market", "p_model", "fee_adj_edge", "tau_minutes",
             "stake", "resolved_yes", "would_win", "would_pnl", "fee_est",
             "would_pnl_net"]


def read_tail(asset: str, nbytes: int) -> pd.DataFrame:
    path = BASE / "results" / f"{asset}_scan_archive.csv"
    with open(path, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - nbytes))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    df = pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")
    return df


def prep_sol(df: pd.DataFrame) -> pd.DataFrame:
    import train_sol_hourly_niche_v3 as v3
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    for c in set(v3.STATIC + v3.SLOPE_BASES + ["spot", "strike", "resolved_yes"]) - {"z_moneyness"}:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_ts"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    return df


def main() -> None:
    with open(MODEL_P, "rb") as f:
        art = pickle.load(f)
    models = art["models"]
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"last_ts": datetime.now(timezone.utc).isoformat(), "traded": []}
    traded = set(st["traded"])
    print(f"[sol-hourly-v8] up. PRIMARY yes pm[{PM_LO},{PM_HI}] edge>={EDGE_MIN} "
          f"flat ${STAKE:.0f}; {len(FEATS)} survivor-core feats. "
          f"{len(traded)} already traded.")

    import csv
    while True:
        try:
            sol_tail = prep_sol(read_tail("sol", SOL_TAIL))
            btc_s = xa.build_book_series("btc", df=read_tail("btc", NBR_TAIL))
            eth_s = xa.build_book_series("eth", df=read_tail("eth", NBR_TAIL))
            sol_s = xa.build_book_series("sol", df=sol_tail)
            df = assemble(sol_tail, btc_s, eth_s, sol_s,
                          bank.fetch_liq_bars_live())
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
                                })
                            traded.add(r["contract_ticker"])
                            print(f"  [TRADE:{side}] {r['contract_ticker']} "
                                  f"pm={r['p_market']:.3f} p={r['p_model']:.3f} "
                                  f"edge={r['edge_v']:.3f} tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-3000:]
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
