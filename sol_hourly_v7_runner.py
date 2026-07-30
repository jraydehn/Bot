"""SOL hourly v7 quantile-model paper runner — forward A/B vs production.
2026-07-30.

Runs the frozen sol_hourly_quantile_v7_20260730.pkl on live hourly scans as
a standalone flat-$100 paper book (results/paper_trades_sol_hourly_v7.csv),
alongside — never touching — the production SOL hourly paper runner. This
is the fresh-forward-data test: the 07-09..07-30 window is burned (6 prior
evaluations) and is never scored.

PRE-REGISTERED (from val 06-25..07-09, chosen before any forward data):
  PRIMARY book: YES side, pm ∈ [0.20, 0.80], fee-adj edge >= 0.05.
  SECONDARY (logged, tagged 'no'): NO side, same band/threshold — val was
  mixed on NO; logged for the review, not part of the primary verdict.
  Review: compare v7 primary book vs production paper book on overlapping
  scans, net of fees, ~2026-08-13 first read, late-Aug decision.

Loop: read archive tail (10 days — covers 480-min slopes, 24h recent-YES,
warmup), rebuild the full frozen feature stack via the same module
functions used in training, predict quantiles, derive p, book new
qualifying contracts, resolve pending from archive backfill.
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

import train_sol_hourly_niche_v3 as v3
import train_sol_hourly_v4_15mborrow as v4
from train_sol_hourly_v5_4h import add_4h_context, add_m15_4h
from kalshi_microstructure_features import build_micro_features
import sol_hourly_banked_signals as bank
from train_sol_hourly_v7_quantile import derive_p, QUANTILES

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "sol_scan_archive.csv"
BOOK = BASE / "results" / "paper_trades_sol_hourly_v7.csv"
STATE = BASE / "results" / ".sol_hourly_v7_state.json"
MODEL_P = BASE / "models" / "sol_hourly_quantile_v7_20260730.pkl"

PM_LO, PM_HI = 0.20, 0.80
EDGE_MIN = 0.05
STAKE = 100.0
TAIL_BYTES = 18_000_000  # ~10 days of archive
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "side", "contract_ticker", "close_ts", "spot",
             "strike", "p_market", "p_model", "fee_adj_edge", "tau_minutes",
             "stake", "resolved_yes", "would_win", "would_pnl", "fee_est",
             "would_pnl_net"]


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
    for c in set(v3.STATIC + v3.SLOPE_BASES + ["spot", "strike", "resolved_yes"]) - {"z_moneyness"}:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_ts"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    return df


def build_features(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df[df["p_market"].notna() & df["p_market"].between(0.02, 0.98)]
    df = df.sort_values("dt").reset_index(drop=True)
    df, _ = add_4h_context(df)
    m15 = v4.build_m15_stream()
    m15, _ = add_m15_4h(m15)
    m15c = [c for c in m15.columns if c != "dt"]
    df = pd.merge_asof(df, m15.rename(columns={c: f"m15_{c}" for c in m15c}),
                       on="dt", direction="backward",
                       tolerance=pd.Timedelta(minutes=45))
    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)
    liq = bank.build_liq_features(bank.fetch_liq_bars_live())
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    df, _ = bank.add_clock_and_daily(df)
    for c in feats:
        if c not in df.columns:
            df[c] = np.nan
    df["tau_h"] = pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1) / 60
    df["needed_scaled"] = (df["strike"] / df["spot"] - 1) * 100 / np.sqrt(df["tau_h"])
    return df


def main() -> None:
    with open(MODEL_P, "rb") as f:
        art = pickle.load(f)
    models, feats = art["models"], art["features"]
    if not BOOK.exists():
        BOOK.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"last_ts": "2026-07-30T00:00:00+00:00", "traded": []}
    traded = set(st["traded"])
    print(f"[sol-hourly-v7] up. PRIMARY yes pm[{PM_LO},{PM_HI}] edge>={EDGE_MIN} "
          f"flat ${STAKE:.0f}; NO logged secondary. {len(feats)} feats, "
          f"{len(QUANTILES)} quantile models. {len(traded)} already traded.")

    import csv
    while True:
        try:
            df = build_features(read_tail(), feats)
            new = df[df["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                cand = new[new["p_market"].between(PM_LO, PM_HI)].copy()
                cand = cand[~cand["contract_ticker"].isin(traded)]
                if len(cand):
                    qp = np.column_stack([models[q].predict(cand[feats])
                                          for q in QUANTILES])
                    p = derive_p(qp, cand["needed_scaled"].values)
                    fee = 0.07 * cand["p_market"] * (1 - cand["p_market"])
                    for side in ["yes", "no"]:
                        edge = (p - cand["p_market"] - fee) if side == "yes" \
                            else (cand["p_market"] - p - fee)
                        hits = cand[np.asarray(edge) >= EDGE_MIN].copy()
                        hits["p_model"] = p[np.asarray(edge) >= EDGE_MIN]
                        hits["edge_v"] = np.asarray(edge)[np.asarray(edge) >= EDGE_MIN]
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
            # resolve pending from archive backfill
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
