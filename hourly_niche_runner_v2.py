"""Hourly YES-niche paper runner v2 (slope features) — 2026-07-28.

Generalized version of btc_hourly_niche_runner.py: adds the d/S slope features
(15/45/120-min deltas and signal-vs-price slopes of SCAN-LEVEL signals,
computed from the scan archive's own history — contract-level columns are
never slope bases, see the contract-mixing lesson) and takes --asset.

v2 validation (frozen train <06-20, July holdout, flat $100, net of fees,
one bet per contract; rule YES + pm∈[0.35,0.65] + fee-adj edge>=0.06):
  BTC: +$6,367 / 357 trades, WR 56.6% vs BE 47.3%, ALL 5 July weeks green
       (static-only v1 same protocol: +$4,762 — slopes add +34% and +87% volume)
  ETH: +$1,379 / 430 trades, WR 54.9% vs BE 51.6%, 4/5 weeks green (only the
       1.5-day partial wk31 red, −$303) — BORDERLINE, deployed as paper trial
       at explicitly lower confidence.
  SOL: NULL (−$1,167) — not deployed.
Walk-forward weekly retraining tested and REJECTED (+$1,513, 3 red weeks —
chasing the recent tape hurts; model stays FROZEN, retrain only deliberately).

Runs alongside the v1 BTC runner as a forward A/B (separate books).
"""
import argparse
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

EDGE_MIN = 0.06
PM_LO, PM_HI = 0.35, 0.65
STAKE = 100.0
TAIL_BYTES = 3_000_000  # must cover >120min of archive history for slopes
LOOP_SEC = 120

BOOK_COLS = ["logged_at", "contract_ticker", "close_ts", "spot", "strike",
             "p_market", "p_model", "fee_adj_edge", "tau_minutes", "stake",
             "resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net"]


def read_archive_tail(archive: Path) -> pd.DataFrame:
    with open(archive, "rb") as f:
        header = f.readline().decode()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(len(header.encode()), size - TAIL_BYTES))
        chunk = f.read().decode(errors="replace")
    body = chunk[chunk.index("\n") + 1:] if "\n" in chunk else ""
    df = pd.read_csv(StringIO(header + body), low_memory=False, on_bad_lines="skip")
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                              errors="coerce", utc=True, format="mixed")
    return df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)


def prep(df: pd.DataFrame, feats: list, slope_bases: list) -> pd.DataFrame:
    static_needed = (set(feats) - {"z_moneyness"}
                     - {c for c in feats if c.startswith(("D15_", "D45_", "D120_", "S15_", "S45_", "S120_", "dprice_"))})
    for c in set(list(static_needed) + slope_bases + ["p_market", "tau_minutes", "strike", "spot", "resolved_yes"]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    ts = df["dt"].astype("int64") / 1e9
    nc = {}
    for tag, sec in [("15", 900), ("45", 2700), ("120", 7200)]:
        idx = np.searchsorted(ts, ts - sec, side="right") - 1
        valid = idx >= 0
        pv = np.where(valid, df["spot"].values[np.clip(idx, 0, None)], np.nan)
        dp = pd.Series((df["spot"].values / pv - 1) * 100, index=df.index)
        nc[f"dprice_{tag}"] = dp
        for c in slope_bases:
            pr = np.where(valid, df[c].values[np.clip(idx, 0, None)], np.nan)
            d = df[c].values - pr
            nc[f"D{tag}_{c}"] = d
            nc[f"S{tag}_{c}"] = np.clip(d / dp.replace(0, np.nan), -50, 50)
    df = pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(df["tau_minutes"].clip(lower=1))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["BTC", "ETH"])
    args = ap.parse_args()
    a = args.asset.lower()
    archive = BASE / "results" / f"{a}_scan_archive.csv"
    book = BASE / "results" / f"paper_trades_{a}_hourly_niche_v2.csv"
    state_p = BASE / "results" / f".{a}_hourly_niche_v2_state.json"
    model_p = BASE / "models" / f"{a}_hourly_lgbm_niche_v2_20260728.pkl"

    with open(model_p, "rb") as f:
        art = pickle.load(f)
    model, feats, slope_bases = art["model"], art["features"], art["slope_bases"]

    if not book.exists():
        book.write_text(",".join(BOOK_COLS) + "\n")
    st = json.loads(state_p.read_text()) if state_p.exists() else \
        {"last_ts": "1970-01-01T00:00:00+00:00", "traded": []}
    traded = set(st["traded"])
    print(f"[niche-v2:{args.asset}] up. YES, pm∈[{PM_LO},{PM_HI}], edge>={EDGE_MIN}, "
          f"flat ${STAKE:.0f}, {len(feats)} feats incl slopes. {len(traded)} tickers already traded.")

    import csv
    while True:
        try:
            arch = prep(read_archive_tail(archive), feats, slope_bases)
            new = arch[arch["dt"] > pd.Timestamp(st["last_ts"])]
            if len(new):
                cand = new[new["p_market"].between(PM_LO, PM_HI)]
                cand = cand[~cand["contract_ticker"].isin(traded)]
                if len(cand):
                    p = model.predict_proba(cand[feats])[:, 1]
                    fee = 0.07 * cand["p_market"] * (1 - cand["p_market"])
                    edge = pd.Series(p, index=cand.index) - cand["p_market"] - fee
                    hits = cand[edge >= EDGE_MIN].copy()
                    hits["p_model"] = pd.Series(p, index=cand.index)[edge >= EDGE_MIN]
                    hits["fee_adj_edge"] = edge[edge >= EDGE_MIN]
                    hits = hits.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
                    for _, r in hits.iterrows():
                        with open(book, "a", newline="") as f:
                            csv.DictWriter(f, fieldnames=BOOK_COLS, extrasaction="ignore").writerow({
                                "logged_at": datetime.now(timezone.utc).isoformat(),
                                "contract_ticker": r["contract_ticker"],
                                "close_ts": r.get("close_ts", ""),
                                "spot": r["spot"], "strike": r["strike"],
                                "p_market": round(float(r["p_market"]), 4),
                                "p_model": round(float(r["p_model"]), 4),
                                "fee_adj_edge": round(float(r["fee_adj_edge"]), 4),
                                "tau_minutes": r["tau_minutes"], "stake": STAKE,
                            })
                        traded.add(r["contract_ticker"])
                        print(f"  [TRADE] YES {r['contract_ticker']} pm={r['p_market']:.3f} "
                              f"p={r['p_model']:.3f} edge={r['fee_adj_edge']:.3f} tau={r['tau_minutes']:.0f}m")
                st["last_ts"] = str(new["dt"].max())
                st["traded"] = sorted(traded)[-3000:]
                state_p.write_text(json.dumps(st))
            # resolve pending from archive backfill
            bk = pd.read_csv(book, low_memory=False)
            pend = bk["resolved_yes"].isna() if len(bk) else pd.Series(dtype=bool)
            if len(bk) and pend.any():
                res = (arch.dropna(subset=["resolved_yes"])
                       .drop_duplicates("contract_ticker", keep="last")
                       .set_index("contract_ticker")["resolved_yes"].to_dict())
                ch = 0
                for i in bk[pend].index:
                    rv = res.get(bk.at[i, "contract_ticker"])
                    if rv is None or (isinstance(rv, float) and rv != rv):
                        continue
                    rv = int(float(rv)); pm = float(bk.at[i, "p_market"]); stk = float(bk.at[i, "stake"])
                    win = rv == 1
                    gross = round(stk * (1 - pm) / pm, 2) if win else -stk
                    feev = round((stk / pm) * 0.07 * pm * (1 - pm), 2)
                    bk.loc[i, ["resolved_yes", "would_win", "would_pnl", "fee_est", "would_pnl_net"]] = \
                        [rv, int(win), gross, feev, round(gross - feev, 2)]
                    ch += 1
                if ch:
                    tmp = book.with_suffix(".csv.tmp")
                    bk.to_csv(tmp, index=False)
                    os.replace(tmp, book)
                    print(f"  [resolve] {ch} filled; book net: "
                          f"{pd.to_numeric(bk['would_pnl_net'], errors='coerce').sum():+.2f}")
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
