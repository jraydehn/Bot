"""Maker-execution simulation for the SOL hourly composite book. 2026-08-04.

Question: does maker execution (resting limits instead of crossing) recover
enough of the 3-4c spread to flip the composite-signal book — whose signals
show clean dose-response (WR 56.7->60.9% with |score|) but lose to taker/mid
economics — into viability?

Fill model (per-minute candles from kalshi_hourly_candle_backfill):
  YES buy: at signal minute t, rest limit at yes_bid(t). CONSERVATIVE fill:
    a later minute's traded price_low <= limit - 0.01 (strictly through —
    queue priority unknowable). OPTIMISTIC: price_low <= limit.
  NO buy: rest at no_bid = 1 - yes_ask(t). CONSERVATIVE fill: later traded
    price_high >= yes_ask(t) + 0.01. OPTIMISTIC: >= yes_ask(t).
  Unfilled by close = missed trade (the maker's real cost: opportunity).
  Kalshi fees still apply on fills (maker saves spread, not fees).

Protocol: K swept on screen (<07-09) UNDER MAKER ECONOMICS (new economics =
legitimate fresh selection), frozen eval 07-09+ single shot, vs the
mid-priced book as reference. Composite signals recomputed exactly as
sol_hourly_composite_backtest (frozen constants, no fitting).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sol_hourly_composite_backtest import assemble, SIGNALS, SPLIT, PM_LO, PM_HI

BASE = Path(__file__).parent


def load_candles():
    c = pd.read_csv(BASE / "results" / "kalshi_1h_candles_sol.csv", low_memory=False)
    for col in ["bid_close", "ask_close", "price_close", "price_low", "price_high", "end_ts"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["end_ts"]).sort_values(["ticker", "end_ts"])
    return c


def maker_fill(candles_t, t_epoch, side, conservative=True):
    """Return (filled, cost) for a limit rested at minute t's quote."""
    now = candles_t[candles_t["end_ts"] <= t_epoch]
    later = candles_t[candles_t["end_ts"] > t_epoch]
    if now.empty or later.empty:
        return False, np.nan
    q = now.iloc[-1]
    eps = 0.01 if conservative else 0.0
    if side == "yes":
        lim = q["bid_close"]
        if not (lim == lim) or lim <= 0:
            return False, np.nan
        hit = (later["price_low"] <= lim - eps).any()
        return bool(hit), float(lim)
    else:
        ask = q["ask_close"]
        if not (ask == ask) or ask >= 1:
            return False, np.nan
        hit = (later["price_high"] >= ask + eps).any()
        return bool(hit), float(1 - ask)


def main():
    print("assembling composite signals…")
    df = assemble()
    scr_mask = df["dt"] < SPLIT
    consts = {s: (float(df.loc[scr_mask, s].mean()), float(df.loc[scr_mask, s].std()))
              for s in SIGNALS}
    df["score"] = sum((df[s] - consts[s][0]) / consts[s][1] for s in SIGNALS)
    df = df[df["p_market"].between(PM_LO, PM_HI)].copy()
    df["side"] = np.where(df["score"] > 0, "yes", "no")
    df["t_epoch"] = df["dt"].astype("int64") / 1e9

    print("loading candles…")
    cd = load_candles()
    groups = {t: g for t, g in cd.groupby("ticker")}
    print(f"candle tickers: {len(groups)}")

    def book(dd, K, conservative=True):
        q = dd[dd["score"].abs() >= K].sort_values("dt").drop_duplicates(
            "contract_ticker", keep="first")
        rows = []
        n_nocandle = 0
        for _, r in q.iterrows():
            g = groups.get(r["contract_ticker"])
            if g is None:
                n_nocandle += 1
                continue
            filled, cost = maker_fill(g, r["t_epoch"], r["side"], conservative)
            if not filled or not (0.02 <= cost <= 0.98):
                rows.append({"dt": r["dt"], "filled": False, "pnl": 0.0,
                             "win": np.nan, "cost": np.nan})
                continue
            win = (r["resolved_yes"] == 1) if r["side"] == "yes" else (r["resolved_yes"] == 0)
            pm_eff = cost if r["side"] == "yes" else 1 - cost
            fee = 0.07 * pm_eff * (1 - pm_eff)
            pnl = (100 * (1 - cost) / cost if win else -100.0) - (100 / cost) * fee
            rows.append({"dt": r["dt"], "filled": True, "pnl": pnl,
                         "win": bool(win), "cost": cost})
        b = pd.DataFrame(rows)
        return b, len(q), n_nocandle

    def summ(b, n_sig, n_nc, label):
        if b is None or not len(b):
            return f"{label}: no rows"
        f = b[b["filled"]]
        fr = len(f) / max(len(b), 1)
        wk = f.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum() if len(f) else pd.Series(dtype=float)
        wk = wk[wk != 0]
        wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk.items())
        return (f"{label}: signals={n_sig} candle-miss={n_nc} fillrate={fr:.0%} "
                f"fills={len(f)} net=${f['pnl'].sum():+,.0f} "
                f"WR={f['win'].mean():.1%} BE={f['cost'].mean():.1%} "
                f"wk_green={(wk > 0).mean():.0%} | {wks}")

    scr = df[scr_mask]
    ev = df[~scr_mask]
    print("\n[SCREEN <07-09, MAKER-CONSERVATIVE] K sweep:")
    best_K, best_net = None, -1e18
    for K in [1.0, 1.5, 2.0, 2.5]:
        b, n_sig, n_nc = book(scr, K, conservative=True)
        print("  ", summ(b, n_sig, n_nc, f"|score|>={K}"))
        f = b[b["filled"]]
        if len(f) >= 40:
            wk = f.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
            green = (wk[wk != 0] > 0).mean() if (wk != 0).any() else 0
            if f["pnl"].sum() > best_net and green >= 0.6:
                best_K, best_net = K, f["pnl"].sum()
    if best_K is None:
        print("\nNO K clears the screen bar even at maker — stopping honestly.")
        return
    print(f"\nCHOSEN K={best_K} → [FROZEN EVAL 07-09.., conservative + optimistic]:")
    b, n_sig, n_nc = book(ev, best_K, conservative=True)
    print("  ", summ(b, n_sig, n_nc, "maker-CONSERVATIVE"))
    b2, n2, nc2 = book(ev, best_K, conservative=False)
    print("  ", summ(b2, n2, nc2, "maker-OPTIMISTIC"))


if __name__ == "__main__":
    main()
