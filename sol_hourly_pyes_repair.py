"""SOL hourly production-model calibration repair — 2026-07-29.

The analytic p_yes_model has looked ~10pp too bullish on its (all-YES) trades.
Here we measure it on ALL logged scans (paper_trades_sol.csv no_trade+trade
rows since the 07-07 reset, outcomes joined from sol_scan_archive.csv), then
fit simple corrections on 07-07..07-21 and evaluate ONCE on 07-21..07-30 by
simulated PnL (flat $100, both sides allowed, fee-adj edge>=0.05, one bet per
contract, net of fees) — never by calibration error alone.

Corrections tested (all fit on the fit-window only):
  A. logit shift        p' = expit(logit(p) + b)
  B. Platt              p' = expit(a*logit(p) + b)
  C. market blend       p' = expit(w*logit(p) + (1-w)*logit(pm)), w on grid
  D. none (production as-is)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit, expit
from scipy.optimize import minimize

BASE = Path(__file__).parent
FIT_END = pd.Timestamp("2026-07-21", tz="UTC")


def load() -> pd.DataFrame:
    pt = pd.read_csv(BASE / "results" / "paper_trades_sol.csv", low_memory=False)
    pt["dt"] = pd.to_datetime(pt["logged_at"], errors="coerce", utc=True)
    for c in ["p_yes_model", "p_market"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt = pt.dropna(subset=["dt", "p_yes_model", "p_market", "contract_ticker"])
    arch = pd.read_csv(BASE / "results" / "sol_scan_archive.csv",
                       usecols=["contract_ticker", "resolved_yes"], low_memory=False)
    arch["resolved_yes"] = pd.to_numeric(arch["resolved_yes"], errors="coerce")
    res = (arch.dropna(subset=["resolved_yes"])
               .drop_duplicates("contract_ticker", keep="last")
               .set_index("contract_ticker")["resolved_yes"])
    pt["resolved_yes"] = pt["contract_ticker"].map(res)
    pt = pt.dropna(subset=["resolved_yes"])
    pt = pt[pt["p_market"].between(0.03, 0.97) & pt["p_yes_model"].between(0.01, 0.99)]
    return pt.sort_values("dt").reset_index(drop=True)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def sim_book(df, p, edge_min=0.05):
    s = df.copy()
    s["p"] = p
    fee = 0.07 * s["p_market"] * (1 - s["p_market"])
    ey = s["p"] - s["p_market"] - fee
    en = s["p_market"] - s["p"] - fee
    s["side"] = np.where(ey >= en, "yes", "no")
    s["edge"] = np.maximum(ey, en)
    q = s[s["edge"] >= edge_min].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first")
    cost = np.where(q["side"] == "yes", q["p_market"], 1 - q["p_market"])
    win = np.where(q["side"] == "yes", q["resolved_yes"] == 1, q["resolved_yes"] == 0)
    feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
    pnl = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * feeq
    q = q.copy()
    q["pnl"], q["winb"], q["cost"] = pnl, win, cost
    return q


def summ(q, label):
    if not len(q):
        return f"{label}: n=0"
    ys = (q["side"] == "yes").sum()
    wk = q.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
    wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
    return (f"{label}: n={len(q)} (yes={ys}/no={len(q)-ys}) net=${q['pnl'].sum():+,.0f} "
            f"WR={q['winb'].mean():.1%} BE={q['cost'].mean():.1%} | {wks}")


def main():
    df = load()
    fit = df[df["dt"] < FIT_END]
    test = df[df["dt"] >= FIT_END]
    print(f"scans: fit n={len(fit)} ({fit['dt'].min().date()}..{FIT_END.date()}), "
          f"test n={len(test)} (..{test['dt'].max().date()})")

    for name, d in [("fit", fit), ("test", test)]:
        print(f"  [{name}] mean p_model={d['p_yes_model'].mean():.3f} "
              f"mean pm={d['p_market'].mean():.3f} actual={d['resolved_yes'].mean():.3f} "
              f"| brier model={brier(d['p_yes_model'], d['resolved_yes']):.4f} "
              f"market={brier(d['p_market'], d['resolved_yes']):.4f}")

    lp_f = logit(fit["p_yes_model"]).values
    y_f = fit["resolved_yes"].values
    lm_f = logit(fit["p_market"]).values

    def nll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    b_shift = minimize(lambda b: nll(expit(lp_f + b[0]), y_f), [0.0]).x[0]
    ab = minimize(lambda t: nll(expit(t[0] * lp_f + t[1]), y_f), [1.0, 0.0]).x
    ws = np.arange(0.0, 1.01, 0.1)
    w_best = min(ws, key=lambda w: nll(expit(w * lp_f + (1 - w) * lm_f), y_f))
    print(f"\nfitted: shift b={b_shift:+.3f} | platt a={ab[0]:.3f} b={ab[1]:+.3f} "
          f"| blend w={w_best:.1f} (w=weight on model)")

    lp_t = logit(test["p_yes_model"]).values
    lm_t = logit(test["p_market"]).values
    variants = {
        "D none (production)": test["p_yes_model"].values,
        "A logit-shift": expit(lp_t + b_shift),
        "B platt": expit(ab[0] * lp_t + ab[1]),
        f"C blend w={w_best:.1f}": expit(w_best * lp_t + (1 - w_best) * lm_t),
    }
    print("\nTEST window (07-21..) — simulated books, flat $100, edge>=0.05:")
    for name, p in variants.items():
        q = sim_book(test, p)
        print("  ", summ(q, name))
        print(f"      brier={brier(p, test['resolved_yes'].values):.4f}")

    print("\nfit-window books (in-sample reference, same rule):")
    lv = {"D none": fit["p_yes_model"].values,
          "A shift": expit(lp_f + b_shift),
          "B platt": expit(ab[0] * lp_f + ab[1]),
          f"C blend {w_best:.1f}": expit(w_best * lp_f + (1 - w_best) * lm_f)}
    for name, p in lv.items():
        print("  ", summ(sim_book(fit, p), name))


if __name__ == "__main__":
    main()
