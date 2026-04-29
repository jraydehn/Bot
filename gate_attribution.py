#!/usr/bin/env python3
"""
gate_attribution.py — Per-gate PnL attribution against the Kalshi scan archive.

For each gate in the production stack, replay every scan-hour at the
*current* production calibration (BTC drift=1.00, ETH=0.80, SOL=0.20,
BTC isotonic if present) and measure:

  baseline      — full production gate stack ON
  leave_one_out — that gate OFF, all others ON   → PnL_baseline - PnL_LOO = gate's contribution
  solo          — only that gate ON              → diagnostic only

Then run A/B counterfactuals on the four 2026-04-27 changes:
  • Gate NS BTC threshold 0.40 → 0.50
  • Block BTC YES p_market < 0.15 (hard)
  • Gate NS ETH threshold 0.45 → 0.55
  • Block ETH YES OTM p_market < 0.45

This is diagnosis only. No production code is touched.
"""

import math, sys, glob, warnings, time, datetime as dt, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

RESULTS_DIR = Path(__file__).parent / "results"
BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD

# ---- production calibration as of 2026-04-28 ----
DRIFT_MULT = {"BTC": 1.00, "ETH": 0.80, "SOL": 0.20}

ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max": 0.50, "g3_min": 0.01,
            "yes_pm_min": 0.55, "yes_pm_otm_block": 0.15},   # btc_otm_gate
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max": 0.55, "g3_min": 0.005,
            "yes_pm_min": 0.35, "eth_otm_yes_block": 0.45},  # eth_otm_gate (only when strike>spot)
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max": 0.55, "g3_min": 0.01,
            "yes_pm_min": None},
}
GCS_MIN = 0.55           # YES OTM requires composite_p_up ≥ 0.55
GCI_MIN_BEARISH = 0.45   # YES ITM (offset≤0) requires composite_p_up ≥ 0.45
RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = 4.0, 0.33, 3.0, 0.08
PURE_EDGE_THRESHOLD = 0.08

# Counter-tape thresholds (severity == 1 at thr; ≥1.5 hard block; 0.5-1.5 dampen)
TAPE_THRESHOLDS = {
    "BTC": {"chg_5m": 0.0016, "chg_10m": 0.0024, "chg_30m": 0.0040},
    "ETH": {"chg_5m": 0.0015, "chg_10m": 0.0025, "chg_30m": 0.0040},
    "SOL": {"chg_5m": 0.0025, "chg_10m": 0.0040, "chg_30m": 0.0065},
}

# Edge-tier OTM minimums (Gate OTM)
OTM_TIERS = [(0.15, 0.04), (0.25, 0.03), (0.35, 0.02)]  # (pm_threshold, min_net_edge)

# All gate names — used for leave-one-out attribution
ALL_GATES = {
    "G0_pm",        # saturation pm bounds
    "GCS",          # YES OTM p_up ≥ 0.55
    "GCI",          # YES ITM bearish p_up ≥ 0.45
    "GNS",          # NO OTM p_up ≤ ns_max
    "GOTM",         # YES p_market tier minimum edges
    "G3",           # baseline edge floor
    "GRR",          # RR bounds
    # NOTE: legacy Gate PM (BTC YES pm≥0.55, ETH YES pm≥0.35) is guarded by
    # `if not composite_active` in decision.py:384 — it does not fire when
    # composite is active (every row in this archive). Excluded from attribution.
    "Gpm15_btc",    # BTC YES p_market < 0.15 hard block (2026-04-27)
    "Gpm45_eth",    # ETH YES OTM p_market < 0.45 hard block (2026-04-27)
    "Gtape",        # counter-tape severity
    "Gpup_btc",     # BTC YES composite_p_up < 0.52 hard block (with rescue)
}

# ---------- isotonic load (BTC only, optional) ----------
ISO_PATH = Path(__file__).parent / "reform_results" / "btc_iso_calibration.pkl"
_iso = None
if ISO_PATH.exists():
    try:
        with open(ISO_PATH, "rb") as f:
            _iso = pickle.load(f)
    except Exception:
        _iso = None

def apply_iso(asset, p):
    if asset != "BTC" or _iso is None:
        return p
    try:
        return float(np.clip(_iso.predict([p])[0], 0.01, 0.99))
    except Exception:
        return p


# ---------- archive loading ----------
def load_archive(asset):
    if asset == "BTC":
        patterns = ["paper_trades_archive_2026*.csv", "paper_trades.csv"]
    elif asset == "ETH":
        patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    else:
        patterns = ["paper_trades_sol_archive_*.csv", "paper_trades_sol.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))
    if asset == "BTC":
        files = [f for f in files if "_eth" not in f.name and "_sol" not in f.name]
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    needed = ["decision_time", "contract_ticker", "spot", "strike", "p_market",
              "vol_eff", "tau_minutes", "composite_trend", "composite_rev",
              "composite_p_up", "resolved_yes", "chg_30m", "chg_10m", "chg_5m"]
    for c in needed:
        if c not in raw.columns:
            return pd.DataFrame()
    raw = raw.dropna(subset=["decision_time", "contract_ticker", "spot", "strike",
                              "p_market", "vol_eff", "tau_minutes",
                              "composite_trend", "composite_rev", "composite_p_up",
                              "resolved_yes"])
    for c in ["spot", "strike", "p_market", "vol_eff", "tau_minutes",
              "composite_trend", "composite_rev", "composite_p_up",
              "resolved_yes", "chg_30m", "chg_10m", "chg_5m"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in ["chg_30m", "chg_10m", "chg_5m"]:
        raw[c] = raw[c] / 100.0  # stored as percent
    raw = raw.dropna(subset=["spot", "strike", "p_market", "vol_eff",
                              "tau_minutes", "composite_p_up", "resolved_yes"])
    raw = raw.drop_duplicates(subset=["decision_time", "contract_ticker"], keep="last")
    raw = raw.sort_values("decision_time").reset_index(drop=True)
    return raw


# ---------- pricing ----------
def compute_pmodel(spot, strike, vol_eff, tau, p_up, k_drift):
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k_drift
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def evaluate_row(row, asset, gates, ns_max_override=None):
    """
    Replay a scan row's candidate selection with `gates` ON.
    Returns best dict {side, pm, p_model, net_edge, offset, would_win} or None.
    """
    p = ASSET_PARAMS[asset]
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    vol_eff = row["vol_eff"]; tau = row["tau_minutes"]
    p_up = row["composite_p_up"]; resolved_yes = int(row["resolved_yes"])

    if vol_eff <= 0 or tau <= 0 or pm <= 0 or pm >= 1:
        return None
    offset = (strike - spot) / spot if spot > 0 else 0.0
    p_model_raw = compute_pmodel(spot, strike, vol_eff, tau, p_up, DRIFT_MULT[asset])
    p_model = apply_iso(asset, p_model_raw)
    ns_max = ns_max_override if ns_max_override is not None else p["ns_max"]

    best = None
    for side in ("yes", "no"):
        # G0_pm — saturation bounds on p_market
        if "G0_pm" in gates:
            if not (p["pm_min"] <= pm <= p["pm_max"]):
                continue
        # GpmYesMin — YES p_market floors
        if side == "yes" and "GpmYesMin" in gates:
            if asset == "BTC" and pm < p["yes_pm_min"]:
                continue
            if asset == "ETH" and pm < p["yes_pm_min"]:
                continue
        # Gpm15_btc — BTC YES p_market < 0.15 hard block (2026-04-27)
        if side == "yes" and asset == "BTC" and "Gpm15_btc" in gates and pm < 0.15:
            continue
        # Gpm45_eth — ETH YES OTM p_market < 0.45 hard block (2026-04-27)
        if side == "yes" and asset == "ETH" and "Gpm45_eth" in gates \
                and offset > 0 and pm < 0.45:
            continue
        # GCS / GCI — YES composite gates
        if side == "yes":
            if "GCS" in gates and offset > 0 and p_up < GCS_MIN:
                continue
            if "GCI" in gates and offset <= 0 and p_up < GCI_MIN_BEARISH:
                continue
        # GNS — NO OTM composite gate
        if side == "no":
            if "GNS" in gates and offset < 0 and p_up > ns_max:
                continue
        # Gpup_btc — BTC YES p_up < 0.52 hard block (with net_edge ≥ 4% rescue)
        if side == "yes" and asset == "BTC" and "Gpup_btc" in gates and p_up < 0.52:
            # rescue: requires net_edge ≥ 4% — compute it first
            pre_fee = (p_model - pm) if side == "yes" else (pm - p_model)
            ne = pre_fee - kalshi_fee(pm) - SLIPPAGE - SPREAD
            if ne < 0.04:
                continue

        # Edge math
        fee = kalshi_fee(pm)
        if side == "yes":
            raw = p_model - pm
            net = raw - fee - SLIPPAGE - SPREAD
            rr = pm / (1 - pm) if pm < 1 else 999
            if "GRR" in gates and rr > RR_MAX_YES:
                continue
            if "GOTM" in gates:
                tier_min = 0.0
                for thr, mn in OTM_TIERS:
                    if pm < thr:
                        tier_min = mn
                        break
                if net < tier_min:
                    continue
        else:
            raw = pm - p_model
            net = raw - fee - SLIPPAGE - SPREAD
            rr = (1 - pm) / pm if pm > 0 else 999
            if "GRR" in gates and (rr < RR_MIN_NO or rr > RR_MAX_NO) and net < RR_EDGE_EXC:
                continue
        if "G3" in gates and net < p["g3_min"]:
            continue

        won = (resolved_yes == 1 and side == "yes") or (resolved_yes == 0 and side == "no")
        if best is None or net > best["net"]:
            best = {"side": side, "pm": pm, "p_model": p_model, "net": net,
                    "offset": offset, "won": won, "p_up": p_up,
                    "chg_5m": row.get("chg_5m", 0), "chg_10m": row.get("chg_10m", 0),
                    "chg_30m": row.get("chg_30m", 0)}
    return best


def severity(side, c5, c10, c30, thr):
    if any(pd.isna(x) for x in (c5, c10, c30)):
        return 0.0
    if side == "yes":
        c5, c10, c30 = -c5, -c10, -c30
    return max(0.0, c5 / thr["chg_5m"], c10 / thr["chg_10m"], c30 / thr["chg_30m"])


def kelly_bet(p_model, pm, side, bankroll, kelly_scale=1.0):
    if side == "yes":
        b = (1 - pm) / pm if pm > 0 else 0
        p, q = p_model, 1 - p_model
    else:
        b = pm / (1 - pm) if pm < 1 else 0
        p, q = 1 - p_model, p_model
    if b <= 0:
        return 0.0
    kf = max(0.0, (b * p - q) / b)
    bf = min(kf * KELLY_MULT * kelly_scale, KELLY_CAP)
    return round(bankroll * bf, 2)


def trade_pnl(bet, side, pm, won):
    if bet <= 0:
        return 0.0
    fee = kalshi_fee(pm)
    if side == "yes":
        if won:
            n_ct = bet / pm if pm > 0 else 0
            return bet * (1 - pm) / pm - fee * n_ct
        return -bet
    else:
        if won:
            n_ct = bet / (1 - pm) if pm < 1 else 0
            return bet * pm / (1 - pm) - fee * n_ct
        return -bet


def run_backtest(asset, df, gates, ns_max_override=None):
    """One pass through scans → trade list → summary."""
    thr = TAPE_THRESHOLDS[asset]
    bankroll = BANKROLL_0
    trades = []  # (decision_time, side, pm, p_model, net, won, pnl, sev, dampened)
    for dt_, group in df.groupby("decision_time", sort=True):
        cands = []
        for _, row in group.iterrows():
            c = evaluate_row(row, asset, gates, ns_max_override=ns_max_override)
            if c is not None:
                cands.append(c)
        if not cands:
            continue
        # Pick highest net_edge OR pure-edge bypass (any cand with net >= 0.08 wins
        # against gates we may have switched off)
        best = max(cands, key=lambda c: c["net"])
        # Counter-tape gate
        sev = severity(best["side"], best["chg_5m"], best["chg_10m"],
                       best["chg_30m"], thr) if "Gtape" in gates else 0.0
        kelly_scale = 1.0; dampened = False
        if "Gtape" in gates and sev >= 1.5:
            trades.append({"dt": dt_, "side": best["side"], "pm": best["pm"],
                           "p_model": best["p_model"], "net": best["net"],
                           "won": best["won"], "pnl": 0.0, "sev": sev,
                           "blocked_by_tape": True, "dampened": False,
                           "would_have_pnl": trade_pnl(
                                kelly_bet(best["p_model"], best["pm"], best["side"], bankroll),
                                best["side"], best["pm"], best["won"])})
            continue
        if "Gtape" in gates and sev >= 0.5:
            kelly_scale = max(0.25, 1.0 - (sev - 0.5) * 0.75)
            dampened = True
        bet = kelly_bet(best["p_model"], best["pm"], best["side"], bankroll, kelly_scale)
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, best["side"], best["pm"], best["won"])
        bankroll = max(1.0, bankroll + pnl)
        trades.append({"dt": dt_, "side": best["side"], "pm": best["pm"],
                       "p_model": best["p_model"], "net": best["net"],
                       "won": best["won"], "pnl": pnl, "sev": sev,
                       "blocked_by_tape": False, "dampened": dampened,
                       "would_have_pnl": pnl})
    df_t = pd.DataFrame(trades)
    if df_t.empty:
        return {"n": 0, "wins": 0, "wr": 0.0, "pnl": 0.0, "trades": df_t,
                "max_dd": 0.0, "blocked_by_tape": 0}
    taken = df_t[~df_t["blocked_by_tape"]]
    n = len(taken); wins = int(taken["won"].sum())
    pnl = float(taken["pnl"].sum())
    # max drawdown
    eq = (BANKROLL_0 + taken["pnl"].cumsum()).tolist()
    peak = BANKROLL_0; dd = 0.0
    for e in eq:
        peak = max(peak, e); dd = max(dd, (peak - e) / peak)
    return {"n": n, "wins": wins, "wr": wins / n if n else 0.0, "pnl": pnl,
            "trades": df_t, "max_dd": dd,
            "blocked_by_tape": int(df_t["blocked_by_tape"].sum())}


# ---------- attribution & A/B ----------
def attribute(asset, df):
    print(f"\n{'='*78}\n  [{asset}] gate attribution — {len(df):,} scans, "
          f"{df['decision_time'].nunique():,} hours\n{'='*78}", flush=True)
    base = run_backtest(asset, df, ALL_GATES)
    print(f"  baseline (all gates ON):  n={base['n']:4d}  WR={base['wr']:.1%}  "
          f"pnl=${base['pnl']:+8.2f}  maxDD={base['max_dd']:.1%}  "
          f"tape_blocks={base['blocked_by_tape']}", flush=True)
    rows = []
    for g in sorted(ALL_GATES):
        loo_gates = ALL_GATES - {g}
        loo = run_backtest(asset, df, loo_gates)
        # gate's contribution = baseline - LOO (positive → gate helps)
        delta = base["pnl"] - loo["pnl"]
        # what trades does this gate block? trades in LOO not in baseline (by dt)
        base_dts = set(base["trades"][~base["trades"]["blocked_by_tape"]]["dt"])
        loo_dts = set(loo["trades"][~loo["trades"]["blocked_by_tape"]]["dt"])
        only_loo = loo_dts - base_dts
        blocked_df = loo["trades"][loo["trades"]["dt"].isin(only_loo)]
        n_blocked = len(blocked_df)
        wins_blocked = int(blocked_df["won"].sum()) if n_blocked else 0
        losses_blocked = n_blocked - wins_blocked
        pnl_of_blocked = float(blocked_df["pnl"].sum()) if n_blocked else 0.0
        rows.append({
            "gate": g, "delta_pnl": delta,
            "n_blocks": n_blocked,
            "wins_blocked": wins_blocked, "losses_blocked": losses_blocked,
            "wr_blocked": wins_blocked / n_blocked if n_blocked else 0.0,
            "pnl_of_blocked_trades": pnl_of_blocked,
            "loo_n": loo["n"], "loo_wr": loo["wr"], "loo_pnl": loo["pnl"],
            "loo_dd": loo["max_dd"],
        })
    df_attr = pd.DataFrame(rows).sort_values("delta_pnl", ascending=False)
    print(f"\n  gate                  Δpnl     n_blk  wins/loss  wr_blk  $blk_trades  loo_pnl  loo_DD", flush=True)
    for _, r in df_attr.iterrows():
        print(f"  {r['gate']:<18s}  {r['delta_pnl']:+8.2f}  {r['n_blocks']:4d}   "
              f"{r['wins_blocked']:3d}/{r['losses_blocked']:<3d}   "
              f"{r['wr_blocked']:.1%}   ${r['pnl_of_blocked_trades']:+7.2f}   "
              f"${r['loo_pnl']:+8.2f}  {r['loo_dd']:.1%}", flush=True)
    df_attr.insert(0, "asset", asset)
    df_attr["baseline_pnl"] = base["pnl"]
    df_attr["baseline_n"] = base["n"]
    df_attr["baseline_wr"] = base["wr"]
    df_attr["baseline_dd"] = base["max_dd"]
    return df_attr


def ab_test_recent_changes(df_btc, df_eth):
    """Counterfactuals on the four 2026-04-27 changes."""
    print(f"\n{'='*78}\n  A/B counterfactuals on 2026-04-27 changes\n{'='*78}", flush=True)
    rows = []

    # 1) NS BTC 0.40 → 0.50: rerun baseline with ns_max=0.40 on BTC
    new = run_backtest("BTC", df_btc, ALL_GATES, ns_max_override=0.50)
    old = run_backtest("BTC", df_btc, ALL_GATES, ns_max_override=0.40)
    rows.append({"change": "BTC GNS 0.40→0.50", "asset": "BTC",
                 "old_pnl": old["pnl"], "new_pnl": new["pnl"],
                 "delta": new["pnl"] - old["pnl"],
                 "old_n": old["n"], "new_n": new["n"]})

    # 2) BTC YES pm<0.15 hard block: rerun with Gpm15_btc OFF
    on = run_backtest("BTC", df_btc, ALL_GATES)
    off = run_backtest("BTC", df_btc, ALL_GATES - {"Gpm15_btc"})
    rows.append({"change": "BTC block YES pm<0.15", "asset": "BTC",
                 "old_pnl": off["pnl"], "new_pnl": on["pnl"],
                 "delta": on["pnl"] - off["pnl"],
                 "old_n": off["n"], "new_n": on["n"]})

    # 3) NS ETH 0.45 → 0.55
    new_e = run_backtest("ETH", df_eth, ALL_GATES, ns_max_override=0.55)
    old_e = run_backtest("ETH", df_eth, ALL_GATES, ns_max_override=0.45)
    rows.append({"change": "ETH GNS 0.45→0.55", "asset": "ETH",
                 "old_pnl": old_e["pnl"], "new_pnl": new_e["pnl"],
                 "delta": new_e["pnl"] - old_e["pnl"],
                 "old_n": old_e["n"], "new_n": new_e["n"]})

    # 4) ETH YES OTM pm<0.45 hard block
    on_e = run_backtest("ETH", df_eth, ALL_GATES)
    off_e = run_backtest("ETH", df_eth, ALL_GATES - {"Gpm45_eth"})
    rows.append({"change": "ETH block YES OTM pm<0.45", "asset": "ETH",
                 "old_pnl": off_e["pnl"], "new_pnl": on_e["pnl"],
                 "delta": on_e["pnl"] - off_e["pnl"],
                 "old_n": off_e["n"], "new_n": on_e["n"]})

    df_ab = pd.DataFrame(rows)
    print(f"\n  change                          old_pnl    new_pnl    Δ          old_n  new_n", flush=True)
    for _, r in df_ab.iterrows():
        print(f"  {r['change']:<28s}    ${r['old_pnl']:+8.2f}  ${r['new_pnl']:+8.2f}  "
              f"${r['delta']:+8.2f}   {r['old_n']:4d}  {r['new_n']:4d}", flush=True)
    return df_ab


def main():
    t0 = time.time()
    print(f"gate_attribution.py — production calibration: drift={DRIFT_MULT}, "
          f"isotonic={'BTC loaded' if _iso else 'none'}", flush=True)
    btc = load_archive("BTC"); eth = load_archive("ETH"); sol = load_archive("SOL")
    attrs = []
    for asset, df in (("BTC", btc), ("ETH", eth), ("SOL", sol)):
        if df.empty:
            print(f"  [{asset}] no data"); continue
        attrs.append(attribute(asset, df))
    df_ab = ab_test_recent_changes(btc, eth)

    # write CSVs
    today = dt.datetime.now().strftime("%Y%m%d")
    out_attr = RESULTS_DIR / f"gate_attribution_{today}.csv"
    out_ab = RESULTS_DIR / f"gate_attribution_ab_{today}.csv"
    if attrs:
        pd.concat(attrs, ignore_index=True).to_csv(out_attr, index=False)
        print(f"\n  wrote {out_attr}", flush=True)
    df_ab.to_csv(out_ab, index=False)
    print(f"  wrote {out_ab}", flush=True)
    print(f"  total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
