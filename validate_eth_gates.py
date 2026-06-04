"""
validate_eth_gates.py

Permutation + walk-forward significance tests for live ETH gates.

Gates tested:
  Archive method (condition replicated from scan archive columns):
    eth_no_vwap_stretch2_gate  — NO, vwap_stretch==2 + not(tau>40 & vol_score==0)
    eth_no_adx_gate            — NO, adx_1h>40 + not(ema_stack==-1)  [partial: drops vol_ratio rescue]
    eth_funding_vol_yes_gate   — YES, funding_bias==-1                  [partial: drops vol_ratio>=0.70]
    eth_squeeze_gate           — BOTH, squeeze_1h==1 & stoch_k not in [30,60) [partial: drops vol_ratio<0.80]

  Blocked-trades outcome join (vol_ratio not in archive):
    eth_vol_regime_gate        — needs vol_ratio>1.20; join blocked_trades to archive for outcomes

Two tests per gate:
  1. Standard permutation (full archive, label shuffle within pm bins)
  2. Walk-forward (train=first 75%, OOS=last 25%; permute only OOS outcomes)

Usage:
  python3 validate_eth_gates.py [--n_perms 500] [--gate all|<name>] [--train_frac 0.75]
"""
import argparse
import math
from pathlib import Path
from scipy import stats as scipy_stats

import numpy as np
import pandas as pd

BASE    = Path(__file__).parent
RESULTS = BASE / "results"
FEE     = 0.07
FLAT    = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--n_perms",    type=int,   default=500)
parser.add_argument("--train_frac", type=float, default=0.75)
parser.add_argument("--gate", default="all",
                    choices=["all",
                             "eth_no_vwap_stretch2_gate",
                             "eth_no_adx_gate",
                             "eth_funding_vol_yes_gate",
                             "eth_squeeze_gate",
                             "eth_vol_regime_gate"])
args = parser.parse_args()


# ── Load scan archive ─────────────────────────────────────────────────────────

print("Loading ETH scan archive …")
sa = pd.read_csv(RESULTS / "eth_scan_archive.csv", low_memory=False)
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
if "close_ts" in sa.columns:
    sa["close_ts"] = pd.to_datetime(sa["close_ts"], errors="coerce", utc=True)
sa = sa[sa["resolved_yes"].notna()].copy()
sa["p_market"] = pd.to_numeric(sa["p_market"], errors="coerce")

for col in ["vwap_stretch_score", "squeeze_1h", "adx_1h", "ema_stack_bias",
            "funding_bias", "stoch_k", "vol_score", "tau_minutes",
            "composite_trend", "composite_rev"]:
    if col in sa.columns:
        sa[col] = pd.to_numeric(sa[col], errors="coerce")

print(f"  {len(sa):,} resolved rows  |  "
      f"{sa['logged_at'].min().date()} → {sa['logged_at'].max().date()}")


# ── Load blocked trades for outcome-join gates ────────────────────────────────

print("Loading blocked trades …")
bt = pd.read_csv(RESULTS / "blocked_trades.csv", low_memory=False)
bt["logged_at"] = pd.to_datetime(bt["logged_at"], errors="coerce", utc=True)
bt["close_ts"]  = pd.to_datetime(bt["close_ts"],  errors="coerce", utc=True)
bt["p_market"]  = pd.to_numeric(bt["pm"],         errors="coerce")
bt_eth = bt[(bt["asset"] == "ETH") & (bt["logged_at"] >= sa["logged_at"].min())].copy()
print(f"  ETH blocked (archive era): {len(bt_eth):,} rows")


# ── P&L helpers ───────────────────────────────────────────────────────────────

def pnl_vec(pm_arr, won_arr, block_mask=None):
    """
    Flat P&L for a portfolio of bets.
    won_arr: bool array (True = bet wins).
    For NO bets: won = resolved_yes==0. For YES bets: won = resolved_yes==1.
    Formula is symmetric: win → +(1-pm)*(1-fee)*flat; lose → -pm*(1-fee)*flat.
    """
    keep = ~block_mask if block_mask is not None else np.ones(len(pm_arr), dtype=bool)
    n_blk = int(block_mask.sum()) if block_mask is not None else 0
    pm  = pm_arr[keep]
    won = won_arr[keep]
    return float((np.where(won, (1 - pm), -pm) * (1 - FEE) * FLAT).sum()), n_blk


def bkev_for_side(pm_arr, side):
    """Breakeven WR: pm for YES bets, 1-pm for NO bets."""
    return float(pm_arr.mean() if side == "yes" else (1 - pm_arr).mean())


# ── Gate masks (archive method) ───────────────────────────────────────────────

GATE_SIDE = {
    "eth_no_vwap_stretch2_gate": "no",
    "eth_no_adx_gate":           "no",
    "eth_funding_vol_yes_gate":  "yes",
    "eth_squeeze_gate":          "both",
    "eth_vol_regime_gate":       "both",
}


def make_gate_mask(df, gate_name):
    """Return bool Series: True = block this row."""

    if gate_name == "eth_no_vwap_stretch2_gate":
        stretch  = pd.to_numeric(df.get("vwap_stretch_score", np.nan), errors="coerce").fillna(0)
        tau      = pd.to_numeric(df.get("tau_minutes",        0),      errors="coerce").fillna(0)
        vol_sc   = pd.to_numeric(df.get("vol_score",          0),      errors="coerce").fillna(0)
        rescue   = (tau > 40) & (vol_sc == 0)
        return (stretch == 2) & ~rescue

    elif gate_name == "eth_no_adx_gate":
        # Partial: drops vol_ratio rescue (conservative — blocks more than actual gate).
        # Actual gate also rescues when vol_ratio∈[0.80,1.20); omitted here.
        adx = pd.to_numeric(df.get("adx_1h",       0),   errors="coerce").fillna(0)
        ema = pd.to_numeric(df.get("ema_stack_bias", 0),  errors="coerce").fillna(0)
        return (adx > 40) & (ema != -1)

    elif gate_name == "eth_funding_vol_yes_gate":
        # Partial: drops vol_ratio>=0.70 condition (conservative — blocks more than actual gate).
        fund = pd.to_numeric(df.get("funding_bias", 0), errors="coerce").fillna(0)
        return fund == -1

    elif gate_name == "eth_squeeze_gate":
        # Partial: drops vol_ratio<0.80 condition.
        sq  = pd.to_numeric(df.get("squeeze_1h", 0), errors="coerce").fillna(0)
        sk  = pd.to_numeric(df.get("stoch_k",   50), errors="coerce").fillna(50)
        rescue = (sk >= 30) & (sk < 60)
        return (sq == 1) & ~rescue

    else:
        raise ValueError(f"No archive mask for gate: {gate_name}")


# ── Permutation helper ────────────────────────────────────────────────────────

def shuffle_within_bins(won_arr, pm_bins, rng):
    wp = won_arr.copy()
    for b in range(11):
        idx = np.where(pm_bins == b)[0]
        if len(idx) > 1:
            vals = wp[idx]; rng.shuffle(vals); wp[idx] = vals
    return wp


def shuffle_test_only(won_arr, test_mask, pm_bins, rng):
    wp = won_arr.copy()
    for b in range(11):
        idx = np.where((pm_bins == b) & test_mask)[0]
        if len(idx) > 1:
            vals = wp[idx]; rng.shuffle(vals); wp[idx] = vals
    return wp


# ── Archive-method test ───────────────────────────────────────────────────────

def run_archive_test(gate_name, sa, n_perms, train_frac):
    side    = GATE_SIDE[gate_name]
    pm_arr  = sa["p_market"].astype(float).values
    ry      = pd.to_numeric(sa["resolved_yes"], errors="coerce").fillna(-1)
    # For "both" gates: use NO side (most fires are NO in ETH squeeze gate)
    eff_side = "no" if side == "both" else side
    won_arr  = (ry == (1 if eff_side == "yes" else 0)).values

    block_arr = make_gate_mask(sa, gate_name).values
    pm_blk    = pm_arr[block_arr]
    won_blk   = won_arr[block_arr]

    pnl_all,  _     = pnl_vec(pm_arr, won_arr)
    pnl_gate, n_blk = pnl_vec(pm_arr, won_arr, block_arr)
    real_delta = pnl_gate - pnl_all

    blk_wr   = float(won_blk.mean())    if n_blk else float("nan")
    blk_bkev = bkev_for_side(pm_blk, eff_side) if n_blk else float("nan")
    blk_pnl, _ = pnl_vec(pm_blk, won_blk)

    print(f"\n── {gate_name}  [archive, side={eff_side}]  "
          + ("[PARTIAL — vol_ratio condition omitted]" if gate_name in
             ("eth_no_adx_gate", "eth_funding_vol_yes_gate", "eth_squeeze_gate") else ""))
    print(f"  Ungated : ${pnl_all:+,.2f}  ({len(sa):,} bets)")
    print(f"  Gated   : ${pnl_gate:+,.2f}  ({len(sa)-n_blk:,} bets, {n_blk} blocked)")
    print(f"  Δ (real): ${real_delta:+,.2f}")
    if n_blk:
        print(f"  Blocked : n={n_blk}  WR={blk_wr:.1%}  bkev={blk_bkev:.1%}  "
              f"edge={blk_wr-blk_bkev:+.1%}  PnL=${blk_pnl:+,.2f}")

    # ── Standard permutation ──────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    pm_bins = np.floor(pm_arr * 10).astype(int)
    perm_deltas = []
    print(f"  Perm ({n_perms}) …", end=" ", flush=True)
    for i in range(n_perms):
        wp = shuffle_within_bins(won_arr, pm_bins, rng)
        pg, _ = pnl_vec(pm_arr, wp, block_arr)
        pu, _ = pnl_vec(pm_arr, wp)
        perm_deltas.append(pg - pu)
        if (i + 1) % 100 == 0:
            print(i + 1, end=" ", flush=True)
    print("done.")

    pd_arr = np.array(perm_deltas)
    p_perm = float((pd_arr >= real_delta).mean())
    sig    = p_perm < 0.05
    print(f"  Perm  : Δ=${real_delta:+,.2f}  null p5/p50/p95 = "
          f"${np.percentile(pd_arr,5):+,.2f}/${np.percentile(pd_arr,50):+,.2f}/${np.percentile(pd_arr,95):+,.2f}"
          f"  p={p_perm:.3f}  {'✓ SIGNIFICANT' if sig else 'not significant'}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    sa_s = sa.sort_values("logged_at").reset_index(drop=True)
    split = int(len(sa_s) * train_frac)
    split_ts = sa_s["logged_at"].iloc[split]
    test_mask = np.zeros(len(sa_s), dtype=bool)
    test_mask[split:] = True
    test_idx = np.where(test_mask)[0]

    pm_s   = sa_s["p_market"].astype(float).values
    ry_s   = pd.to_numeric(sa_s["resolved_yes"], errors="coerce").fillna(-1)
    won_s  = (ry_s == (1 if eff_side == "yes" else 0)).values
    blk_s  = make_gate_mask(sa_s, gate_name).values
    bins_s = np.floor(pm_s * 10).astype(int)

    pm_te  = pm_s[test_idx]
    won_te = won_s[test_idx]
    blk_te = blk_s[test_idx]

    base_oos, _  = pnl_vec(pm_te, won_te)
    gate_oos, nb = pnl_vec(pm_te, won_te, blk_te)
    oos_delta    = gate_oos - base_oos

    wr_blk_oos   = float(won_te[blk_te].mean()) if nb else float("nan")
    bkev_oos     = bkev_for_side(pm_te[blk_te], eff_side) if nb else float("nan")

    print(f"  WF split → {split_ts.date()}  |  OOS: {len(test_idx):,} rows, {nb} blocked")
    print(f"  OOS Δ={oos_delta:+,.2f}  blocked WR={wr_blk_oos:.1%}  bkev={bkev_oos:.1%}")

    rng2 = np.random.default_rng(99)
    perm_oos = []
    print(f"  WF perm ({n_perms}) …", end=" ", flush=True)
    for i in range(n_perms):
        wp = shuffle_test_only(won_s, test_mask, bins_s, rng2)
        g, _ = pnl_vec(pm_te, wp[test_idx], blk_te)
        u, _ = pnl_vec(pm_te, wp[test_idx])
        perm_oos.append(g - u)
        if (i + 1) % 100 == 0:
            print(i + 1, end=" ", flush=True)
    print("done.")

    poo = np.array(perm_oos)
    p_wf = float((poo >= oos_delta).mean())
    print(f"  WF    : OOS Δ=${oos_delta:+,.2f}  null p50=${np.percentile(poo,50):+,.2f}"
          f"  p={p_wf:.3f}  {'✓ SIGNIFICANT' if p_wf < 0.05 else 'not significant'}")


# ── Blocked-trades outcome-join test ─────────────────────────────────────────

def run_blocked_join_test(gate_name, sa, bt_eth, n_perms, train_frac):
    """
    Join actual blocked trades to scan archive on contract_ticker to get outcomes.
    Run:  (a) stats + binomial test  (b) walk-forward split by date
    """
    side = GATE_SIDE[gate_name]

    # Filter blocked trades for this gate
    blk_gate = bt_eth[bt_eth["gate_name"] == gate_name].copy()
    print(f"\n── {gate_name}  [blocked-trades join]")
    print(f"  Blocked trades: {len(blk_gate):,} total")

    if len(blk_gate) == 0:
        print("  No blocked trades found — skipping.")
        return

    # Build outcome lookup: contract_ticker → resolved_yes
    ticker_outcome = (sa.dropna(subset=["resolved_yes"])
                        .drop_duplicates("contract_ticker")
                        .set_index("contract_ticker")["resolved_yes"]
                        .astype(int))
    blk_gate = blk_gate[blk_gate["contract_ticker"].isin(ticker_outcome.index)].copy()
    blk_gate["resolved_yes"] = blk_gate["contract_ticker"].map(ticker_outcome)
    blk_gate = blk_gate.dropna(subset=["resolved_yes"]).copy()

    print(f"  Matched to archive: {len(blk_gate):,} rows")
    if len(blk_gate) < 5:
        print("  Too few matched rows — skipping.")
        return

    # Compute stats per side
    for s in (["yes", "no"] if side == "both" else [side]):
        sub = blk_gate[blk_gate["side"] == s] if "side" in blk_gate.columns else blk_gate
        if len(sub) == 0:
            continue
        pm_arr  = sub["p_market"].astype(float).values
        won_arr = (sub["resolved_yes"].astype(int) == (1 if s == "yes" else 0)).values
        wr      = float(won_arr.mean())
        bkev    = float(pm_arr.mean() if s == "yes" else (1 - pm_arr).mean())
        pnl     = float((np.where(won_arr, (1 - pm_arr), -pm_arr) * (1 - FEE) * FLAT).sum())
        n_wins  = int(won_arr.sum())

        # Binomial test: H0 = WR = bkev
        binom   = scipy_stats.binomtest(n_wins, len(sub), bkev,
                                        alternative="less" if wr < bkev else "greater")
        p_binom = binom.pvalue
        print(f"  [{s.upper()}] n={len(sub)}  WR={wr:.1%}  bkev={bkev:.1%}  "
              f"edge={wr-bkev:+.1%}  PnL=${pnl:+,.2f}  p_binomial={p_binom:.3f}"
              f"  {'✓ sig' if p_binom < 0.05 else ''}")

    # Walk-forward split
    if "logged_at" in blk_gate.columns and blk_gate["logged_at"].notna().any():
        blk_sorted = blk_gate.sort_values("logged_at")
        split_idx  = int(len(blk_sorted) * train_frac)
        if split_idx < 5 or split_idx >= len(blk_sorted) - 5:
            print("  WF: not enough rows for split.")
            return
        split_ts = blk_sorted["logged_at"].iloc[split_idx]
        oos_rows = blk_sorted.iloc[split_idx:]

        for s in (["yes", "no"] if side == "both" else [side]):
            sub_oos = oos_rows[oos_rows["side"] == s] if "side" in oos_rows.columns else oos_rows
            if len(sub_oos) < 5:
                continue
            pm_oos  = sub_oos["p_market"].astype(float).values
            won_oos = (sub_oos["resolved_yes"].astype(int) == (1 if s == "yes" else 0)).values
            wr_oos  = float(won_oos.mean())
            bkev_oos = float(pm_oos.mean() if s == "yes" else (1 - pm_oos).mean())
            n_w_oos  = int(won_oos.sum())
            pnl_oos  = float((np.where(won_oos, (1-pm_oos), -pm_oos) * (1-FEE) * FLAT).sum())
            binom_oos = scipy_stats.binomtest(n_w_oos, len(sub_oos), bkev_oos,
                                              alternative="less" if wr_oos < bkev_oos else "greater")
            print(f"  WF OOS [{s.upper()}] split→{split_ts.date()}  "
                  f"n={len(sub_oos)}  WR={wr_oos:.1%}  bkev={bkev_oos:.1%}  "
                  f"PnL=${pnl_oos:+,.2f}  p={binom_oos.pvalue:.3f}"
                  f"  {'✓ sig' if binom_oos.pvalue < 0.05 else ''}")
    else:
        print("  WF: logged_at not available in blocked trades.")


# ── Dispatch ──────────────────────────────────────────────────────────────────

ARCHIVE_GATES = [
    "eth_no_vwap_stretch2_gate",
    "eth_no_adx_gate",
    "eth_funding_vol_yes_gate",
    "eth_squeeze_gate",
]
JOIN_GATES = [
    "eth_vol_regime_gate",
]

if args.gate == "all":
    run_these_archive = ARCHIVE_GATES
    run_these_join    = JOIN_GATES
else:
    run_these_archive = [args.gate] if args.gate in ARCHIVE_GATES else []
    run_these_join    = [args.gate] if args.gate in JOIN_GATES    else []

print(f"\n{'='*70}")
print(f"  ETH GATE VALIDATION   n_perms={args.n_perms}   train_frac={args.train_frac:.0%}")
print(f"{'='*70}")

for g in run_these_archive:
    run_archive_test(g, sa, args.n_perms, args.train_frac)

for g in run_these_join:
    run_blocked_join_test(g, sa, bt_eth, args.n_perms, args.train_frac)

print(f"\n{'='*70}")
print("Done.")
