"""
S13 -- direct loss autopsy. The obi_score finding is an opportunistic add,
not a fix for the active bleed. Pull the ACTUAL TAKEN trades (not scan
candidates) for the degradation window (06-29 -> now, n=73, PnL=-$923.57,
WR=65.75% but well below whatever breakeven applies given the loss) and
compare winners vs losers directly across every available signal column,
looking for what a BLOCKING gate could have caught. This is diagnosis of
what's actually bleeding money, not a search for a new positive edge.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

d1 = pd.read_csv("results/paper_trades_sol_archive_20260707_2013_pre_contrarian_ls_gate.csv", low_memory=False)
d2 = pd.read_csv("results/paper_trades_sol.csv", low_memory=False)
common = [c for c in d1.columns if c in d2.columns]
d1["decision_time"] = pd.to_datetime(d1["decision_time"], utc=True, errors="coerce", format="mixed")
d2["decision_time"] = pd.to_datetime(d2["decision_time"], utc=True, errors="coerce", format="mixed")
combo = pd.concat([d1[common], d2[common]], ignore_index=True)
combo = combo[pd.to_numeric(combo["bet_amount"], errors="coerce") > 0].dropna(subset=["would_pnl", "resolved_yes", "side"])
recent = combo[combo["decision_time"] >= pd.Timestamp("2026-06-29", tz="UTC")].copy()
recent["won"] = np.where(recent["side"] == "yes", recent["resolved_yes"] == True, recent["resolved_yes"] == False)
recent["p_market"] = pd.to_numeric(recent["p_market"], errors="coerce")
recent["cost"] = np.where(recent["side"] == "yes", recent["p_market"], 1 - recent["p_market"])
n = len(recent)
print(f"recent taken trades (06-29 -> now): n={n}  WR={recent['won'].mean():.1%}  BE={recent['cost'].mean():.1%}  "
      f"total_pnl=${recent['would_pnl'].sum():.2f}")
print(f"by side: \n{recent.groupby('side').agg(n=('won','size'), wr=('won','mean'), be=('cost','mean'), pnl=('would_pnl','sum'))}")

losers = recent[~recent["won"]]
winners = recent[recent["won"]]
print(f"\nlosers: n={len(losers)}  total_lost=${losers['would_pnl'].sum():.2f}")
print(f"winners: n={len(winners)}  total_won=${winners['would_pnl'].sum():.2f}")

# candidate diagnostic columns -- signal state at decision time
DIAG_COLS = [c for c in [
    "p_market", "offset_pct", "tau_minutes", "composite_p_up", "composite_trend", "composite_rev",
    "no_score", "confirmation_score", "obi_score", "vpin_score", "vol_score",
    "ls_long_pct", "oi_chg_pct", "liq_score", "liq_bias", "funding_bias",
    "adx_1h", "rvol_1h", "squeeze_1h", "stoch_k", "ema_stack_bias", "ema_stretch_score",
    "vwap_distance_pct", "chg_30m", "chg_10m", "chg_5m", "bp_5m",
    "macro_regime_bull", "macro_regime_sdwy", "macro_regime_bear",
    "p_up_v2", "pup15m", "kelly_fraction", "bet_amount",
] if c in recent.columns]

print(f"\n=== winners vs losers, mean of each diagnostic column ===")
print(f"{'column':<22s} {'winners':>10s} {'losers':>10s} {'diff':>10s}")
for c in DIAG_COLS:
    w = pd.to_numeric(winners[c], errors="coerce")
    l = pd.to_numeric(losers[c], errors="coerce")
    if w.notna().sum() < 5 or l.notna().sum() < 5:
        continue
    print(f"{c:<22s} {w.mean():10.4f} {l.mean():10.4f} {w.mean()-l.mean():+10.4f}")

print(f"\n=== per-loss detail (all {len(losers)} losing trades) ===")
show_cols = ["decision_time", "contract_ticker", "side", "p_market", "bet_amount", "would_pnl"]
show_cols += [c for c in ["no_score", "obi_score", "composite_rev", "composite_trend", "ls_long_pct", "offset_pct"] if c in losers.columns]
print(losers[show_cols].to_string())

print("\nDONE_S13")
