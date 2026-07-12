"""
S2 -- two-target loss autopsy for BTC hourly:
  (a) YES+AGREE: small but persistently negative every week it has volume
      (06-22/28: -$90, 06-29/07-05: -$60, 07-06/07-12: -$42). Structural,
      not a single bad week -- compare winners vs losers across the full
      06-17->now window.
  (b) NO+AGREE, most recent week only (07-06->07-12): this is 85% of all
      volume and just flipped from 3 straight good weeks to -$269 this
      week. Only 1 week of data so treat cautiously, but worth checking
      for an early discriminator before deciding whether to act.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/paper_trades.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[pd.to_numeric(df["bet_amount"], errors="coerce") > 0].dropna(subset=["would_pnl", "resolved_yes", "side", "p_market"])
t["p_market"] = pd.to_numeric(t["p_market"], errors="coerce")
t["won"] = np.where(t["side"] == "yes", t["resolved_yes"] == True, t["resolved_yes"] == False)
t["cost"] = np.where(t["side"] == "yes", t["p_market"], 1 - t["p_market"])
t["contrarian"] = ((t.side == "yes") & (t.p_market < 0.5)) | ((t.side == "no") & (t.p_market > 0.5))

DIAG_COLS = [c for c in [
    "p_market", "offset_pct", "tau_minutes", "composite_p_up", "composite_trend", "composite_rev",
    "no_score", "confirmation_score", "obi_score", "vpin_score", "vol_score", "z_score",
    "ls_long_pct", "oi_chg_pct", "liq_score", "liq_bias", "funding_bias",
    "adx_1h", "rvol_1h", "squeeze_1h", "stoch_k", "stoch_k_4h", "ema_stack_bias", "ema_stretch_score",
    "vwap_distance_pct", "vwap_stretch_score", "chg_30m", "chg_10m", "chg_5m", "chg_1h", "chg_2h", "chg_3h",
    "bp_5m", "bp_1h", "direction_strength", "raw_edge", "net_edge", "kelly_fraction",
    "macro_regime_bull", "macro_regime_sdwy", "macro_regime_bear",
    "hmm_vol_state", "hmm_r1_prob", "markov_regime_daily", "markov_regime_7state",
    "vwap_1h_state", "p_up_v3", "pup_v3_hmm_state", "hmm_zdrift_state",
    "p_gbdt", "ob_imbalance", "ob_ask_frac", "hurst_exponent", "ou_z_score",
] if c in t.columns]


def autopsy(sub, label):
    print(f"\n{'='*70}\n  {label}  (n={len(sub)})\n{'='*70}")
    print(f"  WR={sub['won'].mean():.1%}  BE={sub['cost'].mean():.1%}  edge={sub['won'].mean()-sub['cost'].mean():+.1%}  "
          f"total_pnl=${sub['would_pnl'].sum():.2f}")
    w = sub[sub["won"]]
    l = sub[~sub["won"]]
    print(f"  winners: n={len(w)} ${w['would_pnl'].sum():.2f}   losers: n={len(l)} ${l['would_pnl'].sum():.2f}")
    print(f"\n  {'column':<22s} {'winners':>10s} {'losers':>10s} {'diff':>10s}")
    for c in DIAG_COLS:
        wc = pd.to_numeric(w[c], errors="coerce")
        lc = pd.to_numeric(l[c], errors="coerce")
        if wc.notna().sum() < 3 or lc.notna().sum() < 3:
            continue
        d = wc.mean() - lc.mean()
        flag = " <<<" if abs(d) > 0.5 * (abs(wc.mean()) + abs(lc.mean()) + 1e-9) else ""
        print(f"  {c:<22s} {wc.mean():10.4f} {lc.mean():10.4f} {d:+10.4f}{flag}")


ya = t[(t.side == "yes") & (~t.contrarian)]
autopsy(ya, "YES + AGREE, full window (06-17 -> now)")

recent_na = t[(t.side == "no") & (~t.contrarian) & (t.decision_time >= pd.Timestamp("2026-07-06", tz="UTC"))]
autopsy(recent_na, "NO + AGREE, most recent week only (07-06 -> now)")

print(f"\n=== YES+AGREE loss detail ===")
show = ["decision_time", "contract_ticker", "p_market", "would_pnl", "composite_rev", "composite_trend", "no_score", "obi_score", "net_edge"]
show = [c for c in show if c in ya.columns]
print(ya[~ya.won][show].to_string())

print("\nDONE_S2")
