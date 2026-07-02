"""
check_pending.py — single command status check for all shadow-logged signals
and pending gate proposals.

Run: python3 check_pending.py
"""

import math, os, sys, warnings, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE    = Path(__file__).parent
RESULTS = BASE / "results"

def _load(path):
    try: return pd.read_csv(path, low_memory=False)
    except: return pd.DataFrame()

def _to_dt(df, col='logged_at'):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='mixed')
    return df[df[col].notna()]

def _n_resolved(df, col='resolved_yes'):
    if df.empty or col not in df.columns: return 0
    return int(df[col].notna().sum())

def _n_resolved_since(df, since, date_col='logged_at', res_col='resolved_yes'):
    try:
        df = _to_dt(df, date_col)
        sub = df[df[date_col] >= since]
        return _n_resolved(sub, res_col)
    except: return 0

def _n_state(df, col, val, date_col='logged_at', since='2026-05-25'):
    try:
        df = _to_dt(df, date_col)
        sub = df[df[date_col] >= since]
        return int((sub[col] == val).sum()) if col in sub.columns else 0
    except: return 0

def _bar(n, threshold):
    frac  = min(n / max(threshold, 1), 1.0)
    filled = int(frac * 20)
    bar   = "█" * filled + "░" * (20 - filled)
    if n >= threshold:          tag = "✓ READY"
    elif n >= threshold * 0.6:  tag = "~ SOON (~{} left)".format(threshold - n)
    else:                       tag = "· {} left".format(threshold - n)
    return f"[{bar}] {n:>5}/{threshold:<5} {tag}"

def _model_mtime(name):
    p = BASE / "models" / name
    if not p.exists(): return None
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).date().isoformat()

NOW = datetime.datetime.now(datetime.timezone.utc)
print(f"\n{'═'*70}")
print(f"  PENDING SIGNAL TRACKER   {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"{'═'*70}\n")

# ── Load CSVs once ─────────────────────────────────────────────────────────
sa15   = _load(RESULTS / "btc_scan_archive_15m.csv")
sa1h   = _load(RESULTS / "btc_scan_archive.csv")
pt1h   = _load(RESULTS / "paper_trades.csv")
pt15m  = _load(RESULTS / "paper_trades_btc15m.csv")
pt_eth = _load(RESULTS / "paper_trades_eth.csv")
pt_sol = _load(RESULTS / "paper_trades_sol.csv")

# ══════════════════════════════════════════════════════════════════════════
print("── SHADOW-LOGGED SIGNALS ─────────────────────────────────────────────\n")

# 1. HMM vol state — 15m BTC scan archive
n_hmm15 = _n_state(sa15, 'hmm_vol_state', 1.0, since='2026-05-25')
print("1. HMM vol state — 15m BTC scan archive (live Jun-02)")
print("   Hypothesis : R1 (high-vol) NO bets lose; OTM_NO zone worst")
print("   Gate target: block R1 when pm∈[0.30,0.45) — needs n≥100 R1 obs")
print(f"   R1 obs since May-25 : {_bar(n_hmm15, 100)}")
print()

# 2. HMM vol state — 1h BTC paper trades
pt1h_btc = pt1h[pt1h['asset']=='BTC'] if 'asset' in pt1h.columns else pt1h
n_hmm1h = _n_state(pt1h_btc, 'hmm_vol_state', 1, since='2026-05-25')
print("2. HMM vol state — 1h BTC paper trades (live Jun-03)")
print("   Hypothesis : R1 loses -$3.65/trade (current model era May-Jun)")
print("   Gate target: block R1 + pm<0.45 — needs n≥100 R1 obs")
print(f"   R1 obs since May-25 : {_bar(n_hmm1h, 100)}")
print()

# 3. p_gbdt — 15m BTC LGBM shadow
lgbm_15m_mtime = _model_mtime("lgbm_15m_btc.pkl") or "unknown"
n_sa15_oos = _n_resolved_since(sa15, lgbm_15m_mtime) if lgbm_15m_mtime != "unknown" else _n_resolved(sa15)
n_pgbdt    = int(sa15['p_gbdt'].notna().sum()) if 'p_gbdt' in sa15.columns else 0
print("3. p_gbdt — 15m BTC LGBM shadow model")
print(f"   LGBM trained       : {lgbm_15m_mtime}")
print(f"   Hypothesis         : LGBM may replace p_up_v2 as 15m primary model")
print(f"   Condition          : retrain on OOS scan archive rows (2,000+)")
print(f"   OOS resolved rows  : {_bar(n_sa15_oos, 2000)}")
print()

# 4. BTC 1h p_up_v2 — disabled May-19
lgbm_1h_mtime = _model_mtime("reform_results/btc_lgbm.pkl") or "2026-05-19"
n_sa1h_oos = _n_resolved_since(sa1h, '2026-05-19')
print("4. BTC p_up_v2 (1h LGBM directional) — disabled May-19")
print("   Reason    : systematically bearish in uptrend (1W-8L, -$172)")
print("   Condition : retrain on OOS scan archive rows (2,000+) then re-enable")
print(f"   OOS resolved rows since May-19: {_bar(n_sa1h_oos, 2000)}")
print()

# 5. BTC 3-factor drift reform
print("5. BTC 3-factor drift (Stoch_4h × RVOL_inv × |EMA20z_4h|)")
print("   IC_yes=+0.056 IC_no=+0.102 vs standalone (+0.014/+0.058)")
print("   Blocked by : p_up_v2 disabled — same retrain dependency as #4")
print("   Status     : · blocked on #4 resolution")
print()

# ══════════════════════════════════════════════════════════════════════════
print("── PENDING GATE PROPOSALS ────────────────────────────────────────────\n")

# 6. ETH stoch-cross NO gate
n_eth = _n_resolved_since(pt_eth, '2026-05-31')
print("6. ETH stoch-cross NO gate")
print("   Proposal  : block stoch_bias=+1 + stoch_k<20 + pm[0.20,0.40) → +$286 sim")
print("   Condition : live data post May-31 gate changes (n≥100)")
print(f"   Resolved since May-31: {_bar(n_eth, 100)}")
print()

# 7. ETH pending gates A+B combo
n_eth_ab = _n_resolved_since(pt_eth, '2026-05-31')
print("7. ETH gates A (ema neutral) + B (stoch+ema bearish) combo")
print("   Sim result: +$559 combined — ready after same data as #6")
print(f"   Resolved since May-31: same as above ({n_eth_ab})")
print()

# 8. SOL p_market < 0.55 block
print("8. SOL p_market < 0.55 block")
print("   Status    : · simulation not yet run — do this first before collecting data")
print()

# 9. BTC 15m 4-gate changes (live Jun-01)
n_sa15_jun = _n_resolved_since(sa15, '2026-06-01')
print("9. BTC 15m 4-gate block (SW/SW, Bear/Bull, pm>0.85, midpm+liq)")
print("   Sim result: +$558 saved — validate each gate with n≥150 blocked")
print(f"   Scan archive rows since Jun-01: {n_sa15_jun} (check log for per-gate fire counts)")
print()

# 10. 7-state HMM regime model
print("10. 7-state HMM regime model (hmm_7state_btc.pkl)")
print("    Condition : wire in after scan archive spans a clear regime transition")
print("    Status    : · waiting — current market hasn't had a clear regime flip")
print()

# ══════════════════════════════════════════════════════════════════════════
print("── SUMMARY ───────────────────────────────────────────────────────────\n")

ready = []
if n_sa15_oos >= 2000:   ready.append("#3 15m LGBM retrain")
if n_sa1h_oos >= 2000:   ready.append("#4 p_up_v2 retrain+re-enable")
if n_hmm15    >= 100:    ready.append("#1 HMM R1 gate (15m)")
if n_hmm1h    >= 100:    ready.append("#2 HMM R1 gate (1h)")
if n_eth      >= 100:    ready.append("#6/#7 ETH gates live test")

if ready:
    print("  ✓ READY: " + " | ".join(ready))
else:
    print("  Nothing ready — all signals still accumulating.")

all_pending = [
    (n_hmm15,    100,   "#1 HMM R1 gate (15m)"),
    (n_hmm1h,    100,   "#2 HMM R1 gate (1h)"),
    (n_sa15_oos, 2000,  "#3 15m LGBM retrain"),
    (n_sa1h_oos, 2000,  "#4 p_up_v2 retrain"),
    (n_eth,      100,   "#6 ETH stoch-cross gate"),
]
gaps = sorted([(t-n, lbl) for n,t,lbl in all_pending if n < t])
if gaps:
    g, lbl = gaps[0]
    print(f"  → Nearest threshold: {lbl}  (needs {g} more obs)")
print()
