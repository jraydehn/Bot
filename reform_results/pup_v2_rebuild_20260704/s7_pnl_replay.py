#!/usr/bin/env python3
"""S7 — Approximate BTC 15m PnL replay: substitute the honest v3 hour-level
score for the leaky p_up_v2 as the drift source, grid over K, and compare
against (a) the actual runner selection and (b) an old-model replay under the
IDENTICAL simplified rule (isolates the model swap from the gate stack).

ASSUMPTIONS (stated per task):
  1. Selection rule = raw edge >= 0.04 on the better side, one bet per
     contract_ticker (first qualifying scan). The live runner's gate stack /
     Kelly sizing / per-expiry caps are NOT replayed.
  2. Flat $100 stake per bet (feedback_flat_bankroll_backtest); entry at
     p_market +/- half the logged spread; Kalshi fee 0.07*price*(1-price)/ct.
  3. z_strike reconstructed from logged spot/floor_strike/tau/realized_vol +
     implied vol blend (weight 0.35 realized) — the runner's "vol_multi"
     override is not logged, so reconstruction is approximate; accuracy is
     quantified against logged p_model_15m on rows where the pup_v2 path ran.
  4. Hour-level score at decision d = WF OOS prediction at bar T =
     floor(d,'h') - 1h (last completed 1h bar; forecasts the hour containing
     the contract window). Honest preds from wf_preds_FINAL.parquet.
  5. K_NO = 0.6 * K_YES (preserves live 0.30/0.50 ratio); a K_NO=K_YES
     variant is also reported.

Outputs: replay_grid.csv, replay_zones.csv + stdout report.
"""
import math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
MINS_PER_YEAR = 525600.0
W_REAL = 0.35
EDGE = 0.04
STAKE = 100.0

df = pd.read_csv(PROJ / "results" / "paper_trades_btc15m.csv", low_memory=False)
df = df[df["asset"] == "BTC"].copy()
for c in ("spot", "floor_strike", "tau_minutes", "spread", "p_market",
          "p_model_15m", "raw_edge", "realized_vol_annual", "p_up_v2_btc",
          "would_pnl", "bet_amount"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["resolved"] = df["resolved_yes"].astype(str).str.lower().map(
    {"1.0": 1, "1": 1, "true": 1, "0.0": 0, "0": 0, "false": 0})
df["dt"] = pd.to_datetime(df["decision_time"], utc=True, format="mixed")
df = df[df["resolved"].notna() & df["p_market"].between(0.03, 0.97)
        & (df["tau_minutes"] > 0.5) & (df["spot"] > 0) & (df["floor_strike"] > 0)].copy()
df["spread"] = df["spread"].fillna(0.01).clip(0, 0.10)

# honest hour-level score at bar floor(d)-1h
wf = pd.read_parquet(HERE / "wf_preds_FINAL.parquet")["p"]
df["bar"] = df["dt"].dt.floor("h") - pd.Timedelta(hours=1)
df["p_hat"] = wf.reindex(pd.DatetimeIndex(df["bar"], tz="UTC")).values
print(f"rows: {len(df):,}  with honest p_hat: {df['p_hat'].notna().sum():,}  "
      f"with old p_up_v2: {df['p_up_v2_btc'].notna().sum():,}")

# ── z_strike reconstruction ───────────────────────────────────────────────
def z_strike_row(r):
    vol_real = r.realized_vol_annual / math.sqrt(MINS_PER_YEAR) \
        if r.realized_vol_annual and r.realized_vol_annual > 0 else np.nan
    ld = math.log(r.floor_strike / r.spot)
    zi = norm.ppf(1.0 - r.p_market)
    vol_imp = ld / (zi * math.sqrt(r.tau_minutes)) if zi != 0 and ld * zi > 0 else np.nan
    if vol_imp and vol_imp > 0 and not math.isnan(vol_imp):
        vol_eff = W_REAL * vol_real + (1 - W_REAL) * vol_imp if vol_real == vol_real else vol_imp
    else:
        vol_eff = vol_real
    if not (vol_eff and vol_eff > 0):
        return np.nan
    return ld / max(vol_eff * math.sqrt(r.tau_minutes), 1e-6)

df["z_strike"] = [z_strike_row(r) for r in df.itertuples()]
df = df[df["z_strike"].notna()].copy()

def p_sides(p_hat, K_yes, K_no, tau, zs):
    z = norm.ppf(np.clip(p_hat, 0.02, 0.98))
    tsc = np.sqrt(np.minimum(tau, 60.0) / 60.0)
    py = np.clip(norm.cdf(z * K_yes * tsc - zs), 0.03, 0.97)
    pn = np.clip(1.0 - norm.cdf(z * K_no * tsc - zs), 0.03, 0.97)
    return py, pn

# reconstruction sanity: old model on taken YES/NO rows
chk = df[(df["decision"] == "trade") & df["p_up_v2_btc"].notna() & df["p_model_15m"].notna()]
if len(chk):
    py, pn = p_sides(chk["p_up_v2_btc"].values, 0.50, 0.30,
                     chk["tau_minutes"].values, chk["z_strike"].values)
    rec = np.where(chk["side"] == "yes", py, pn)
    err = np.abs(rec - chk["p_model_15m"].values)
    print(f"reconstruction check (taken rows n={len(chk)}): median |err|={np.median(err):.4f} "
          f"p90={np.percentile(err, 90):.4f}")

def replay(sub, p_col, K_yes, K_no, zone=None):
    """One bet per ticker (first qualifying scan); returns dict of stats."""
    py, pn = p_sides(sub[p_col].values, K_yes, K_no,
                     sub["tau_minutes"].values, sub["z_strike"].values)
    ask_y = sub["p_market"].values + sub["spread"].values / 2
    ask_n = 1 - sub["p_market"].values + sub["spread"].values / 2
    e_y, e_n = py - ask_y, pn - ask_n
    side_no = e_n > e_y
    edge = np.where(side_no, e_n, e_y)
    price = np.clip(np.where(side_no, ask_n, ask_y), 0.03, 0.99)
    fire = edge >= EDGE
    if zone is not None:
        lo, hi = zone
        fire &= np.where(side_no, sub[p_col].values <= lo, sub[p_col].values >= hi)
    win = np.where(side_no, 1 - sub["resolved"].values, sub["resolved"].values)
    ct = STAKE / price
    fee = 0.07 * price * (1 - price) * ct
    pnl = np.where(win == 1, ct * (1 - price) - fee, -STAKE - fee)
    t = sub.loc[fire, ["contract_ticker", "dt"]].copy()
    t["pnl"] = pnl[fire]; t["win"] = win[fire]; t["no"] = side_no[fire]
    t = t.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
    return {"n": len(t), "n_no": int(t["no"].sum()), "wr": t["win"].mean() if len(t) else np.nan,
            "pnl": t["pnl"].sum(), "pnl_per_bet": t["pnl"].mean() if len(t) else np.nan}

# baseline (a): actual runner selection, standardized to $100 stake
act = df[df["decision"] == "trade"].copy()
ask = np.where(act["side"] == "no", 1 - act["p_market"] + act["spread"] / 2,
               act["p_market"] + act["spread"] / 2)
price = np.clip(ask, 0.03, 0.99)
win = np.where(act["side"] == "no", 1 - act["resolved"], act["resolved"])
ct = STAKE / price
fee = 0.07 * price * (1 - price) * ct
pnl = np.where(win == 1, ct * (1 - price) - fee, -STAKE - fee)
a = act.assign(pnl=pnl, win=win).sort_values("dt").drop_duplicates("contract_ticker", keep="first")
print(f"\n[actual runner trades, $100 std] n={len(a)}  WR={a['win'].mean():.3f}  "
      f"PnL=${a['pnl'].sum():,.0f}  per-bet=${a['pnl'].mean():.2f}")

# baseline (b): old leaky model under the simplified rule
old = df[df["p_up_v2_btc"].notna()].copy()
r = replay(old, "p_up_v2_btc", 0.50, 0.30)
print(f"[old model replay K=0.50/0.30, same rule] {r}")

# grid: new model
new = df[df["p_hat"].notna()].copy()
rows = []
for K in (0.5, 1.0, 1.5, 2.0, 3.0):
    for ratio, tag in ((0.6, "Kno=0.6K"), (1.0, "Kno=K")):
        r = replay(new, "p_hat", K, K * ratio)
        rows.append({"K_yes": K, "K_no": round(K * ratio, 2), "variant": tag, **r})
        print(f"[new v3 K={K}/{K*ratio:.2f}] {r}")
grid = pd.DataFrame(rows)
grid.to_csv(HERE / "replay_grid.csv", index=False)

# fire zones on the honest output for the best K
best = grid.sort_values("pnl", ascending=False).iloc[0]
Kb, Knb = best["K_yes"], best["K_no"]
print(f"\nbest K: {Kb}/{Knb}  — zone sweep (NO fires p_hat<=lo, YES fires p_hat>=hi):")
zrows = []
for lo, hi in [(1.0, 0.0), (0.50, 0.50), (0.48, 0.52), (0.46, 0.54),
               (0.45, 0.55), (0.44, 0.56), (0.43, 0.57), (0.42, 0.58)]:
    r = replay(new, "p_hat", Kb, Knb, zone=(lo, hi))
    zrows.append({"zone_lo": lo, "zone_hi": hi, **r})
    print(f"  zone <= {lo} / >= {hi}: {r}")
pd.DataFrame(zrows).to_csv(HERE / "replay_zones.csv", index=False)
print("\nS7 DONE")
