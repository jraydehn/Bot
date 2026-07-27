"""
review_deployed_mechanisms.py
------------------------------
Forward-validation review harness for the four mechanisms deployed 2026-07-26/27,
with PRE-REGISTERED acceptance criteria (written before any forward data existed,
so the goalposts cannot move). Run any time: prints per-mechanism evidence,
verdict, and ETA to sufficiency.

    python3 review_deployed_mechanisms.py

Mechanisms & criteria (registered 2026-07-27, before forward data):

1. SOL z-expansion (p' = Phi(Phi^-1(p)*1.8); rows with p_model_pre_expand).
   Affected selections = rows where expansion changed the take/skip decision
   (added: raw edge < 0.04 <= expanded edge; dropped: raw >= 0.04 > expanded).
   PASS  : n_affected >= 100 AND net_delta > 0 AND added-trade edge_pp > 0
   FAIL  : n_affected >= 150 AND net_delta < 0
   else INSUFFICIENT.

2. BTC KC reversion correction (rows with kc_rev_shift_5m != 0).
   Affected selections = rows where the shift changed take/skip.
   PASS  : n_affected >= 75 AND net_delta > 0
   FAIL  : n_affected >= 150 AND net_delta < 0
   else INSUFFICIENT.
   Secondary (direction sanity, all shifted rows): corr(shift, resolved - p_raw) > 0.

3. Losing-streak boosts/dampener (reconstructed firing conditions on post-deploy
   trades): SOL NO boost (vol_chg_trend12_15m <= -0.0368), SOL YES boost
   (wick_upper_trend12_15m > 0.00833), ETH NO boost (bb_pct_trend3_1h <= -0.0178)
   -- fired trades should be BETTER than same-period non-fired trades;
   BTC YES dampener (kalman_velocity_trend12_5m > 4.72e-05) -- fired should be WORSE.
   PASS  : n_fired >= 25 AND edge diff has the predicted sign
   FAIL  : n_fired >= 25 AND edge diff has the OPPOSITE sign with |diff| > 5pp
   else INSUFFICIENT.

All PnL is flat-stake $50 counterfactual (feedback_flat_bankroll_backtest); this
harness judges SELECTION/CONDITION quality, not Kelly sizing.
"""
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
RESULTS = Path(__file__).parent / "results"
STAKE = 50.0
EDGE_MIN = 0.04
DEPLOY_BOOSTS = pd.Timestamp("2026-07-26 07:05", tz="UTC")   # streak boosts live (all twins restarted)

def load(asset):
    df = pd.read_csv(RESULTS / f"paper_trades_{asset}15m.csv", low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    for c in ("p_market", "p_model_15m", "p_model_pre_expand", "kc_rev_shift_5m",
              "vol_chg_trend12_15m", "wick_upper_trend12_15m", "bb_pct_trend3_1h",
              "kalman_velocity_trend12_5m", "resolved_yes", "would_win"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("logged_at").reset_index(drop=True)

def cf_pnl(row_p_market, side, resolved_yes):
    is_yes = np.asarray(side) == "yes"
    win = np.where(is_yes, resolved_yes, 1 - resolved_yes)
    be = np.where(is_yes, row_p_market, 1 - row_p_market)
    return np.where(win == 1, (1 - be) / be * STAKE, -STAKE), win, be

def selection(p, pm):
    ey, en = p - pm, pm - p
    side = np.where(ey >= en, "yes", "no")
    return side, np.maximum(ey, en)

def selection_delta(df, p_raw_col, p_final_col, label):
    """Added/dropped/net analysis on RESOLVED rows where the correction changed take/skip."""
    d = df.dropna(subset=[p_raw_col, p_final_col, "p_market", "resolved_yes"]).copy()
    if d.empty:
        print(f"  {label}: no rows yet");  return None
    s_raw, e_raw = selection(d[p_raw_col].values, d["p_market"].values)
    s_fin, e_fin = selection(d[p_final_col].values, d["p_market"].values)
    take_raw, take_fin = e_raw >= EDGE_MIN, e_fin >= EDGE_MIN
    added   = d[~take_raw & take_fin].copy();  added["side"]   = s_fin[~take_raw & take_fin]
    dropped = d[take_raw & ~take_fin].copy();  dropped["side"] = s_raw[take_raw & ~take_fin]
    for sub in (added, dropped):
        if len(sub):
            pnl, win, be = cf_pnl(sub["p_market"].values, sub["side"].values, sub["resolved_yes"].values)
            sub["pnl"], sub["win"], sub["be"] = pnl, win, be
    n_aff = len(added) + len(dropped)
    add_pnl = added["pnl"].sum() if len(added) else 0.0
    drop_pnl = dropped["pnl"].sum() if len(dropped) else 0.0
    net = add_pnl - drop_pnl          # dropped pnl is foregone: avoiding losers is +
    add_edge = ((added["win"].mean() - added["be"].mean()) * 100) if len(added) else np.nan
    print(f"  {label}: affected={n_aff} (added={len(added)}, dropped={len(dropped)})  "
          f"added_pnl=${add_pnl:+.2f} (edge {add_edge:+.1f}pp)  dropped_foregone=${drop_pnl:+.2f}  NET=${net:+.2f}")
    return {"n_aff": n_aff, "net": net, "add_edge": add_edge}

def verdict_delta(r, n_pass, n_fail, extra_ok=True):
    if r is None or r["n_aff"] < min(n_pass, n_fail):
        return "INSUFFICIENT"
    if r["n_aff"] >= n_pass and r["net"] > 0 and extra_ok:
        return "PASS"
    if r["n_aff"] >= n_fail and r["net"] < 0:
        return "FAIL"
    return "INSUFFICIENT"

print("=" * 90)
print("  DEPLOYED-MECHANISM FORWARD REVIEW  (criteria pre-registered 2026-07-27)")
print("=" * 90)

# ---- 1. SOL z-expansion --------------------------------------------------
print("\n[1] SOL z-expansion (k=1.8)")
sol = load("sol")
zx = sol[sol["p_model_pre_expand"].notna()]
r1 = selection_delta(zx, "p_model_pre_expand", "p_model_15m", "expansion")
v1 = verdict_delta(r1, 100, 150, extra_ok=(r1 is None or not np.isnan(r1["add_edge"]) and r1["add_edge"] > 0))
if r1: print(f"  resolved affected-rows accumulating since 2026-07-27 ~07:20 UTC")
print(f"  VERDICT: {v1}")

# ---- 2. BTC KC correction ------------------------------------------------
print("\n[2] BTC KC reversion correction")
btc = load("btc")
kc = btc[btc["kc_rev_shift_5m"].notna() & (btc["kc_rev_shift_5m"] != 0)].copy()
if len(kc):
    kc["p_raw"] = np.clip(kc["p_model_15m"] - kc["kc_rev_shift_5m"], 0.01, 0.99)
    r2 = selection_delta(kc, "p_raw", "p_model_15m", "kc_shift")
    res = kc.dropna(subset=["resolved_yes"])
    if len(res) >= 30:
        c = np.corrcoef(res["kc_rev_shift_5m"], res["resolved_yes"] - (res["p_model_15m"] - res["kc_rev_shift_5m"]))[0, 1]
        print(f"  direction sanity: corr(shift, resolved - p_raw) = {c:+.3f} on n={len(res)} (want > 0)")
else:
    print("  no shifted rows yet"); r2 = None
v2 = verdict_delta(r2, 75, 150)
print(f"  VERDICT: {v2}")

# ---- 3. streak boosts / dampener ------------------------------------------
print("\n[3] losing-streak boosts / dampener (fired vs same-period non-fired, resolved trades)")
def streak_series(df):
    tr = df[(df["decision"] == "trade") & df["would_win"].notna()].sort_values("logged_at")
    cur, out = 0, {}
    for idx, w in zip(tr.index, tr["would_win"].values):
        out[idx] = cur
        cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
    return pd.Series(out)

MECHS = [
    ("SOL NO boost",   "sol", "no",  "vol_chg_trend12_15m",       lambda v: v <= -0.0368, "better"),
    ("SOL YES boost",  "sol", "yes", "wick_upper_trend12_15m",    lambda v: v > 0.00833,  "better"),
    ("ETH NO boost",   "eth", "no",  "bb_pct_trend3_1h",          lambda v: v <= -0.0178, "better"),
    ("BTC YES dampener","btc","yes", "kalman_velocity_trend12_5m",lambda v: v > 4.72e-05, "worse"),
]
frames = {"sol": sol, "btc": btc, "eth": load("eth")}
for name, asset, side, sigcol, cond, want in MECHS:
    df = frames[asset]
    post = df[df["logged_at"] >= DEPLOY_BOOSTS]
    tr = post[(post["decision"] == "trade") & (post["side"] == side) & post["would_win"].notna()].copy()
    if sigcol not in tr.columns or tr.empty:
        print(f"  {name}: no data yet — INSUFFICIENT"); continue
    st = streak_series(df).reindex(tr.index)
    tr["in_ls"] = st <= -2
    tr["fired"] = tr["in_ls"] & tr[sigcol].apply(lambda v: bool(cond(v)) if pd.notna(v) else False)
    fired, rest = tr[tr["fired"]], tr[~tr["fired"]]
    if len(fired) == 0 or len(rest) < 10:
        print(f"  {name}: fired={len(fired)} non-fired={len(rest)} — INSUFFICIENT"); continue
    def edge(g):
        be = np.where(g["side"] == "yes", g["p_market"], 1 - g["p_market"])
        return (g["would_win"].mean() - be.mean()) * 100
    diff = edge(fired) - edge(rest)
    ok_sign = diff > 0 if want == "better" else diff < 0
    if len(fired) >= 25:
        v = "PASS" if ok_sign else ("FAIL" if abs(diff) > 5 else "INSUFFICIENT")
    else:
        v = "INSUFFICIENT"
    print(f"  {name}: fired={len(fired)} edge={edge(fired):+.1f}pp vs non-fired({len(rest)}) {edge(rest):+.1f}pp  "
          f"diff={diff:+.1f}pp (want {want})  VERDICT: {v}")

print("\nnote: INSUFFICIENT is the expected verdict until enough forward data accumulates;")
print("re-run daily. SOL z-expansion reaches criteria fastest (~1 week).")
