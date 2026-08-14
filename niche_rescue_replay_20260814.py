"""BTC hourly niche v1 — PRE-REGISTERED rescue books, frozen 2026-08-14.

Found by the full comprehensive-rescue sweep (1,974 tests over the rule's
rejected populations, forward window 07-28+). Both are MINED — this script
is the referee: rerun at the reviews; only FORWARD rows (dt >= FREEZE) are
decision evidence. No live runner needed: the scan archive accrues and the
model pkl is frozen, so the books replay deterministically.

  RESCUE-A: YES, pm 0.35-0.65, fee-adj edge in [0.03, 0.06), adx_1h >= 22
            (discovery: n=52 WR 62% +$1,747, wks +819/-26/+1,071)
  RESCUE-B: NO,  pm 0.20-0.80, NO fee-adj edge >= 0.06, liq_bias >= 1.0
            (discovery: n=315 WR 62% +$7,875, ALL wks +, topday 19%,
             positive in every pm band; mechanism: long-liquidation-
             dominant tape -> downside pressure -> NO)

Evaluate ~08-25: forward-only n/WR/net + bootstraps; deploy bar = the
usual (weeks consistency, concentration <50%, p<=0.05 forward).
"""
import pandas as pd, numpy as np, pickle, warnings, sys
warnings.filterwarnings("ignore")
FREEZE = pd.Timestamp("2026-08-14 05:00", tz="UTC")
with open("models/btc_hourly_lgbm_niche_20260728.pkl", "rb") as f:
    art = pickle.load(f)
model, FEATS = art["model"], art["features"]
df = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
df = df.dropna(subset=["dt"])
df = df[df["dt"] >= (FREEZE if "--forward" in sys.argv
                     else pd.Timestamp("2026-07-28 12:00", tz="UTC"))].copy()
for c in set(FEATS + ["p_market", "tau_minutes", "strike", "spot",
                      "resolved_yes", "adx_1h", "liq_bias"]) - {"z_moneyness"}:
    df[c] = pd.to_numeric(df.get(c), errors="coerce") if c in df.columns else np.nan
with np.errstate(divide="ignore", invalid="ignore"):
    df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(df["tau_minutes"].clip(lower=1))
df = df.dropna(subset=["p_market"])
df["p"] = model.predict_proba(df[FEATS])[:, 1]
fee = 0.07 * df["p_market"] * (1 - df["p_market"])
df["edge_yes"] = df["p"] - df["p_market"] - fee
df["edge_no"] = df["p_market"] - df["p"] - fee
res = df[df["resolved_yes"].notna()]
# [2026-08-14 addendum, same day] RESCUE-C: the mean-reversion rescue the
# original sweep COULD NOT SEE — the archive logs no MR columns (stoch/rsi/
# dprice/kalman all absent; Phase-2 coverage failure), so they were
# reconstructed from the archive's own spot stream per Phase 3. z_spot_6h =
# (spot - 6h rolling mean)/6h rolling std, causal. Discovery: z>1.5 ->
# n=239, 64% WR, +$6,218, p=0.000, 3/3 wks, topday 22%. Partially distinct
# from RESCUE-B (135/239 overlap; C-only +$2,664): liquidation pressure and
# price extension are two mechanisms converging on NO-into-stretched-tape.
# Union B|C: n=419, +$10,539, 3/3 wks — the composite candidate.
with np.errstate(divide="ignore", invalid="ignore"):
    _ss = pd.Series(df["spot"].values, index=pd.DatetimeIndex(df["dt"]))
    _rl = _ss.rolling("6h")
    df["z_spot_6h"] = ((_ss - _rl.mean()) / _rl.std()).values
BOOKS = {
    "RESCUE-A": (res["p_market"].between(0.35, 0.65) & (df["edge_yes"] >= 0.03)
                 & (df["edge_yes"] < 0.06) & (df["adx_1h"] >= 22), "yes"),
    "RESCUE-B": (res["p_market"].between(0.20, 0.80) & (df["edge_no"] >= 0.06)
                 & (df["liq_bias"] >= 1.0), "no"),
    "RESCUE-C": (res["p_market"].between(0.20, 0.80) & (df["edge_no"] >= 0.06)
                 & (df["z_spot_6h"] > 1.5), "no"),
    "RESCUE-B|C": (res["p_market"].between(0.20, 0.80) & (df["edge_no"] >= 0.06)
                   & ((df["liq_bias"] >= 1.0) | (df["z_spot_6h"] > 1.5)), "no"),
}
for name, (mask, side) in BOOKS.items():
    q = res[mask.reindex(res.index).fillna(False)].sort_values("dt") \
        .drop_duplicates("contract_ticker", keep="first")
    if not len(q):
        print(f"{name}: no rows yet"); continue
    win = (q["resolved_yes"] == 1) if side == "yes" else (q["resolved_yes"] == 0)
    cost = q["p_market"] if side == "yes" else 1 - q["p_market"]
    f2 = 0.07 * q["p_market"] * (1 - q["p_market"])
    pnl = np.where(win, 100 * (1 - cost) / cost, -100) - (100 / cost) * f2
    wk = pd.Series(pnl, index=q.index).groupby(q["dt"].dt.isocalendar().week).sum()
    print(f"{name} ({side}): n={len(q)} WR={win.mean():.0%} vs BE={cost.mean():.0%} "
          f"net=${pnl.sum():+,.0f}  weekly={{{', '.join(f'{int(k)}: {v:+.0f}' for k, v in wk.items())}}}")
