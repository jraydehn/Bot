"""
S3 -- comprehensive_rescue search on the two BLOCK-candidate states from the
SOL 15m VWAP HMM (State 1 YES, State 5 NO), following the project's
5-phase methodology: enumerate every column, verify coverage, sweep full
decile grid both directions, bootstrap significance on survivors.

Also runs a lighter robustness check on the BOOST candidates (State 2 NO,
State 5 YES, State 1 NO) -- not a rescue (nothing to rescue, they're
already winning), just confirms no single confounding column is secretly
driving the effect before considering deployment.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/vwap_hmm_sol15m_20260708"
rng = np.random.default_rng(23)

df = pd.read_csv(f"{OUT}/full_analysis.csv", low_memory=False)
df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")

EXCLUDE = {
    "logged_at", "contract_ticker", "close_ts", "spot", "strike", "spot_at_expiry",
    "price_move_pct", "miss_pct", "resolved_yes", "won", "be", "trade_edge",
    "close_ts_dt", "iso_week", "vwap_hmm_state", "model_side", "edge_yes", "edge_no",
    "p_model_yes", "p_model_no", "p_market",
}
CAND_COLS = [c for c in df.columns if c not in EXCLUDE]


def coverage_and_sweep(sub, label):
    print(f"\n=== {label}: n={len(sub)}  base WR={sub['won'].mean():.3f}  base BE={sub['be'].mean():.3f} ===")
    found = []
    skipped = []
    for feat in CAND_COLS:
        col = sub[feat]
        nn = col.notna().sum()
        if nn < 15:
            skipped.append((feat, nn))
            continue
        if col.dtype == bool or col.dropna().nunique() <= 6:
            for val in col.dropna().unique():
                mask = col == val
                s = sub[mask]
                if len(s) < 10 or len(sub[~mask]) < 10:
                    continue
                wr, be = s["won"].mean(), s["be"].mean()
                found.append({"feature": feat, "split": f"=={val}", "n": len(s),
                             "wr": wr, "be": be, "edge": wr - be})
            continue
        vals = pd.to_numeric(col, errors="coerce")
        s2 = sub[vals.notna()]
        if len(s2) < 15:
            continue
        vv = vals.dropna()
        for q in np.arange(0.1, 1.0, 0.1):
            thresh = vv.quantile(q)
            for direction, mask in [(">=", vv >= thresh), ("<", vv < thresh)]:
                s = s2.loc[mask.index[mask]]
                if len(s) < 10 or (len(s2) - len(s)) < 10:
                    continue
                wr, be = s["won"].mean(), s["be"].mean()
                found.append({"feature": feat, "split": f"{direction}{thresh:.4g}(q{q:.1f})",
                             "n": len(s), "wr": wr, "be": be, "edge": wr - be})
    fd = pd.DataFrame(found)
    print(f"  {len(CAND_COLS)} candidate cols, {len(skipped)} skipped for <15 non-null, {len(fd)} splits tested")
    if skipped:
        print(f"  skipped: {[s[0] for s in skipped]}")
    return fd


def boot_p(edges, n_boot=4000):
    e = np.asarray(edges); n = len(e)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()


# ── BLOCK candidates: rescue search ──────────────────────────────────────────
print("#" * 70)
print("RESCUE SEARCH ON BLOCK CANDIDATES")
print("#" * 70)

for state, side in [(1, "YES"), (5, "NO")]:
    pop = df[(df["vwap_hmm_state"] == state) & (df["model_side"] == side)]
    fd = coverage_and_sweep(pop, f"State {state} {side} (blocked bucket)")
    if len(fd) == 0:
        continue
    real = fd[(fd["edge"] > 0) & (fd["n"] >= 15)]
    print(f"  splits with positive edge (n>=15): {len(real)} / {len(fd)}")
    if len(real):
        top = real.sort_values("edge", ascending=False).head(10)
        print(top.round(3).to_string(index=False))
        print("\n  bootstrap on top candidates:")
        for _, r in top.head(5).iterrows():
            feat, split = r["feature"], r["split"]
            if split.startswith("=="):
                val = split[2:]
                try:
                    val = float(val)
                except Exception:
                    pass
                mask = pop[feat] == val
            else:
                direction = split[0] if split[0] in "<>" else split[:2]
                thresh = float(split.split("(")[0].lstrip("><=").strip())
                vals = pd.to_numeric(pop[feat], errors="coerce")
                mask = vals >= thresh if split.startswith(">=") else vals < thresh
            sub_r = pop[mask]
            edges = (sub_r["won"].astype(float) - sub_r["be"]).values
            m, lo, hi, p = boot_p(edges)
            print(f"    {feat} {split}: n={len(sub_r)} edge_mean={m:+.4f} "
                  f"CI=[{lo:+.4f},{hi:+.4f}] P(edge<=0)={p:.4f}")

# ── BOOST candidates: confound check ─────────────────────────────────────────
print("\n" + "#" * 70)
print("CONFOUND CHECK ON BOOST CANDIDATES (not a rescue -- already winning)")
print("#" * 70)

for state, side in [(2, "NO"), (5, "YES"), (1, "NO")]:
    pop = df[(df["vwap_hmm_state"] == state) & (df["model_side"] == side)]
    print(f"\n=== State {state} {side}: n={len(pop)} WR={pop['won'].mean():.3f} BE={pop['be'].mean():.3f} ===")
    # Check if a single other column's extreme values fully explain the effect
    # (i.e. if excluding the top decile of some column kills the edge, it's a confound)
    for feat in ["ls_long_pct", "liq_bias", "composite_p_up", "chg_15m", "stoch_k_15m", "rsi_1h"]:
        if feat not in pop.columns:
            continue
        vals = pd.to_numeric(pop[feat], errors="coerce")
        if vals.notna().sum() < 15:
            print(f"  {feat}: insufficient coverage ({vals.notna().sum()})")
            continue
        corr = vals.corr(pop["won"].astype(float))
        print(f"  {feat}: coverage={vals.notna().sum()}/{len(pop)}  corr_with_win={corr:+.3f}")

print("\nDone.")
