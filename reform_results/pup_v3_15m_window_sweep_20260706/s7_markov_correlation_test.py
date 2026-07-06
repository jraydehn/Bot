"""
S7 -- Test p_up_v3's correlation with price movement through a Markov-chain
lens: discretize p_up_v3 into states, build the empirical transition
table P(next-hour actually up | current p_up_v3 state), and test whether
it deviates from the unconditional base rate (chi-square + per-state CI).

Uses the HONEST walk-forward OOS predictions from the original v3 build
(wf_preds_FINAL.parquet, 48,119 hours, 2021-2026) -- the biggest and most
rigorous p_up_v3 dataset available, not the smaller real-trade backfill.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

REBUILD = "reform_results/pup_v2_rebuild_20260704"
ev = pd.read_parquet(f"{REBUILD}/wf_preds_FINAL.parquet").dropna()
print(f"n={len(ev)}  ({ev.index.min()} -> {ev.index.max()})")
print(f"base rate (unconditional P(up)): {ev['label'].mean():.4f}")

# ── Markov-chain style: discretize p_up_v3 into 5 states, build the
# transition table state -> realized outcome ────────────────────────────
ev = ev.copy()
ev["state"] = pd.qcut(ev["p"], 5, labels=["S1(lowest)", "S2", "S3", "S4", "S5(highest)"])

print("\n=== Transition table: P(next-hour up | p_up_v3 state) ===")
tab = ev.groupby("state").agg(n=("label", "size"), p_up=("label", "mean"),
                              p_mean=("p", "mean"), p_lo=("p", "min"), p_hi=("p", "max"))
tab["se"] = np.sqrt(tab["p_up"] * (1 - tab["p_up"]) / tab["n"])
tab["ci95_lo"] = tab["p_up"] - 1.96 * tab["se"]
tab["ci95_hi"] = tab["p_up"] + 1.96 * tab["se"]
print(tab.round(4).to_string())

# chi-square test: is state independent of outcome?
ct = pd.crosstab(ev["state"], ev["label"])
chi2, pval, dof, exp = chi2_contingency(ct)
print(f"\nchi-square test (state vs realized outcome): chi2={chi2:.2f}  p={pval:.2e}  dof={dof}")

# ── First-order Markov chain on STATE TRANSITIONS themselves: does a
# high p_up_v3 state predict the NEXT hour's p_up_v3 state (persistence),
# and does that in turn correlate with 2-steps-ahead realized direction? ──
ev_sorted = ev.sort_index()
ev_sorted["state_next"] = ev_sorted["state"].shift(-1)
trans = pd.crosstab(ev_sorted["state"], ev_sorted["state_next"], normalize="index")
print("\n=== State-to-state transition matrix (does p_up_v3 persist?) ===")
print(trans.round(3).to_string())

# ── Year-by-year robustness of the S1 vs S5 spread (is the correlation
# stable, or concentrated in specific years like the earlier findings?) ──
print("\n=== S1 vs S5 P(up) spread, by year ===")
ev_sorted["year"] = ev_sorted.index.year
for yr, g in ev_sorted.groupby("year"):
    s1 = g[g["state"] == "S1(lowest)"]["label"].mean()
    s5 = g[g["state"] == "S5(highest)"]["label"].mean()
    n1 = (g["state"] == "S1(lowest)").sum()
    n5 = (g["state"] == "S5(highest)").sum()
    print(f"  {yr}: S1 P(up)={s1:.3f} (n={n1})   S5 P(up)={s5:.3f} (n={n5})   spread={s5-s1:+.3f}")
