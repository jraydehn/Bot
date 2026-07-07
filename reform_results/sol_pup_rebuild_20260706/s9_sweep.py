"""
S9 -- Full decile-grid comprehensive rescue sweep for BOTH SOL AGREE and
DISAGREE populations, against every candidate column present in the merged
full-column CSVs (s7's output, optionally enriched with s8's reconstructed
columns if present). Per comprehensive_rescue skill: full decile grid both
directions for continuous, direct category comparison for boolean/low-card.
"""
import sys
import numpy as np
import pandas as pd

OUT = "reform_results/sol_pup_rebuild_20260706"
EXCLUDE = {
    "logged_at", "decision_time", "contract_ticker", "close_ts", "p_market_source",
    "decision", "side", "gate_blocked", "kelly_fraction", "bet_fraction", "bet_amount",
    "bankroll", "contracts_scanned", "resolved_yes", "would_win", "would_win_bf", "would_pnl",
    "would_pnl_bf", "spot_at_expiry", "price_move_pct", "miss_pct", "loss_margin_pct",
    "loss_category", "logged_at_parsed", "week", "yw", "be", "agree", "p_sol", "asset",
    "spot", "strike", "neutral_gate", "pure_edge_gate",
}


def sweep(pop, label):
    pop = pop.copy()
    pop["would_win"] = pop["would_win_bf"] if "would_win_bf" in pop.columns else pop["would_win"]
    pop["would_pnl"] = pop["would_pnl_bf"] if "would_pnl_bf" in pop.columns else pop["would_pnl"]
    cand_cols = [c for c in pop.columns if c not in EXCLUDE]
    print(f"\n=== {label}: n={len(pop)}, base WR={pop['would_win'].mean():.3f}, "
          f"base BE={pop['be'].mean():.3f}, base PnL=${pop['would_pnl'].sum():.2f}, "
          f"candidate cols={len(cand_cols)} ===")

    found = []
    n_tests = 0
    coverage_rows = []
    for feat in cand_cols:
        col = pop[feat]
        nn = col.notna().sum()
        coverage_rows.append({"feature": feat, "non_null": nn})
        if nn < 20:
            continue
        if col.dtype == bool or col.dropna().nunique() <= 6:
            for val in col.dropna().unique():
                mask = col == val
                rescued = pop[mask]; remainder = pop[~mask]
                n_tests += 1
                if len(rescued) < 10 or len(remainder) < 10:
                    continue
                r_edge = rescued["would_win"].mean() - rescued["be"].mean()
                rem_edge = remainder["would_win"].mean() - remainder["be"].mean()
                wk_pnl = rescued.groupby("yw")["would_pnl"].sum()
                worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
                found.append({"feature": feat, "split": f"=={val}", "n_rescued": len(rescued),
                             "wr_rescued": rescued["would_win"].mean(), "edge_rescued": r_edge,
                             "pnl_rescued": rescued["would_pnl"].sum(), "n_weeks": len(wk_pnl),
                             "worst_wk_share": worst_share, "edge_remainder": rem_edge,
                             "pnl_remainder": remainder["would_pnl"].sum()})
            continue
        vals = pd.to_numeric(col, errors="coerce")
        sub = pop[vals.notna()].copy()
        if len(sub) < 20:
            continue
        vv = vals.dropna()
        for q in np.arange(0.1, 1.0, 0.1):
            thresh = vv.quantile(q)
            for direction, mask in [(">=", vv >= thresh), ("<", vv < thresh)]:
                n_tests += 1
                rescued = sub.loc[mask.index[mask]]; remainder = sub.loc[mask.index[~mask]]
                if len(rescued) < 10 or len(remainder) < 10:
                    continue
                r_edge = rescued["would_win"].mean() - rescued["be"].mean()
                rem_edge = remainder["would_win"].mean() - remainder["be"].mean()
                wk_pnl = rescued.groupby("yw")["would_pnl"].sum()
                worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
                found.append({"feature": feat, "split": f"{direction}{thresh:.4g}(q{q:.1f})",
                             "n_rescued": len(rescued), "wr_rescued": rescued["would_win"].mean(),
                             "edge_rescued": r_edge, "pnl_rescued": rescued["would_pnl"].sum(),
                             "n_weeks": len(wk_pnl), "worst_wk_share": worst_share,
                             "edge_remainder": rem_edge, "pnl_remainder": remainder["would_pnl"].sum()})

    df_found = pd.DataFrame(found)
    cov = pd.DataFrame(coverage_rows).sort_values("non_null")
    skipped = cov[cov["non_null"] < 20]
    print(f"total splits tested: {n_tests} (across {len(cand_cols)} candidate columns)")
    print(f"columns SKIPPED for <20 non-null (silently excluded otherwise): {len(skipped)}")
    print(skipped.to_string(index=False))
    if len(df_found):
        real = df_found[(df_found["edge_rescued"] > 0) & (df_found["n_rescued"] >= 15)]
        print(f"\nsplits where subset crosses breakeven (edge>0, n>=15): {len(real)}")
        if len(real):
            print(real.sort_values("edge_rescued", ascending=False).head(25).round(3).to_string(index=False))
    df_found.to_csv(f"{OUT}/rescue_sweep_{label}_all.csv", index=False)
    return df_found


if __name__ == "__main__":
    import glob
    for pop_file, label in [("sol_agree_full.csv", "agree"), ("sol_disagree_full.csv", "disagree")]:
        recon_file = f"{OUT}/sol_{label}_fully_reconstructed.csv"
        import os
        path = recon_file if os.path.exists(recon_file) else f"{OUT}/{pop_file}"
        pop = pd.read_csv(path, low_memory=False)
        sweep(pop, label)
