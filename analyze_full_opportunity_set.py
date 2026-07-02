"""
Full Opportunity Set Analysis
Analyzes executed + blocked trades to find exploitable mispricings.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FILE = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/full_opportunity_analysis.txt"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_executed_trades():
    """Load BTC/ETH/SOL executed trades, filter to resolved trades only."""
    dfs = []
    for asset, path in [
        ("BTC", "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv"),
        ("ETH", "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_eth.csv"),
        ("SOL", "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_sol.csv"),
    ]:
        df = pd.read_csv(path, low_memory=False)
        df["asset"] = asset
        # Filter to executed trades with resolved outcomes
        df = df[(df["decision"] == "trade") & df["resolved_yes"].notna()].copy()
        # Rename columns to match blocked trades schema
        rename_map = {
            "p_market": "pm",
            "p_yes_model": "p_model",
            "vwap_stretch_score": "vwap_stretch",
            "ema_stretch_score": "ema_stretch",
        }
        df = df.rename(columns=rename_map)
        # Standardize columns
        df["source"] = "executed"
        df["gate_name"] = "executed"
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    executed = pd.concat(dfs, ignore_index=True)
    executed["resolved_yes"] = executed["resolved_yes"].astype(float)
    return executed


def load_blocked_trades():
    """Load blocked trades CSV."""
    df = pd.read_csv(
        "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/blocked_trades.csv",
        low_memory=False
    )
    df = df[df["resolved_yes"].notna()].copy()
    df["source"] = "blocked"
    df["resolved_yes"] = df["resolved_yes"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def flat_pnl(side, pm, resolved_yes, stake=10):
    """Compute flat-stake P&L for a single trade."""
    pm = np.asarray(pm, dtype=float)
    resolved_yes = np.asarray(resolved_yes, dtype=float)
    side = np.asarray(side)
    yes_mask = side == "yes"
    no_mask = side == "no"
    pnl = np.zeros(len(pm))
    # YES side: win if resolved_yes==1
    pnl[yes_mask & (resolved_yes == 1)] = (1 - pm[yes_mask & (resolved_yes == 1)]) * stake
    pnl[yes_mask & (resolved_yes == 0)] = -pm[yes_mask & (resolved_yes == 0)] * stake
    # NO side: win if resolved_yes==0
    pnl[no_mask & (resolved_yes == 0)] = pm[no_mask & (resolved_yes == 0)] * stake
    pnl[no_mask & (resolved_yes == 1)] = -(1 - pm[no_mask & (resolved_yes == 1)]) * stake
    return pnl


def actual_win_rate(side, pm, resolved_yes):
    """Return actual win rate for a group of trades."""
    side = np.asarray(side)
    resolved_yes = np.asarray(resolved_yes, dtype=float)
    yes_mask = side == "yes"
    no_mask = side == "no"
    wins = np.zeros(len(side))
    wins[yes_mask] = resolved_yes[yes_mask]
    wins[no_mask] = 1 - resolved_yes[no_mask]
    return wins.mean() if len(wins) > 0 else np.nan


def implied_wr(side, pm):
    """Market-implied win rate (pm for YES, 1-pm for NO)."""
    side = np.asarray(side)
    pm = np.asarray(pm, dtype=float)
    imp = np.where(side == "yes", pm, 1 - pm)
    return imp.mean() if len(imp) > 0 else np.nan


def binomial_pval(n_wins, n_total, p_expected):
    """Two-sided binomial test p-value."""
    if n_total == 0:
        return np.nan
    try:
        result = binomtest(int(n_wins), int(n_total), float(p_expected), alternative='two-sided')
        return result.pvalue
    except Exception:
        return np.nan


def group_stats(df, label=""):
    """Compute summary stats for a group DataFrame."""
    if len(df) == 0:
        return None
    side = df["side"].values
    pm = df["pm"].values.astype(float)
    ry = df["resolved_yes"].values.astype(float)

    # Win flag: YES side wins when resolved_yes==1, NO side wins when resolved_yes==0
    yes_mask = side == "yes"
    no_mask = side == "no"
    wins = np.zeros(len(side))
    wins[yes_mask] = ry[yes_mask]
    wins[no_mask] = 1 - ry[no_mask]

    n = len(df)
    n_wins = wins.sum()
    act_wr = wins.mean()
    imp = np.where(yes_mask, pm, 1 - pm).mean()
    edge = act_wr - imp
    pnl = flat_pnl(side, pm, ry).sum()
    pval = binomial_pval(n_wins, n, imp)
    return dict(n=n, actual_wr=act_wr, implied_wr=imp, edge=edge, flat_pnl=pnl, pval=pval)


def fmt_row(stats_dict, label):
    if stats_dict is None:
        return f"  {label}: n=0"
    s = stats_dict
    pval_str = f"{s['pval']:.4f}" if s['pval'] is not None and not np.isnan(s['pval']) else "n/a"
    return (
        f"  {label}: n={s['n']:4d}  actual_WR={s['actual_wr']:.3f}  "
        f"implied_WR={s['implied_wr']:.3f}  edge={s['edge']:+.3f}  "
        f"flat_PnL=${s['flat_pnl']:+.1f}  p={pval_str}"
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 80)
    out("FULL OPPORTUNITY SET ANALYSIS")
    out("=" * 80)

    # ---- Load ----
    out("\nLoading data...")
    executed = load_executed_trades()
    blocked = load_blocked_trades()

    out(f"  Executed trades (resolved): {len(executed)}")
    out(f"  Blocked trades  (resolved): {len(blocked)}")

    # Standardize blocked trades columns to match executed where needed
    # blocked has 'pm' already; executed renamed p_market->pm
    # Pool for combined analysis
    exec_cols = ["asset", "source", "gate_name", "side", "pm", "p_model", "net_edge",
                 "offset_pct", "tau_minutes", "ema_stack_bias", "composite_trend",
                 "composite_rev", "composite_p_up", "stoch_k", "vwap_stretch",
                 "vol_score", "vpin_score", "obi_score", "ema_stretch",
                 "structure_bias", "funding_bias", "resolved_yes", "contract_ticker"]

    blk_cols = ["asset", "source", "gate_name", "side", "pm", "p_model", "net_edge",
                "offset_pct", "tau_minutes", "ema_stack_bias", "composite_trend",
                "composite_rev", "composite_p_up", "stoch_k", "vwap_stretch",
                "vol_score", "vpin_score", "obi_score", "ema_stretch",
                "structure_bias", "funding_bias", "resolved_yes", "contract_ticker"]

    def safe_subset(df, cols):
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    exec_sub = safe_subset(executed, exec_cols)
    blk_sub = safe_subset(blocked, blk_cols)
    combined = pd.concat([exec_sub, blk_sub], ignore_index=True)
    combined["pm"] = pd.to_numeric(combined["pm"], errors="coerce")
    combined["resolved_yes"] = pd.to_numeric(combined["resolved_yes"], errors="coerce")
    combined = combined.dropna(subset=["pm", "resolved_yes", "side"])
    combined["side"] = combined["side"].str.lower().str.strip()
    combined = combined[combined["side"].isin(["yes", "no"])]

    out(f"  Combined (after cleanup): {len(combined)}")
    out(f"  Combined by source: {combined['source'].value_counts().to_dict()}")
    out(f"  Combined by side: {combined['side'].value_counts().to_dict()}")
    out(f"  Combined by asset: {combined['asset'].value_counts().to_dict()}")

    # =========================================================================
    # SECTION 1: Calibration of market price (pm) vs actual outcome
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 1: MARKET PRICE CALIBRATION (pm vs actual outcome)")
    out("=" * 80)
    out("Bucketing all trades by pm in 0.05-wide bins. For YES: WR=fraction(resolved_yes==1).")
    out("For NO: WR=fraction(resolved_yes==0). Edge = actual_WR - implied_WR.")
    out("")

    pm_bins = np.arange(0.0, 1.05, 0.05)
    pm_labels = [f"{b:.2f}-{b+0.05:.2f}" for b in pm_bins[:-1]]
    combined["pm_bin"] = pd.cut(combined["pm"], bins=pm_bins, labels=pm_labels, right=False)

    for src_label, src_df in [("ALL (exec+blocked)", combined),
                               ("EXECUTED ONLY", combined[combined["source"] == "executed"]),
                               ("BLOCKED ONLY", combined[combined["source"] == "blocked"])]:
        out(f"\n  -- {src_label} --")
        out(f"  {'pm_bin':<14} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        out(f"  {'-'*14} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")

        high_edge_rows = []

        for pm_label in pm_labels:
            for side in ["yes", "no"]:
                grp = src_df[(src_df["pm_bin"] == pm_label) & (src_df["side"] == side)]
                if len(grp) < 10:
                    continue
                s = group_stats(grp)
                if s is None:
                    continue
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                edge_flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                out(f"  {pm_label:<14} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} "
                    f"{pval_str:>7}{edge_flag}")
                if abs(s["edge"]) > 0.05 and s["n"] > 30 and s["flat_pnl"] > 50:
                    high_edge_rows.append(
                        dict(pm_bin=pm_label, side=side, src=src_label[:8], **s)
                    )

        if high_edge_rows:
            out(f"\n  HIGH-EDGE BUCKETS (|edge|>5%, n>30, PnL>$50):")
            for row in sorted(high_edge_rows, key=lambda r: r["flat_pnl"], reverse=True):
                out(f"    pm={row['pm_bin']} {row['side']} | n={row['n']} | "
                    f"act_WR={row['actual_wr']:.3f} | implied={row['implied_wr']:.3f} | "
                    f"edge={row['edge']:+.3f} | PnL=${row['flat_pnl']:+.1f} | p={row['pval']:.4f}")

    # =========================================================================
    # SECTION 2: Feature-based edge analysis on BLOCKED trades
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 2: FEATURE-BASED EDGE ANALYSIS (BLOCKED TRADES)")
    out("=" * 80)

    blk = combined[combined["source"] == "blocked"].copy()

    features = {
        "ema_stack_bias": sorted(blk["ema_stack_bias"].dropna().unique()) if "ema_stack_bias" in blk.columns else [],
        "composite_trend_bucket": None,
        "composite_rev_bucket": None,
        "stoch_k_bucket": None,
        "vpin_score_bucket": None,
    }

    def bucket_composite_trend(df):
        col = pd.to_numeric(df["composite_trend"], errors="coerce")
        bins = [-np.inf, -3, -1.5, 1.5, 3, np.inf]
        labels = ["<=-3", "-2to-1", "0", "1to2", ">=3"]
        return pd.cut(col, bins=bins, labels=labels)

    def bucket_composite_rev(df):
        col = pd.to_numeric(df["composite_rev"], errors="coerce")
        bins = [-np.inf, -2, 0, 2, 4, np.inf]
        labels = ["<=-2", "-1to0", "1to2", "3to4", ">=5"]
        return pd.cut(col, bins=bins, labels=labels)

    def bucket_stoch_k(df):
        col = pd.to_numeric(df["stoch_k"], errors="coerce")
        bins = [0, 20, 40, 60, 80, 100]
        labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        return pd.cut(col, bins=bins, labels=labels, right=True)

    def bucket_vpin(df):
        col = pd.to_numeric(df["vpin_score"], errors="coerce")
        bins = [-np.inf, -1, 0, 1, np.inf]
        labels = ["-2", "-1", "0", "1"]
        return pd.cut(col, bins=bins, labels=labels)

    if "ema_stack_bias" in blk.columns:
        out("\n  2a. ema_stack_bias x side:")
        out(f"  {'ema_stack':<12} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        blk_ema = blk.copy()
        blk_ema["ema_stack_bias"] = pd.to_numeric(blk_ema["ema_stack_bias"], errors="coerce")
        for val in [-1, 0, 1]:
            for side in ["yes", "no"]:
                grp = blk_ema[(blk_ema["ema_stack_bias"] == val) & (blk_ema["side"] == side)]
                s = group_stats(grp)
                if s is None or s["n"] < 10:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                out(f"  {val:<12} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} {pval_str:>7}{flag}")

    if "composite_trend" in blk.columns:
        out("\n  2b. composite_trend bucket x side:")
        out(f"  {'ctbucket':<12} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        blk["ct_bucket"] = bucket_composite_trend(blk)
        for bucket in ["<=-3", "-2to-1", "0", "1to2", ">=3"]:
            for side in ["yes", "no"]:
                grp = blk[(blk["ct_bucket"] == bucket) & (blk["side"] == side)]
                s = group_stats(grp)
                if s is None or s["n"] < 10:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                out(f"  {bucket:<12} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} {pval_str:>7}{flag}")

    if "composite_rev" in blk.columns:
        out("\n  2c. composite_rev bucket x side:")
        out(f"  {'crbucket':<12} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        blk["cr_bucket"] = bucket_composite_rev(blk)
        for bucket in ["<=-2", "-1to0", "1to2", "3to4", ">=5"]:
            for side in ["yes", "no"]:
                grp = blk[(blk["cr_bucket"] == bucket) & (blk["side"] == side)]
                s = group_stats(grp)
                if s is None or s["n"] < 10:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                out(f"  {bucket:<12} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} {pval_str:>7}{flag}")

    if "stoch_k" in blk.columns:
        out("\n  2d. stoch_k bucket x side:")
        out(f"  {'stoch_k':<12} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        blk["stoch_k_numeric"] = pd.to_numeric(blk["stoch_k"], errors="coerce")
        blk["sk_bucket"] = bucket_stoch_k(blk)
        for bucket in ["0-20", "20-40", "40-60", "60-80", "80-100"]:
            for side in ["yes", "no"]:
                grp = blk[(blk["sk_bucket"] == bucket) & (blk["side"] == side)]
                s = group_stats(grp)
                if s is None or s["n"] < 10:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                out(f"  {bucket:<12} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} {pval_str:>7}{flag}")

    if "vpin_score" in blk.columns:
        out("\n  2e. vpin_score bucket x side:")
        out(f"  {'vpin':<12} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9} {'p-val':>7}")
        blk["vpin_bucket"] = bucket_vpin(blk)
        for bucket in ["-2", "-1", "0", "1"]:
            for side in ["yes", "no"]:
                grp = blk[(blk["vpin_bucket"] == bucket) & (blk["side"] == side)]
                s = group_stats(grp)
                if s is None or s["n"] < 10:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                pval_str = f"{s['pval']:.4f}" if not np.isnan(s['pval']) else "  n/a"
                out(f"  {bucket:<12} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f} {pval_str:>7}{flag}")

    # OVER-BLOCKING analysis
    out("\n  2f. OVER-BLOCKING: Blocked trades where actual_WR > implied_WR (we missed edge)")
    out("  (Grouped by pm bin + side, showing top 15 by flat_PnL missed)")
    missed_rows = []
    for pm_label in pm_labels:
        for side in ["yes", "no"]:
            grp = blk[(blk["pm_bin"] == pm_label) & (blk["side"] == side)] if "pm_bin" in blk.columns else \
                  blk[(pd.cut(blk["pm"], bins=pm_bins, labels=pm_labels, right=False) == pm_label) & (blk["side"] == side)]
            if len(grp) < 20:
                continue
            s = group_stats(grp)
            if s is None:
                continue
            if s["actual_wr"] > s["implied_wr"] and s["n"] >= 20:
                missed_rows.append(dict(pm_bin=pm_label, side=side, **s))

    blk["pm_bin_tmp"] = pd.cut(blk["pm"], bins=pm_bins, labels=pm_labels, right=False)
    for row in sorted(missed_rows, key=lambda r: r["flat_pnl"], reverse=True)[:15]:
        out(f"    pm={row['pm_bin']} {row['side']}: n={row['n']} act_WR={row['actual_wr']:.3f} "
            f"implied={row['implied_wr']:.3f} edge={row['edge']:+.3f} PnL=${row['flat_pnl']:+.1f} p={row['pval']:.4f}")

    # =========================================================================
    # SECTION 3: Entry timing analysis
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 3: ENTRY TIMING ANALYSIS (tau_minutes)")
    out("=" * 80)
    out("Does the same contract evaluated at different tau windows show different edge?")
    out("")

    tau_bins = [0, 15, 30, 45, 60, 999]
    tau_labels = ["<15m", "15-30m", "30-45m", "45-60m", ">60m"]

    if "tau_minutes" in combined.columns:
        combined["tau_minutes_num"] = pd.to_numeric(combined["tau_minutes"], errors="coerce")
        combined["tau_bucket"] = pd.cut(combined["tau_minutes_num"], bins=tau_bins, labels=tau_labels, right=False)

        out(f"  {'tau':<10} {'side':<6} {'src':<10} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9}")
        for tau in tau_labels:
            for side in ["yes", "no"]:
                for src_label, src_filter in [("exec+blk", None), ("exec", "executed"), ("blocked", "blocked")]:
                    if src_filter:
                        grp = combined[(combined["tau_bucket"] == tau) & (combined["side"] == side) & (combined["source"] == src_filter)]
                    else:
                        grp = combined[(combined["tau_bucket"] == tau) & (combined["side"] == side)]
                    if len(grp) < 10:
                        continue
                    s = group_stats(grp)
                    if s is None:
                        continue
                    flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                    out(f"  {tau:<10} {side:<6} {src_label:<10} {s['n']:>5} {s['actual_wr']:>7.3f} "
                        f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f}{flag}")

    # PM drift analysis: for contracts evaluated multiple times, does pm drift toward truth?
    out("\n  3b. PM drift: does pm move toward the actual outcome as tau decreases?")
    out("  (For contracts appearing multiple times, rank observations by tau_minutes desc)")
    if "tau_minutes_num" in combined.columns and "contract_ticker" in combined.columns:
        multi = combined.dropna(subset=["tau_minutes_num", "pm", "resolved_yes"]).copy()
        # Only contracts with >1 evaluation
        ticker_counts = multi.groupby("contract_ticker").size()
        multi_tickers = ticker_counts[ticker_counts > 1].index
        multi = multi[multi["contract_ticker"].isin(multi_tickers)].copy()
        multi = multi.sort_values(["contract_ticker", "tau_minutes_num"], ascending=[True, False])
        # Compute correlation: as tau decreases (observations rank 1=first eval, N=last eval),
        # does pm get closer to resolved_yes?
        multi["tau_rank"] = multi.groupby("contract_ticker")["tau_minutes_num"].rank(ascending=False)
        # pm distance from resolved_yes
        multi["pm_dist"] = (multi["pm"] - multi["resolved_yes"]).abs()
        # Early vs late: does early eval have higher pm_dist?
        early = multi[multi["tau_rank"] == 1]
        late = multi[multi["tau_rank"] == multi.groupby("contract_ticker")["tau_rank"].transform("max")]
        early_dist = early["pm_dist"].mean()
        late_dist = late["pm_dist"].mean()
        out(f"  Multi-eval contracts: {len(multi_tickers)}")
        out(f"  Early (high tau) avg |pm - outcome|: {early_dist:.4f}")
        out(f"  Late (low tau)  avg |pm - outcome|: {late_dist:.4f}")
        if early_dist > late_dist:
            out("  -> pm CONVERGES toward outcome as tau decreases (late entry = more info, worse odds)")
        else:
            out("  -> pm does NOT clearly converge (early entry not obviously less informed)")

        # Correlation between tau_rank and pm_dist
        corr, pval = stats.spearmanr(multi["tau_rank"], multi["pm_dist"], nan_policy="omit")
        out(f"  Spearman corr(tau_rank, pm_dist): r={corr:.4f}, p={pval:.4f}")
        out("  (positive r = higher rank = larger tau = larger pm error = early entry less accurate)")

    # =========================================================================
    # SECTION 4: Model vs market vs reality (executed trades)
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 4: MODEL VS MARKET VS REALITY (EXECUTED TRADES)")
    out("=" * 80)

    exec_df = combined[combined["source"] == "executed"].copy()
    exec_df["p_model_num"] = pd.to_numeric(exec_df["p_model"], errors="coerce")
    exec_df["pm_num"] = pd.to_numeric(exec_df["pm"], errors="coerce")
    exec_df = exec_df.dropna(subset=["p_model_num", "pm_num", "resolved_yes"])

    # Overall model bias
    out(f"\n  Executed trades with model+market data: {len(exec_df)}")
    if len(exec_df) > 0:
        # Model sees YES edge: p_model > pm
        yes_edge = exec_df[exec_df["p_model_num"] > exec_df["pm_num"]]
        no_edge = exec_df[exec_df["p_model_num"] <= exec_df["pm_num"]]
        out(f"\n  Model sees YES edge (p_model > pm): n={len(yes_edge)}")
        if len(yes_edge) > 0:
            act_wr = yes_edge["resolved_yes"].mean()
            mean_pm = yes_edge["pm_num"].mean()
            out(f"    actual_WR={act_wr:.3f}  mean_pm={mean_pm:.3f}  edge_vs_pm={act_wr-mean_pm:+.3f}")

        out(f"\n  Model sees NO/flat edge (p_model <= pm): n={len(no_edge)}")
        if len(no_edge) > 0:
            act_wr = no_edge["resolved_yes"].mean()
            mean_pm = no_edge["pm_num"].mean()
            out(f"    actual_WR={act_wr:.3f}  mean_pm={mean_pm:.3f}  edge_vs_pm={act_wr-mean_pm:+.3f}")

        # Model bias: overall avg p_model vs avg actual_WR
        avg_pm = exec_df["pm_num"].mean()
        avg_p_model = exec_df["p_model_num"].mean()
        avg_ry = exec_df["resolved_yes"].mean()
        out(f"\n  Overall: avg_pm={avg_pm:.3f}  avg_p_model={avg_p_model:.3f}  avg_actual_WR={avg_ry:.3f}")
        out(f"  Model bias vs actual: {avg_p_model - avg_ry:+.3f} (positive = overestimates YES probability)")
        out(f"  Market bias vs actual: {avg_pm - avg_ry:+.3f}")

        # Where is model MOST wrong
        exec_df["model_error"] = (exec_df["p_model_num"] - exec_df["resolved_yes"]).abs()
        worst = exec_df.nlargest(20, "model_error")[
            ["contract_ticker", "side", "pm_num", "p_model_num", "resolved_yes", "model_error", "asset"]
        ] if "asset" in exec_df.columns else exec_df.nlargest(20, "model_error")[
            ["contract_ticker", "side", "pm_num", "p_model_num", "resolved_yes", "model_error"]
        ]
        out(f"\n  Biggest model errors (top 20 by |p_model - resolved_yes|):")
        out(f"  {'ticker':<45} {'side':<5} {'pm':>5} {'p_mod':>6} {'ry':>4} {'err':>5}")
        for _, row in worst.iterrows():
            out(f"  {row['contract_ticker']:<45} {row['side']:<5} "
                f"{row['pm_num']:>5.3f} {row['p_model_num']:>6.3f} "
                f"{row['resolved_yes']:>4.0f} {row['model_error']:>5.3f}")

        # Directional bias by asset
        if "asset" in exec_df.columns:
            out(f"\n  Model bias by asset:")
            for asset in exec_df["asset"].dropna().unique():
                adf = exec_df[exec_df["asset"] == asset]
                if len(adf) < 5:
                    continue
                bias = adf["p_model_num"].mean() - adf["resolved_yes"].mean()
                out(f"    {asset}: n={len(adf)}  avg_p_model={adf['p_model_num'].mean():.3f}  "
                    f"avg_actual={adf['resolved_yes'].mean():.3f}  bias={bias:+.3f}")

    # =========================================================================
    # SECTION 5: R:R analysis — does lower pm predict higher actual WR for YES?
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 5: RISK/REWARD ANALYSIS (pm vs actual WR for YES bets)")
    out("=" * 80)
    out("Does lower pm (better R:R) correlate with higher actual WR on YES bets?")
    out("")

    yes_all = combined[combined["side"] == "yes"].copy()
    yes_all["pm_num"] = pd.to_numeric(yes_all["pm"], errors="coerce")
    yes_all = yes_all.dropna(subset=["pm_num", "resolved_yes"])

    out(f"  {'pm_bin':<14} {'n':>5} {'act_WR':>7} {'pm_mean':>8} {'WR-pm':>7} {'flat_PnL':>9}")
    overall_corr_rows = []
    for pm_label in pm_labels:
        grp = yes_all[yes_all["pm_bin"] == pm_label] if "pm_bin" in yes_all.columns else \
              yes_all[pd.cut(yes_all["pm_num"], bins=pm_bins, labels=pm_labels, right=False) == pm_label]
        if len(grp) < 10:
            continue
        act_wr = grp["resolved_yes"].mean()
        mean_pm = grp["pm_num"].mean()
        pnl = flat_pnl(grp["side"].values, grp["pm_num"].values, grp["resolved_yes"].values).sum()
        out(f"  {pm_label:<14} {len(grp):>5} {act_wr:>7.3f} {mean_pm:>8.3f} "
            f"{act_wr - mean_pm:>+7.3f} {pnl:>+9.1f}")
        overall_corr_rows.append((mean_pm, act_wr))

    if len(overall_corr_rows) >= 4:
        xs = [r[0] for r in overall_corr_rows]
        ys = [r[1] for r in overall_corr_rows]
        corr, pval = stats.spearmanr(xs, ys)
        out(f"\n  Spearman corr(mean_pm, actual_WR) for YES: r={corr:.4f}, p={pval:.4f}")
        if corr > 0.5:
            out("  -> Well-calibrated: higher pm = higher actual WR (market efficient for YES)")
        elif corr < -0.3:
            out("  -> MISPRICED: lower pm = higher actual WR (cheap contracts outperform, strong edge)")
        else:
            out("  -> No clear directional relationship between pm and WR")

    # Same analysis for NO bets
    out("\n  NO bets: pm bucket vs actual WR (win = resolved_yes==0)")
    no_all = combined[combined["side"] == "no"].copy()
    no_all["pm_num"] = pd.to_numeric(no_all["pm"], errors="coerce")
    no_all = no_all.dropna(subset=["pm_num", "resolved_yes"])

    out(f"  {'pm_bin':<14} {'n':>5} {'act_WR':>7} {'1-pm_mean':>9} {'WR-imp':>7} {'flat_PnL':>9}")
    no_corr_rows = []
    for pm_label in pm_labels:
        grp = no_all[pd.cut(no_all["pm_num"], bins=pm_bins, labels=pm_labels, right=False) == pm_label]
        if len(grp) < 10:
            continue
        act_wr = (grp["resolved_yes"] == 0).mean()
        mean_pm = grp["pm_num"].mean()
        implied = 1 - mean_pm
        pnl = flat_pnl(grp["side"].values, grp["pm_num"].values, grp["resolved_yes"].values).sum()
        out(f"  {pm_label:<14} {len(grp):>5} {act_wr:>7.3f} {implied:>9.3f} "
            f"{act_wr - implied:>+7.3f} {pnl:>+9.1f}")
        no_corr_rows.append((mean_pm, act_wr))

    # =========================================================================
    # SECTION 6: Cross-feature interactions (top 10 by flat_PnL)
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 6: CROSS-FEATURE INTERACTIONS (combined exec+blocked)")
    out("=" * 80)
    out("Finding 2-feature combos with n>=30, p<0.05, sorted by flat_PnL.")
    out("")

    # Prepare bucketted features on combined
    combined2 = combined.copy()
    combined2["pm_num"] = pd.to_numeric(combined2["pm"], errors="coerce")

    if "composite_trend" in combined2.columns:
        combined2["ct_bucket"] = bucket_composite_trend(combined2)
    if "composite_rev" in combined2.columns:
        combined2["cr_bucket"] = bucket_composite_rev(combined2)
    if "stoch_k" in combined2.columns:
        combined2["stoch_k_num"] = pd.to_numeric(combined2["stoch_k"], errors="coerce")
        combined2["sk_bucket"] = bucket_stoch_k(combined2)
    if "ema_stack_bias" in combined2.columns:
        combined2["ema_num"] = pd.to_numeric(combined2["ema_stack_bias"], errors="coerce")
    if "vpin_score" in combined2.columns:
        combined2["vpin_num"] = pd.to_numeric(combined2["vpin_score"], errors="coerce")
    if "structure_bias" in combined2.columns:
        combined2["struct_num"] = pd.to_numeric(combined2["structure_bias"], errors="coerce")
    if "funding_bias" in combined2.columns:
        combined2["fund_num"] = pd.to_numeric(combined2["funding_bias"], errors="coerce")
    if "vol_score" in combined2.columns:
        combined2["vol_num"] = pd.to_numeric(combined2["vol_score"], errors="coerce")

    # Build feature set for cross interactions
    # Feature-value pairs to iterate
    feature_vals = []
    if "ct_bucket" in combined2.columns:
        for v in combined2["ct_bucket"].dropna().unique():
            feature_vals.append(("ct_bucket", v, f"ct={v}"))
    if "cr_bucket" in combined2.columns:
        for v in combined2["cr_bucket"].dropna().unique():
            feature_vals.append(("cr_bucket", v, f"cr={v}"))
    if "sk_bucket" in combined2.columns:
        for v in combined2["sk_bucket"].dropna().unique():
            feature_vals.append(("sk_bucket", v, f"sk={v}"))
    if "ema_num" in combined2.columns:
        for v in [-1.0, 0.0, 1.0]:
            feature_vals.append(("ema_num", v, f"ema={int(v)}"))
    if "vpin_num" in combined2.columns:
        for v in combined2["vpin_num"].dropna().unique():
            if abs(v) <= 2:
                feature_vals.append(("vpin_num", v, f"vpin={v}"))
    if "struct_num" in combined2.columns:
        for v in [-1.0, 0.0, 1.0]:
            feature_vals.append(("struct_num", v, f"struct={int(v)}"))
    if "fund_num" in combined2.columns:
        for v in [-1.0, 0.0, 1.0]:
            feature_vals.append(("fund_num", v, f"fund={int(v)}"))

    interaction_results = []
    for i, (fcol1, fval1, fname1) in enumerate(feature_vals):
        for j, (fcol2, fval2, fname2) in enumerate(feature_vals):
            if j <= i:
                continue
            for side in ["yes", "no"]:
                try:
                    mask1 = combined2[fcol1] == fval1
                    mask2 = combined2[fcol2] == fval2
                    grp = combined2[mask1 & mask2 & (combined2["side"] == side)]
                    if len(grp) < 30:
                        continue
                    s = group_stats(grp)
                    if s is None or np.isnan(s["pval"]) or s["pval"] >= 0.05:
                        continue
                    interaction_results.append(dict(
                        combo=f"{fname1} + {fname2}", side=side, **s
                    ))
                except Exception:
                    continue

    interaction_results.sort(key=lambda r: r["flat_pnl"], reverse=True)
    top_n = interaction_results[:10]
    out(f"  Found {len(interaction_results)} significant combos (p<0.05, n>=30). Top 10 by flat_PnL:")
    out(f"  {'combo':<45} {'side':<5} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'PnL':>9} {'p-val':>7}")
    for row in top_n:
        pval_str = f"{row['pval']:.4f}" if not np.isnan(row['pval']) else "  n/a"
        out(f"  {row['combo']:<45} {row['side']:<5} {row['n']:>5} "
            f"{row['actual_wr']:>7.3f} {row['implied_wr']:>7.3f} {row['edge']:>+7.3f} "
            f"{row['flat_pnl']:>+9.1f} {pval_str:>7}")

    # Also bottom 10 (negative PnL = market is right, avoid these)
    bottom_n = interaction_results[-10:] if len(interaction_results) >= 10 else []
    if bottom_n:
        out(f"\n  Bottom 10 combos by flat_PnL (avoid these):")
        for row in bottom_n:
            pval_str = f"{row['pval']:.4f}" if not np.isnan(row['pval']) else "  n/a"
            out(f"  {row['combo']:<45} {row['side']:<5} {row['n']:>5} "
                f"{row['actual_wr']:>7.3f} {row['implied_wr']:>7.3f} {row['edge']:>+7.3f} "
                f"{row['flat_pnl']:>+9.1f} {pval_str:>7}")

    # =========================================================================
    # SECTION 7: Gate-specific performance (blocked trades)
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 7: GATE-SPECIFIC PERFORMANCE (which gates are losing edge)")
    out("=" * 80)
    out("For each gate that blocked trades, what was the actual WR of those blocked trades?")
    out("If actual_WR > implied_WR => gate is blocking GOOD trades (over-restrictive).")
    out("")

    if "gate_name" in blk.columns:
        gate_results = []
        for gate in blk["gate_name"].dropna().unique():
            grp = blk[blk["gate_name"] == gate]
            if len(grp) < 20:
                continue
            s = group_stats(grp)
            if s is None:
                continue
            s["gate"] = gate
            gate_results.append(s)

        gate_results.sort(key=lambda r: r["flat_pnl"], reverse=True)
        out(f"  {'gate':<35} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'PnL':>9} {'p-val':>7}")
        for row in gate_results:
            flag = " *** OVER-BLOCKING" if row["edge"] > 0.03 and row["n"] > 30 else ""
            flag2 = " *** CORRECT BLOCK" if row["edge"] < -0.03 and row["n"] > 30 else ""
            pval_str = f"{row['pval']:.4f}" if not np.isnan(row['pval']) else "  n/a"
            out(f"  {row['gate']:<35} {row['n']:>5} {row['actual_wr']:>7.3f} "
                f"{row['implied_wr']:>7.3f} {row['edge']:>+7.3f} "
                f"{row['flat_pnl']:>+9.1f} {pval_str:>7}{flag}{flag2}")

    # =========================================================================
    # SECTION 8: Asset-level summary
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 8: ASSET-LEVEL SUMMARY")
    out("=" * 80)

    for src_label, src_filter in [("ALL", None), ("EXECUTED", "executed"), ("BLOCKED", "blocked")]:
        if src_filter:
            df_src = combined[combined["source"] == src_filter]
        else:
            df_src = combined
        out(f"\n  {src_label}:")
        out(f"  {'asset':<8} {'side':<6} {'n':>5} {'act_WR':>7} {'imp_WR':>7} {'edge':>7} {'flat_PnL':>9}")
        for asset in ["BTC", "ETH", "SOL"]:
            for side in ["yes", "no"]:
                grp = df_src[(df_src["asset"] == asset) & (df_src["side"] == side)] if "asset" in df_src.columns else pd.DataFrame()
                if len(grp) < 5:
                    continue
                s = group_stats(grp)
                if s is None:
                    continue
                flag = " ***" if abs(s["edge"]) > 0.05 and s["n"] > 30 else ""
                out(f"  {asset:<8} {side:<6} {s['n']:>5} {s['actual_wr']:>7.3f} "
                    f"{s['implied_wr']:>7.3f} {s['edge']:>+7.3f} {s['flat_pnl']:>+9.1f}{flag}")

    # =========================================================================
    # SECTION 9: KEY FINDINGS SUMMARY
    # =========================================================================
    out("\n" + "=" * 80)
    out("SECTION 9: TOP ACTIONABLE MISPRICINGS (SUMMARY)")
    out("=" * 80)

    # Collect all edge findings
    all_findings = []

    # From pm calibration (combined)
    for pm_label in pm_labels:
        for side in ["yes", "no"]:
            grp = combined[(combined["pm_bin"] == pm_label) & (combined["side"] == side)]
            if len(grp) < 30:
                continue
            s = group_stats(grp)
            if s is None or np.isnan(s["pval"]) or s["pval"] >= 0.10:
                continue
            if abs(s["edge"]) > 0.04:
                all_findings.append(dict(
                    source="pm_calibration",
                    label=f"pm={pm_label} side={side}",
                    **s
                ))

    # From interactions
    for row in interaction_results[:30]:
        if row["flat_pnl"] > 50:
            all_findings.append(dict(
                source="interaction",
                label=f"{row['combo']} side={row['side']}",
                **{k: v for k, v in row.items() if k not in ("combo", "side")}
            ))

    # From gate analysis
    if "gate_name" in blk.columns:
        for gate in blk["gate_name"].dropna().unique():
            grp = blk[blk["gate_name"] == gate]
            s = group_stats(grp)
            if s is None or s["n"] < 30 or np.isnan(s["pval"]) or s["pval"] >= 0.10:
                continue
            if s["edge"] > 0.04:  # gate is over-blocking
                all_findings.append(dict(
                    source=f"gate:{gate}",
                    label=f"Gate={gate} (OVER-BLOCKING blocked trades)",
                    **s
                ))

    # Sort by flat_pnl
    all_findings.sort(key=lambda r: r["flat_pnl"], reverse=True)

    out("\n  TOP 10 ACTIONABLE MISPRICINGS:")
    out(f"  (Ranked by flat_PnL @ $10/trade. p<0.10, n>=30, |edge|>4%)")
    out("")

    for rank, finding in enumerate(all_findings[:10], 1):
        pval_str = f"{finding['pval']:.4f}" if not np.isnan(finding['pval']) else "n/a"
        out(f"  #{rank} [{finding['source']}]")
        out(f"     {finding['label']}")
        out(f"     n={finding['n']}  actual_WR={finding['actual_wr']:.3f}  implied_WR={finding['implied_wr']:.3f}")
        out(f"     edge={finding['edge']:+.3f}  flat_PnL=${finding['flat_pnl']:+.1f}  p={pval_str}")
        out("")

    # Summary statistics
    out("\n" + "=" * 80)
    out("OVERALL STATISTICS")
    out("=" * 80)
    s_all = group_stats(combined)
    if s_all:
        out(f"  ALL TRADES (exec+blocked): n={s_all['n']}  actual_WR={s_all['actual_wr']:.3f}  "
            f"implied_WR={s_all['implied_wr']:.3f}  edge={s_all['edge']:+.3f}  "
            f"flat_PnL=${s_all['flat_pnl']:+.1f}  p={s_all['pval']:.4f}")
    s_exec = group_stats(combined[combined["source"] == "executed"])
    if s_exec:
        out(f"  EXECUTED ONLY:             n={s_exec['n']}  actual_WR={s_exec['actual_wr']:.3f}  "
            f"implied_WR={s_exec['implied_wr']:.3f}  edge={s_exec['edge']:+.3f}  "
            f"flat_PnL=${s_exec['flat_pnl']:+.1f}  p={s_exec['pval']:.4f}")
    s_blk = group_stats(combined[combined["source"] == "blocked"])
    if s_blk:
        out(f"  BLOCKED ONLY:              n={s_blk['n']}  actual_WR={s_blk['actual_wr']:.3f}  "
            f"implied_WR={s_blk['implied_wr']:.3f}  edge={s_blk['edge']:+.3f}  "
            f"flat_PnL=${s_blk['flat_pnl']:+.1f}  p={s_blk['pval']:.4f}")

    # Save output
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\n\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
