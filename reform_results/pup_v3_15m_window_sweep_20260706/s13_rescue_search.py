"""
S13 -- Mandatory rescue search for the two new p_up_v3 HMM gates
(hmm_pup_v3_rising_yes_gate, hmm_pup_v3_crashing_no_gate), using the same
combined-archive real-trade population as the original backfill (2,995
trades, Apr-Jul 2026). For each gate's BLOCKED population, sweep candidate
signals for a subset that's actually a winner (should be rescued/allowed
through) rather than blocked outright.

Candidate signals restricted to what's present across the whole archive
chain where possible; a few MTF/microstructure signals only exist in the
newer files (smaller effective n, noted per-feature).
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
NEEDED_BASE = ["logged_at", "decision", "side", "would_win", "would_pnl", "p_market", "contract_ticker"]
CANDIDATES = ["composite_trend", "composite_rev", "ema_stack_bias", "vol_score", "funding_bias",
              "stoch_k", "stoch_bias", "structure_bias", "vpin_score", "obi_score",
              "vwap_stretch_score", "liq_score", "liq_bias", "bp_1h", "ou_theta",
              "hurst_exponent", "kalman_velocity", "kalman_residual", "autocorr1_30"]

CHAIN = [
    "results/paper_trades_archive_20260415_1342_precal.csv",
    "results/paper_trades_archive_20260525_1432_pre_branched_drift.csv",
    "results/paper_trades_pre_regime_pup_20260616.csv",
    "results/paper_trades.csv",
]


def load_chain():
    frames = []
    for p in CHAIN:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        use = [c for c in NEEDED_BASE + CANDIDATES if c in cols]
        if "asset" in cols:
            use = use + ["asset"]
        df = pd.read_csv(p, usecols=use, low_memory=False)
        if "asset" in df.columns:
            df = df[df["asset"] == "BTC"]
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    return full.sort_values("logged_at_parsed")


raw = load_chain()
states = pd.read_parquet(f"{OUT}/pup_v3_hmm_states.parquet")
bar_index = states.index
state3_series = states["state"].map({0: "neutral", 1: "rising", 2: "neutral", 3: "crashing"})


def lookup_state(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (logged_at - bar_index[idx]) > pd.Timedelta(hours=2):
        return np.nan
    return state3_series.iloc[idx]


taken = raw[raw["decision"] == "trade"].copy()
taken["state3"] = taken["logged_at_parsed"].apply(lookup_state)
taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")
taken["side"] = taken["side"].str.lower()
taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
taken["yw"] = taken["logged_at_parsed"].dt.isocalendar().year.astype(str) + "-W" + taken["week"].astype(str).str.zfill(2)
covered = taken.dropna(subset=["state3", "would_pnl"]).copy()
covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])


def sweep(pop, label):
    print(f"\n{'='*80}\nRESCUE SEARCH: {label}  (blocked pop n={len(pop)}, base WR={pop['would_win'].mean():.3f}, "
          f"base BE={pop['be'].mean():.3f}, base PnL=${pop['would_pnl'].sum():.2f})\n{'='*80}")
    candidates_found = []
    for feat in CANDIDATES:
        if feat not in pop.columns:
            continue
        vals = pd.to_numeric(pop[feat], errors="coerce")
        sub = pop[vals.notna()].copy()
        sub["_f"] = vals.dropna()
        if len(sub) < 15:
            continue
        # try quantile-based splits (above/below median, and above/below 25th/75th pctile)
        for q, qname in [(0.5, "median"), (0.25, "p25"), (0.75, "p75")]:
            thresh = sub["_f"].quantile(q)
            for direction, mask in [(">=", sub["_f"] >= thresh), ("<", sub["_f"] < thresh)]:
                rescued = sub[mask]
                remainder = sub[~mask]
                if len(rescued) < 10 or len(remainder) < 10:
                    continue
                r_wr = rescued["would_win"].mean()
                r_be = rescued["be"].mean()
                r_edge = r_wr - r_be
                rem_wr = remainder["would_win"].mean()
                rem_be = remainder["be"].mean()
                rem_edge = rem_wr - rem_be
                # a real rescue candidate: rescued subset clearly +EV, remainder clearly -EV
                # (genuine separation, not just noise), and rescued isn't dominated by 1-2 weeks
                if r_edge > 0.10 and rem_edge < -0.05:
                    wk_pnl = rescued.groupby("yw")["would_pnl"].sum()
                    worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
                    candidates_found.append({
                        "feature": feat, "split": f"{direction}{thresh:.3f}({qname})",
                        "n_rescued": len(rescued), "wr_rescued": r_wr, "edge_rescued": r_edge,
                        "pnl_rescued": rescued["would_pnl"].sum(), "n_weeks": len(wk_pnl),
                        "worst_wk_share": worst_share,
                        "n_remainder": len(remainder), "wr_remainder": rem_wr, "edge_remainder": rem_edge,
                        "pnl_remainder": remainder["would_pnl"].sum(),
                    })
    if not candidates_found:
        print("  No candidate clears the bar (rescued edge>+10pp AND remainder edge<-5pp).")
        return pd.DataFrame()
    cf = pd.DataFrame(candidates_found).sort_values("pnl_rescued", ascending=False)
    print(cf.round(3).to_string(index=False))
    return cf


rising_yes = covered[(covered["state3"] == "rising") & (covered["side"] == "yes")]
crashing_no = covered[(covered["state3"] == "crashing") & (covered["side"] == "no")]

r1 = sweep(rising_yes, "hmm_pup_v3_rising_yes_gate blocked population")
r2 = sweep(crashing_no, "hmm_pup_v3_crashing_no_gate blocked population")
