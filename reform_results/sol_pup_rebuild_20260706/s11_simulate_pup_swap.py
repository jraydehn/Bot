"""
S11 -- Simulate swapping SOL hourly's base p_up source: composite lookup_p_up
(current, table-based) vs sol_p_up_v1 (honest ML rebuild), feeding the SAME
score_to_p_model log-normal structure (same k_drift=DRIFT_MULTIPLIER["SOL"]=0.20).
Isolates the effect of the p_up SOURCE, holding the drift mechanism fixed.

Scope / simplifications (disclosed, not hidden):
  - vol_factor approximated as 1.0 (the live vol-regime scaler needs live_1m
    data per scan cycle, not practical to reconstruct for 146k historical
    rows). Applies IDENTICALLY to both OLD and NEW, so it doesn't bias the
    relative comparison, only the absolute p_model level.
  - Decision threshold: flat MIN_NET_EDGE=0.01 (the universal base gate),
    NOT the full tiered/gate-stack pipeline (dozens of SOL-specific micro
    gates in paper_trade_runner.py). SOL uniquely has "Gate PM not applied"
    per decision.py's own comments, so it has fewer asset-specific gates
    than BTC/ETH -- this simplification captures a larger share of the real
    pipeline for SOL than it would elsewhere, but is still NOT the full
    live decision path. Both OLD and NEW pass through the same simplified
    gate, so the comparison is fair even though absolute trade counts would
    differ from the real runner.
  - Uniform $50 flat bet size per qualifying trade (not real Kelly). Isolates
    "did the new p_up select better trades" (WR/edge) from sizing effects.
    Real Kelly sizing (proportional to edge, SOL kelly mult 0.50, caps) would
    change absolute PnL magnitudes but not the WR/edge signal.
  - Per feedback_sim_methodology: uses the FULL scanned archive (all
    candidates, not just taken trades) and real resolved outcomes already
    logged in sol_scan_archive.csv (145,710/146,019 resolved).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
from composite_scorer import lookup_p_up, score_to_p_model, DRIFT_MULTIPLIER
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE

OUT = "reform_results/sol_pup_rebuild_20260706"
rng = np.random.default_rng(99)

# ── load scan archive ────────────────────────────────────────────────────
df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False)
df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
need = ["logged_at_parsed", "spot", "strike", "p_market", "tau_minutes", "vol_eff",
        "composite_trend", "composite_rev", "resolved_yes"]
df = df.dropna(subset=need).copy()
df = df[(df["tau_minutes"] > 0) & (df["vol_eff"] > 0) & (df["p_market"] > 0) & (df["p_market"] < 1)]
print(f"usable scanned rows: {len(df)}  ({df['logged_at_parsed'].min()} -> {df['logged_at_parsed'].max()})")

# ── join sol_p_up_v1 causally (same pattern as s6 backfill) ─────────────
wf = pd.read_parquet(f"{OUT}/wf_preds_A.parquet").dropna(subset=["p"])
bar_index = wf.index
p_series = wf["p"]


def lookup_p_new(ts):
    idx = bar_index.searchsorted(ts, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (ts - bar_index[idx]) > pd.Timedelta(hours=2):
        return np.nan
    return float(p_series.iloc[idx])


df["p_up_new"] = df["logged_at_parsed"].apply(lookup_p_new)
before = len(df)
df = df.dropna(subset=["p_up_new"])
print(f"rows with sol_p_up_v1 coverage: {len(df)}/{before}")

# ── compute p_up_old via the REAL live lookup table function ────────────
df["p_up_old"] = df.apply(lambda r: lookup_p_up(int(r["composite_trend"]), int(r["composite_rev"]), asset="SOL"), axis=1)

# ── sigma_tau (vol_factor=1.0 approximation, disclosed above) ───────────
df["sigma_tau"] = df["vol_eff"] * np.sqrt(df["tau_minutes"])

K_DRIFT = DRIFT_MULTIPLIER["SOL"]
print(f"SOL k_drift (unchanged in both arms): {K_DRIFT}")


def p_model_row(r, p_up_col):
    return score_to_p_model(int(r["composite_trend"]), int(r["composite_rev"]),
                            r["spot"], r["strike"], r["sigma_tau"],
                            asset="SOL", p_up_override=r[p_up_col])


df["p_model_old"] = df.apply(lambda r: p_model_row(r, "p_up_old"), axis=1)
df["p_model_new"] = df.apply(lambda r: p_model_row(r, "p_up_new"), axis=1)


def decide(r, p_model_col):
    pm = r["p_market"]
    pmodel = r[p_model_col]
    fee = kalshi_fee(pm)
    edge_yes = pmodel - pm
    edge_no = pm - pmodel
    net_yes = edge_yes - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD
    net_no = edge_no - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD
    if max(net_yes, net_no) < MIN_NET_EDGE:
        return pd.Series({"decision": False, "side": "", "net_edge": max(net_yes, net_no)})
    side = "yes" if net_yes >= net_no else "no"
    return pd.Series({"decision": True, "side": side, "net_edge": net_yes if side == "yes" else net_no})


for label, pcol in [("old", "p_model_old"), ("new", "p_model_new")]:
    res = df.apply(lambda r: decide(r, pcol), axis=1)
    df[f"trade_{label}"] = res["decision"]
    df[f"side_{label}"] = res["side"]
    df[f"net_edge_{label}"] = res["net_edge"].astype(float)


def outcome_and_pnl(r, side_col):
    side = r[side_col]
    if not side:
        return pd.Series({"would_win": np.nan, "would_pnl": np.nan, "be": np.nan})
    pm = r["p_market"]
    price_side = pm if side == "yes" else (1 - pm)
    won = (side == "yes" and r["resolved_yes"] == 1) or (side == "no" and r["resolved_yes"] == 0)
    contracts = 50.0 / price_side
    pnl = contracts * (1 - price_side) if won else -50.0
    return pd.Series({"would_win": bool(won), "would_pnl": pnl, "be": price_side})


for label in ["old", "new"]:
    res = df.apply(lambda r: outcome_and_pnl(r, f"side_{label}"), axis=1)
    df[f"would_win_{label}"] = res["would_win"]
    df[f"would_pnl_{label}"] = res["would_pnl"]
    df[f"be_{label}"] = res["be"]

df["yw"] = df["logged_at_parsed"].dt.strftime("%G-W%V")


def report(label):
    sub = df[df[f"trade_{label}"]]
    n = len(sub)
    if n == 0:
        print(f"{label}: n=0")
        return
    wr = sub[f"would_win_{label}"].mean()
    be = sub[f"be_{label}"].mean()
    pnl = sub[f"would_pnl_{label}"].sum()
    wk = sub.groupby("yw")[f"would_pnl_{label}"].sum()
    print(f"{label:5s}: n={n:5d}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  "
          f"PnL(flat $50/bet)=${pnl:9.2f}  weeks={len(wk)}  pos_weeks={(wk>0).mean():.2f}")
    return sub


print("\n=== AGGREGATE: OLD (current lookup_p_up) vs NEW (sol_p_up_v1) ===")
sub_old = report("old")
sub_new = report("new")

# ── overlap / what changed ────────────────────────────────────────────────
both = df["trade_old"] & df["trade_new"]
only_old = df["trade_old"] & ~df["trade_new"]
only_new = ~df["trade_old"] & df["trade_new"]
print(f"\ntraded by BOTH: {both.sum()}   ONLY old: {only_old.sum()}   ONLY new: {only_new.sum()}")

if only_old.sum() > 0:
    s = df[only_old]
    print(f"  only-OLD trades (would be REMOVED by swap): WR={s['would_win_old'].mean():.3f} "
          f"BE={s['be_old'].mean():.3f} PnL=${s['would_pnl_old'].sum():.2f}")
if only_new.sum() > 0:
    s = df[only_new]
    print(f"  only-NEW trades (would be ADDED by swap):    WR={s['would_win_new'].mean():.3f} "
          f"BE={s['be_new'].mean():.3f} PnL=${s['would_pnl_new'].sum():.2f}")
if both.sum() > 0:
    s = df[both]
    same_side = (s["side_old"] == s["side_new"]).mean()
    print(f"  BOTH-trades: same side chosen {same_side:.1%} of the time")

# ── bootstrap: is NEW's PnL improvement over OLD significant? ───────────
def trade_boot(edges, n_boot=5000):
    e = np.asarray(edges); n = len(e)
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()


for label, sub in [("old", sub_old), ("new", sub_new)]:
    if sub is None or len(sub) == 0:
        continue
    edges = (sub[f"would_win_{label}"].astype(float) - sub[f"be_{label}"]).values
    m, lo, hi, p = trade_boot(edges)
    print(f"{label}: trade-level edge bootstrap mean={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] P(edge<=0)={p:.4f}")

df.to_csv(f"{OUT}/sim_pup_swap_full.csv", index=False)
print(f"\nsaved {OUT}/sim_pup_swap_full.csv")
