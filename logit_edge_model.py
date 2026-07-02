"""
logit_edge_model.py

Trains a logistic regression model directly on Kalshi microstructure features
to predict P(win), bypassing the lognormal p_yes_model entirely.

Architecture:
  - pm is a feature (the market's base estimate)
  - Microstructure features (funding, EMA, structure, stoch, etc.) provide
    the edge adjustment the market doesn't know about
  - win = 1 if resolved_yes==1 for YES; resolved_yes==0 for NO
  - edge = P(win) - pm (for YES) or P(win) - (1-pm) (for NO)

Train:  results/blocked_trades.csv  (99k obs, May 6–18)
Test:   paper_trades*.csv filtered to logged_at < 2026-05-06  (Apr 15 – May 5)
        + March archive trades (limited features)

Output: results/logit_edge_model_report.txt
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")

FLAT    = 10.0
OUT     = Path("results/logit_edge_model_report.txt")
EDGE_TH = 0.04   # minimum predicted edge to bet

FEATURES = [
    "pm",
    "offset_pct",
    "tau_minutes",
    "ema_stack_bias",
    "composite_trend",
    "composite_rev",
    "composite_p_up",
    "stoch_k",
    "vwap_stretch",
    "vol_score",
    "vpin_score",
    "funding_bias",
    "structure_bias",
    "side_yes",         # 1=YES bet, 0=NO bet
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pnl(side, pm, resolved_yes):
    if side == "yes":
        return (1-pm)*FLAT if resolved_yes == 1 else -pm*FLAT
    return pm*FLAT if resolved_yes == 0 else -(1-pm)*FLAT


def prep(df, source_label, clip_features=True):
    """Normalise a raw trades dataframe into model-ready rows."""
    df = df.copy()
    # rename p_market → pm if needed
    if "p_market" in df.columns and "pm" not in df.columns:
        df["pm"] = df["p_market"]
    if "vwap_stretch_score" in df.columns and "vwap_stretch" not in df.columns:
        df["vwap_stretch"] = df["vwap_stretch_score"]
    for c in FEATURES + ["resolved_yes", "side", "asset", "contract_ticker"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side_yes"] = (df["side"] == "yes").astype(float)
    # win: did we win THIS side?
    df["win"] = np.where(
        df["side"] == "yes",
        (df["resolved_yes"] == 1).astype(float),
        (df["resolved_yes"] == 0).astype(float),
    )
    # market-implied win prob
    df["pm_win"] = np.where(df["side"] == "yes", df["pm"], 1 - df["pm"])
    df["source"] = source_label
    if clip_features:
        df["composite_trend"] = df["composite_trend"].clip(-5, 5)
        df["composite_rev"]   = df["composite_rev"].clip(-6, 10)
        df["composite_p_up"]  = df["composite_p_up"].clip(0.2, 0.8)
        df["offset_pct"]      = df["offset_pct"].clip(-0.5, 0.5)
        df["tau_minutes"]     = df["tau_minutes"].clip(5, 120)
        df["stoch_k"]         = df["stoch_k"].clip(0, 100)
    return df


def load_blocked():
    df = pd.read_csv("results/blocked_trades.csv", low_memory=False)
    df = df[df["resolved_yes"].notna()].copy()
    df["asset"] = df["asset"].fillna("BTC")
    return prep(df, "blocked")


def load_executed(path, asset):
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["decision"] == "trade") & df["resolved_yes"].notna()].copy()
    df["asset"] = asset
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    return prep(df, "executed")


def compute_pnl_series(df, side_col, pm_col, ry_col):
    return [pnl(r[side_col], r[pm_col], r[ry_col]) for _, r in df.iterrows()]


lines = []
def w(*a): lines.append(" ".join(str(x) for x in a))
def ws(): lines.append("")
def wh(t): ws(); lines.append("=" * 72); lines.append(t); lines.append("=" * 72)
def wh2(t): ws(); lines.append("-" * 60); lines.append(t); lines.append("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
blk = load_blocked()
btc = load_executed("results/paper_trades.csv", "BTC")
eth = load_executed("results/paper_trades_eth.csv", "ETH")
sol = load_executed("results/paper_trades_sol.csv", "SOL")
exec_all = pd.concat([btc, eth, sol], ignore_index=True)

# Train: blocked trades (May 6-18)
# Test:  executed trades before May 6 (Apr 15 – May 5)
exec_test  = exec_all[exec_all["logged_at"] < pd.Timestamp("2026-05-06", tz="UTC")].copy()
exec_may   = exec_all[exec_all["logged_at"] >= pd.Timestamp("2026-05-06", tz="UTC")].copy()

# March archives (limited features — used for reduced-feature model test)
march_dfs = []
for f in [
    "results/paper_trades_archive_20260323_003206.csv",
    "results/paper_trades_archive_20260325_090712.csv",
    "results/paper_trades_archive_20260330_103000.csv",
    "results/paper_trades_archive_20260405_1633pdt.csv",
    "results/paper_trades_archive_20260407_122844.csv",
]:
    try:
        df = pd.read_csv(f"results/{Path(f).name}", low_memory=False)
        df = df[(df["decision"] == "trade") & df["resolved_yes"].notna()].copy()
        df["asset"] = df.get("asset", pd.Series(["BTC"]*len(df), index=df.index))
        df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
        march_dfs.append(prep(df, "march"))
    except Exception:
        pass
march = pd.concat(march_dfs, ignore_index=True) if march_dfs else pd.DataFrame()

print(f"  Train (blocked):       {len(blk):>6} obs")
print(f"  Test  (Apr15-May5):    {len(exec_test):>6} obs  "
      f"({(exec_test['logged_at'].min()).strftime('%Y-%m-%d')} – "
      f"{(exec_test['logged_at'].max()).strftime('%Y-%m-%d')})")
print(f"  Test  (May6-18 exec):  {len(exec_may):>6} obs")
print(f"  Test  (Mar archives):  {len(march):>6} obs")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
wh("LOGISTIC REGRESSION EDGE MODEL — TRAINING REPORT")
w("Train: blocked_trades.csv (May 6–18, all gates, all assets)")
w("Test:  executed trades (Apr 15 – May 5) — temporal holdout")
ws()

train = blk.dropna(subset=FEATURES + ["win"]).copy()
X_train = train[FEATURES].values
y_train = train["win"].values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logit",  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
])
pipe.fit(X_train, y_train)

w(f"Training n={len(train)}  (YES={int((train['side']=='yes').sum())}  "
  f"NO={int((train['side']=='no').sum())})")
w(f"Training log-loss: {log_loss(y_train, pipe.predict_proba(X_train)[:,1]):.4f}")
w(f"Training brier:    {brier_score_loss(y_train, pipe.predict_proba(X_train)[:,1]):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────
wh2("Feature Coefficients (log-odds, standardised)")
coef = pipe.named_steps["logit"].coef_[0]
intercept = pipe.named_steps["logit"].intercept_[0]
coef_df = pd.DataFrame({"feature": FEATURES, "coef": coef})
coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
w(f"  {'Feature':>20}  {'Coef':>8}")
w("  " + "-"*32)
for _, r in coef_df.iterrows():
    flag = " ***" if abs(r["coef"]) > 0.3 else ""
    w(f"  {r['feature']:>20}  {r['coef']:>+8.4f}{flag}")
w(f"  {'(intercept)':>20}  {intercept:>+8.4f}")
ws()
w("Interpretation: positive coef = feature increases P(win for YES or NO given side_yes encoding)")
w("  side_yes>0 = YES bets naturally have higher p(win) when pm is high")
w("  funding_bias: negative coef = fund=-1 increases win prob (bearish funding = YES wins)")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET (Apr 15 – May 5 executed trades)
# ─────────────────────────────────────────────────────────────────────────────
wh("TEST RESULTS: Apr 15 – May 5 Executed Trades")

def evaluate(test_df, label):
    test = test_df.dropna(subset=FEATURES + ["win", "pm"]).copy()
    if len(test) == 0:
        w(f"  {label}: no data")
        return
    X_test = test[FEATURES].values
    test["p_win_logit"] = pipe.predict_proba(X_test)[:, 1]
    # Edge: how much better does model think we'll do vs market implied
    test["edge_logit"] = test["p_win_logit"] - test["pm_win"]

    # Current model PnL: what the model actually did (all these are executed = model said yes)
    test["pnl_actual"] = [pnl(r["side"], r["pm"], r["resolved_yes"]) for _, r in test.iterrows()]

    # Logit model: only bet when edge_logit > EDGE_TH
    mask_bet = test["edge_logit"] >= EDGE_TH
    mask_pass = ~mask_bet

    wh2(f"  {label} (n={len(test)})")
    w(f"  {'':30}  {'n':>6}  {'WR':>7}  {'BE':>7}  {'Edge':>7}  {'PnL':>9}")
    w("  " + "-"*70)

    def row_stats(sub, lbl):
        if len(sub) == 0:
            w(f"  {lbl:30}  {'0':>6}")
            return
        wr   = sub["win"].mean()
        be   = sub["pm_win"].mean()
        pnl_ = sub["pnl_actual"].sum()
        try:
            p = binomtest(int(round(wr*len(sub))), len(sub), be, alternative="two-sided").pvalue
        except Exception:
            p = float("nan")
        w(f"  {lbl:30}  {len(sub):>6}  {wr:.1%}  {be:.1%}  {wr-be:>+.1%}  ${pnl_:>+.0f}  p={p:.3f}")

    row_stats(test, "All executed (current model)")
    row_stats(test[mask_bet], "Logit says BET (edge>=0.04)")
    row_stats(test[mask_pass], "Logit says PASS (would block)")

    # What does logit accept that current model also took?
    ws()
    w("  By side:")
    for side in ["yes", "no"]:
        ss = test[test["side"] == side]
        row_stats(ss, f"  Current {side.upper()}")
        row_stats(ss[ss["edge_logit"] >= EDGE_TH], f"  Logit BET {side.upper()}")

    # Calibration of logit predictions
    ws()
    w("  Logit prediction calibration (p_win_logit vs actual):")
    test["p_bucket"] = pd.cut(test["p_win_logit"], bins=np.arange(0, 1.05, 0.10))
    w(f"  {'p_bucket':>15}  {'n':>5}  {'pred':>7}  {'actual':>7}  {'diff':>7}")
    for bkt, grp in test.groupby("p_bucket", observed=True):
        if len(grp) < 10: continue
        pred = grp["p_win_logit"].mean()
        act  = grp["win"].mean()
        w(f"  {str(bkt):>15}  {len(grp):>5}  {pred:.3f}  {act:.3f}  {act-pred:>+.3f}")
    w(f"  Brier score: {brier_score_loss(test['win'], test['p_win_logit']):.4f}"
      f"  (market brier: {brier_score_loss(test['win'], test['pm_win']):.4f})")

    # Edge distribution of logit predictions
    ws()
    w("  Logit edge distribution:")
    for lo, hi in [(-1,-0.1),(-0.1,-0.04),(-0.04,0),(0,0.04),(0.04,0.1),(0.1,0.2),(0.2,1)]:
        grp = test[(test["edge_logit"]>=lo)&(test["edge_logit"]<hi)]
        if len(grp) < 5: continue
        wr = grp["win"].mean(); be = grp["pm_win"].mean()
        pnl_ = grp["pnl_actual"].sum()
        w(f"    edge∈[{lo:+.2f},{hi:+.2f}):  n={len(grp):>5}  WR={wr:.1%}  BE={be:.1%}  "
          f"edge_actual={wr-be:>+.1%}  PnL=${pnl_:>+.0f}")

    return test


test_apr = exec_test.copy()
test_full = evaluate(test_apr, "Apr 15 – May 5 (all assets)")

# Per-asset breakdown
for asset in ["BTC", "ETH", "SOL"]:
    sub = test_apr[test_apr["asset"] == asset]
    if len(sub) > 20:
        evaluate(sub, f"{asset} Apr 15–May5")

# ─────────────────────────────────────────────────────────────────────────────
# ALSO EVALUATE ON MAY 6-18 EXECUTED (in-sample period)
# ─────────────────────────────────────────────────────────────────────────────
wh("IN-SAMPLE CHECK: May 6–18 Executed Trades (same period as train)")
w("NOTE: training set is BLOCKED trades; these are EXECUTED trades in same period.")
w("Not data leakage, but same calendar window — check for structural drift.")
evaluate(exec_may, "May 6–18 executed (all assets)")

# ─────────────────────────────────────────────────────────────────────────────
# MARCH ARCHIVE TEST (REDUCED FEATURES)
# ─────────────────────────────────────────────────────────────────────────────
if len(march) > 50:
    wh("MARCH ARCHIVE TEST (reduced features: pm, offset_pct, tau, structure_bias only)")
    w("These archives lack composite/stoch features — testing with reduced feature set.")

    FEAT_RED = ["pm", "offset_pct", "tau_minutes", "structure_bias", "side_yes"]

    train_red = blk.dropna(subset=FEAT_RED + ["win"]).copy()
    pipe_red = Pipeline([
        ("scaler", StandardScaler()),
        ("logit",  LogisticRegression(max_iter=500, C=1.0)),
    ])
    pipe_red.fit(train_red[FEAT_RED].values, train_red["win"].values)

    test_mar = march.dropna(subset=FEAT_RED + ["win", "pm"]).copy()
    test_mar["p_win_logit"] = pipe_red.predict_proba(test_mar[FEAT_RED].values)[:, 1]
    test_mar["edge_logit"]  = test_mar["p_win_logit"] - test_mar["pm_win"]
    test_mar["pnl_actual"]  = [pnl(r["side"], r["pm"], r["resolved_yes"]) for _, r in test_mar.iterrows()]

    wh2(f"  March archives (n={len(test_mar)})")
    mask = test_mar["edge_logit"] >= EDGE_TH
    for lbl, sub in [("All executed", test_mar), ("Logit BET", test_mar[mask]),
                     ("Logit PASS", test_mar[~mask])]:
        if len(sub) == 0: continue
        wr = sub["win"].mean(); be = sub["pm_win"].mean()
        pnl_ = sub["pnl_actual"].sum()
        try:
            p = binomtest(int(round(wr*len(sub))), len(sub), be, alternative="two-sided").pvalue
        except Exception: p = float("nan")
        w(f"  {lbl:25}  n={len(sub):>4}  WR={wr:.1%}  BE={be:.1%}  "
          f"edge={wr-be:>+.1%}  PnL=${pnl_:>+.0f}  p={p:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# WHAT DOES THE LOGIT MODEL THINK IS EDGE-POSITIVE IN BLOCKED TRADES?
# ─────────────────────────────────────────────────────────────────────────────
wh("LOGIT MODEL: TOP EDGE OPPORTUNITIES IN BLOCKED TRADES (what to unblock)")
w("Blocked trades where logit predicts edge >= 0.06 — actual WR vs implied")

blk_feat = blk.dropna(subset=FEATURES + ["win", "pm"]).copy()
blk_feat["p_win_logit"] = pipe.predict_proba(blk_feat[FEATURES].values)[:, 1]
blk_feat["edge_logit"]  = blk_feat["p_win_logit"] - blk_feat["pm_win"]
blk_feat["pnl_actual"]  = [pnl(r["side"], r["pm"], r["resolved_yes"]) for _, r in blk_feat.iterrows()]

for th in [0.10, 0.08, 0.06, 0.04]:
    sub = blk_feat[blk_feat["edge_logit"] >= th]
    if len(sub) == 0: continue
    wr = sub["win"].mean(); be = sub["pm_win"].mean()
    pnl_ = sub["pnl_actual"].sum()
    try:
        p = binomtest(int(round(wr*len(sub))), len(sub), be, alternative="two-sided").pvalue
    except Exception: p = float("nan")
    w(f"  logit_edge>={th:.2f}: n={len(sub):>6}  WR={wr:.1%}  BE={be:.1%}  "
      f"edge_actual={wr-be:>+.1%}  PnL=${pnl_:>+.0f}  p={p:.3f}")

ws()
w("Top gate breakdown (what gates are blocking the logit's best opportunities):")
sub_high = blk_feat[blk_feat["edge_logit"] >= 0.06]
if "gate_name" in sub_high.columns:
    for gate, cnt in sub_high["gate_name"].value_counts().head(10).items():
        grp = sub_high[sub_high["gate_name"] == gate]
        wr = grp["win"].mean(); be = grp["pm_win"].mean()
        pnl_ = grp["pnl_actual"].sum()
        w(f"  {gate:35}  n={cnt:>5}  WR={wr:.1%}  BE={be:.1%}  "
          f"edge={wr-be:>+.1%}  PnL=${pnl_:>+.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# LOGIT EDGE THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────
wh("EDGE THRESHOLD SWEEP (test Apr15-May5, logit model)")
w("Choosing the right edge_threshold for the logit model vs current model.")
ws()

test_eval = test_full if test_full is not None else test_apr.dropna(subset=FEATURES)
if test_eval is None or len(test_eval) == 0:
    test_eval = test_apr.dropna(subset=FEATURES + ["win"]).copy()
    test_eval["p_win_logit"] = pipe.predict_proba(test_eval[FEATURES].values)[:, 1]
    test_eval["edge_logit"]  = test_eval["p_win_logit"] - test_eval["pm_win"]
    test_eval["pnl_actual"]  = [pnl(r["side"], r["pm"], r["resolved_yes"]) for _, r in test_eval.iterrows()]

current_pnl = test_eval["pnl_actual"].sum() if "pnl_actual" in test_eval.columns else 0
w(f"  Current model (all executed Apr-May5): n={len(test_eval)}, PnL=${current_pnl:+.0f}")
ws()
w(f"  {'Threshold':>10}  {'n_bet':>7}  {'WR':>7}  {'BE':>7}  {'Edge':>7}  {'PnL':>9}")
w("  " + "-"*60)
for th in [0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
    sub = test_eval[test_eval["edge_logit"] >= th] if "edge_logit" in test_eval.columns else pd.DataFrame()
    if len(sub) == 0: continue
    wr = sub["win"].mean(); be = sub["pm_win"].mean()
    pnl_ = sub["pnl_actual"].sum()
    try:
        p = binomtest(int(round(wr*len(sub))), len(sub), be, alternative="two-sided").pvalue
    except Exception: p = float("nan")
    flag = " <-- opt" if abs(th - EDGE_TH) < 0.001 else ""
    w(f"  {th:>10.2f}  {len(sub):>7}  {wr:.1%}  {be:.1%}  {wr-be:>+.1%}  ${pnl_:>+.0f}  p={p:.3f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nReport written to {OUT}")
