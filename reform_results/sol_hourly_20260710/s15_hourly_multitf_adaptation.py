"""
S15 -- proper hourly adaptation of SOL 15m's architecture, using hourly-scaled
timeframes instead of copying 15m-scale ones. SOL 15m's ratio of short:medium:
context signal windows is 5m:15m:1h (~1:3:12). Hourly contracts run up to
tau=60min (4x longer than 15m's tau<=15), so the same ratio scaled up maps to
15m:1h:4h -- that's the timeframe substitution the user asked for.

Same core design principles as 15m (see s14 + the architecture presentation):
  - z_score (geometric distance via vol-blended sigma_tau), NOT p_market
  - raw per-timeframe technicals (bp, stoch_k, body, dir, consec_dir, chg),
    NOT a pre-aggregated composite score
  - CalibratedClassifierCV wrapping the raw classifier
  - orthogonal microstructure folded in directly (vpin/obi/liq/ls_long_pct),
    same signals independently validated as carrying real info on 07-10

Uses the EXACT canonical indicator formulas from backfill_hmm_features.py
(buying_pressure, stoch_k, body_size, direction, consec_dir) so this matches
codebase convention rather than inventing new ones. Bars are shifted by their
own period before merge_asof so only genuinely CLOSED bars are ever visible
at each archive timestamp -- zero lookahead, same discipline as every prior
finding this investigation relied on.

Validated the same way as s14: ticker-grouped split, Brier + corr + $ sim
in the uncertain zone (p_market 0.35-0.65), checked on BOTH test and
validation sets before trusting anything.
"""
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
    BASE_CLF = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4,
                                            num_leaves=15, min_child_samples=30, reg_lambda=5.0,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    BASE_CLF = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=4,
                                                        min_samples_leaf=30, l2_regularization=5.0, random_state=42)


# ── canonical indicator formulas (from backfill_hmm_features.py) ────────────
def buying_pressure(df, n=14):
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    bp = (df["close"] - df["low"]) / hl
    return bp.rolling(n, min_periods=1).mean()

def stoch_k(df, k=14, smooth=3):
    lowest = df["low"].rolling(k, min_periods=1).min()
    highest = df["high"].rolling(k, min_periods=1).max()
    rng = (highest - lowest).replace(0, np.nan)
    raw_k = 100 * (df["close"] - lowest) / rng
    return raw_k.rolling(smooth, min_periods=1).mean()

def body_size(df):
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]).abs() / hl

def direction(df):
    return np.sign(df["close"] - df["open"])

def consec_dir(df, window=4):
    d = np.sign(df["close"] - df["open"])
    out = []
    for i in range(len(d)):
        if i < 1:
            out.append(0); continue
        w = d.iloc[max(0, i - window + 1):i + 1]
        if (w == 1).all(): out.append(int((w == 1).sum()))
        elif (w == -1).all(): out.append(-int((w == -1).sum()))
        else: out.append(0)
    return pd.Series(out, index=df.index, dtype=float)


def build_timeframe(df1m, rule, period_td):
    o = df1m.resample(rule, origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    o["bp"] = buying_pressure(o)
    o["stoch_k"] = stoch_k(o)
    o["body"] = body_size(o)
    o["dir"] = direction(o)
    o["consec_dir"] = consec_dir(o)
    o["chg"] = o["close"].pct_change() * 100
    # shift index forward by one bar period -- this bar's data is only
    # actually available once the bar CLOSES, not at its label (left-edge) time
    o = o.copy()
    o.index = o.index + period_td
    return o[["bp", "stoch_k", "body", "dir", "consec_dir", "chg"]]


print("Loading SOL 1m data...")
p1m = sorted(glob.glob("data/binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m)
df1m.index = pd.to_datetime(df1m.index, utc=True)
df1m = df1m.sort_index()
# only need recent history (archive starts 05-21) plus warmup for rolling windows
df1m = df1m[df1m.index >= pd.Timestamp("2026-05-01", tz="UTC")]
print(f"  1m rows: {len(df1m):,}")

print("Building 15m / 1h / 4h timeframe layers...")
tf15 = build_timeframe(df1m, "15min", pd.Timedelta(minutes=15))
tf1h = build_timeframe(df1m, "1h", pd.Timedelta(hours=1))
tf4h = build_timeframe(df1m, "4h", pd.Timedelta(hours=4))
print(f"  15m: {len(tf15)}  1h: {len(tf1h)}  4h: {len(tf4h)}")

ARCHIVE_COLS = ["logged_at", "contract_ticker", "p_market", "resolved_yes", "spot", "strike",
                 "tau_minutes", "vol_eff", "offset_pct",
                 "vpin_score", "obi_score", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct", "funding_bias"]
df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False, usecols=ARCHIVE_COLS)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "resolved_yes", "p_market", "spot", "strike", "tau_minutes", "vol_eff"])
df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
df["vol_eff"] = pd.to_numeric(df["vol_eff"], errors="coerce")
df = df[(df["vol_eff"] > 0) & (df["tau_minutes"] > 0)]
sigma_tau = df["vol_eff"] * np.sqrt(df["tau_minutes"])
df["z_score"] = np.log(df["strike"] / df["spot"]) / sigma_tau
df = df.sort_values("logged_at").reset_index(drop=True)
print(f"\narchive rows: {len(df)}  tickers: {df['contract_ticker'].nunique()}")

for label, tf in [("15m", tf15), ("1h", tf1h), ("4h", tf4h)]:
    tf = tf.copy()
    tf.index.name = "ts"
    tf_reset = tf.reset_index().sort_values("ts")
    df = pd.merge_asof(df.sort_values("logged_at"), tf_reset, left_on="logged_at", right_on="ts", direction="backward")
    df = df.rename(columns={c: f"{c}_{label}" for c in ["bp", "stoch_k", "body", "dir", "consec_dir", "chg"]})
    df = df.drop(columns=["ts"])

MICRO = ["vpin_score", "obi_score", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct", "funding_bias"]
FEATURES = ["z_score", "offset_pct"] + \
    [f"{c}_15m" for c in ["bp", "stoch_k", "body", "dir", "consec_dir", "chg"]] + \
    [f"{c}_1h" for c in ["bp", "stoch_k", "body", "dir", "consec_dir", "chg"]] + \
    [f"{c}_4h" for c in ["bp", "stoch_k", "body", "dir", "consec_dir", "chg"]] + \
    MICRO

for c in FEATURES + ["resolved_yes"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[c for c in FEATURES if c != "z_score"] + ["z_score", "resolved_yes"])
print(f"rows after feature merge + dropna: {len(df)}  tickers: {df['contract_ticker'].nunique()}")

tk_order = df.groupby("contract_ticker")["logged_at"].min().sort_values()
n_tk = len(tk_order)
tk_train = set(tk_order.index[:int(n_tk * 0.70)])
tk_val = set(tk_order.index[int(n_tk * 0.70):int(n_tk * 0.80)])
tk_test = set(tk_order.index[int(n_tk * 0.80):])
tr = df[df["contract_ticker"].isin(tk_train)]
va = df[df["contract_ticker"].isin(tk_val)]
te = df[df["contract_ticker"].isin(tk_test)]
print(f"train tk={len(tk_train)} n={len(tr)} | val tk={len(tk_val)} n={len(va)} | test tk={len(tk_test)} n={len(te)}")

X_tr, y_tr = tr[FEATURES].values, tr["resolved_yes"].values.astype(int)
X_va, y_va = va[FEATURES].values, va["resolved_yes"].values.astype(int)
X_te, y_te = te[FEATURES].values, te["resolved_yes"].values.astype(int)

base = BASE_CLF()
clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
clf.fit(X_tr, y_tr)


def evaluate(X, y, sub_df, label):
    p = clf.predict_proba(X)[:, 1]
    sub = sub_df.copy()
    sub["p_new"] = p
    auc_row = roc_auc_score(y, p)
    tk = sub.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    auc_tk = roc_auc_score((tk["y"] >= 0.5).astype(int), tk["p"])
    brier_new = float(np.mean((tk["p"] - tk["y"]) ** 2))
    brier_pm = float(np.mean((tk["pm"] - tk["y"]) ** 2))
    print(f"\n=== {label} (n={len(sub)}, tickers={len(tk)}) ===")
    print(f"  row AUC={auc_row:.4f}  ticker-clustered AUC={auc_tk:.4f}")
    print(f"  Brier: model={brier_new:.4f}  p_market={brier_pm:.4f}")

    unc = sub[sub["p_market"].between(0.35, 0.65)]
    tk_unc = unc.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    if len(tk_unc) < 15:
        print(f"  uncertain zone too thin: tickers={len(tk_unc)}")
        return
    brier_u = float(np.mean((tk_unc["p"] - tk_unc["y"]) ** 2))
    brier_pm_u = float(np.mean((tk_unc["pm"] - tk_unc["y"]) ** 2))
    corr_m = np.corrcoef(tk_unc["p"] - 0.5, tk_unc["y"])[0, 1]
    corr_pm = np.corrcoef(tk_unc["pm"] - 0.5, tk_unc["y"])[0, 1]
    print(f"  UNCERTAIN ZONE n={len(unc)} tickers={len(tk_unc)}")
    print(f"    Brier: model={brier_u:.4f}  p_market={brier_pm_u:.4f}")
    print(f"    corr: model={corr_m:+.4f}  p_market={corr_pm:+.4f}")
    for margin in [0.03, 0.05]:
        edge_yes = unc["p_new"] - unc["p_market"]
        edge_no = (1 - unc["p_new"]) - (1 - unc["p_market"])
        take_yes, take_no = edge_yes > margin, edge_no > margin
        bets = []
        if take_yes.sum() > 0:
            s = unc[take_yes]; bets.append(pd.DataFrame({"win": s["resolved_yes"], "cost": s["p_market"], "tk": s["contract_ticker"]}))
        if take_no.sum() > 0:
            s = unc[take_no]; bets.append(pd.DataFrame({"win": 1 - s["resolved_yes"], "cost": 1 - s["p_market"], "tk": s["contract_ticker"]}))
        if not bets:
            print(f"    margin={margin}: no bets"); continue
        ab = pd.concat(bets); tkb = ab.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
        nc = 100.0 / tkb["cost"]; pnl = np.where(tkb["win"] >= 0.5, nc * (1 - tkb["cost"]), -nc * tkb["cost"])
        print(f"    margin={margin}: n={len(tkb):3d}  WR={tkb['win'].mean():.1%}  BE={tkb['cost'].mean():.1%}  total=${pnl.sum():.2f}")


evaluate(X_te, y_te, te, "TEST SET")
evaluate(X_va, y_va, va, "VALIDATION SET")

print(f"\nFeature importance:")
base2 = BASE_CLF(); base2.fit(X_tr, y_tr)
try:
    imp = pd.Series(base2.feature_importances_ / (base2.feature_importances_.sum() or 1), index=FEATURES).sort_values(ascending=False)
    for f, v in imp.items():
        print(f"  {f:<16s} {v:.3f}")
except Exception as e:
    print(f"  (skipped: {e})")

print("\nDONE_S15")
