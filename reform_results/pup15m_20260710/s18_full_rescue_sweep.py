"""
S18 -- EXHAUSTIVE rescue sweep for wait_15m_agreement_gate (user directive:
obi, funding, liquidations, all CG signals, dir, bp, other HMMs, kalman,
multi-TF rsi/donchian/bollinger/stoch, trend, rev, offset, tau, price change,
garch/arima/vol, vwap, ema stack, macd, adx, ...).

Population: the 599 disagree-at-entry trades from the full 8-wk sim.
Target: per-trade delta = wait_pnl - would_pnl. A RESCUE is a decision-time
bucket where delta is reliably NEGATIVE (waiting hurts -> enter immediately).

Sources:
  A. Logged book columns -- full-coverage (all 3 file generations) and
     cur-only (06-17+, ~60% of population), each tested at native coverage.
  B. Reconstructed MTF indicators from Binance 1m (zero-lookahead: completed
     bars only, effective = bar_open + frame): RSI14, stoch14, BB %B + width,
     Donchian pos, Keltner pos, MACD hist (norm), ADX14, EMA-stack score,
     chg, rvol at 5m / 15m / 1h / 4h.
  C. Session backfills: cg30_state, macro posteriors, pv3_state, sc strength.

Stats: episode-clustered bootstrap P(wait_helps) per bucket (quartiles for
numerics, values for categoricals, n>=60). Multiplicity: report total test
count + expected null minimum; candidates (P<=0.02) must ALSO show negative
delta in BOTH time halves to be considered.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
rng = np.random.default_rng(47)

sim = pd.read_csv(f"{OUT}/wait_sim_full8wk.csv", parse_dates=["decision_time"])
pop = sim[sim["action"] != "unchanged"].copy()

# ---- A: all logged columns ----
files = ["results/paper_trades_archive_20260525_1432_pre_branched_drift.csv",
         "results/paper_trades_pre_regime_pup_20260616.csv",
         "results/paper_trades.csv"]
frames = []
for f in files:
    d = pd.read_csv(f, low_memory=False)
    d["decision_time"] = pd.to_datetime(d["decision_time"], utc=True, errors="coerce", format="mixed")
    frames.append(d)
book = pd.concat(frames, ignore_index=True).drop_duplicates(
    subset=["contract_ticker", "decision_time"], keep="last")
pop = pop.merge(book, on=["contract_ticker", "decision_time"], how="left", suffixes=("", "_b"))

NUMERIC_LOGGED = [
    "offset_pct", "tau_minutes", "p_market", "spread", "z_score", "z_shift",
    "stoch_k", "stoch_d", "stoch_k_4h", "pc1_rsi", "adx_1h",
    "composite_trend", "composite_rev", "composite_p_up", "p_up_v2", "p_up_v3",
    "obi_score", "obi_raw", "vpin_score", "vpin_raw", "vol_score",
    "funding_bias", "avg_funding_rate", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
    "cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h", "cg_spot_cb_ratio_4h",
    "cg_liq_ratio_4h", "cg_liq_total_4h", "all_liq_ratio_1h", "all_liq_ratio_4h",
    "hl_liq_ratio_4h", "cvd_4h", "fear_greed" if "fear_greed" in pop.columns else "vol_eff",
    "vol_ratio", "vol_60m", "vol_60m_model", "vol_eff", "vol_implied_kalshi", "rvol_1h",
    "kalman_residual", "kalman_velocity", "hurst_exponent",
    "ou_theta", "ou_z_score", "ou_halflife_min", "ou_tau_drift",
    "arima_forecast_1h", "vwap_distance_pct", "vwap_score", "vwap_stretch_score", "vwap_total",
    "ema_stretch_score", "ema_alignment", "bp_5m", "bp_1h",
    "chg_5m", "chg_10m", "chg_30m", "chg_1h", "chg_2h", "chg_3h", "body_15m", "pm_drift_5m",
    "direction_strength", "demand_pct", "kelly_fraction", "hmm_ms_prob", "hmm_vd_prob",
    "hmm_of_prob", "hmm_r1_prob", "hmm_time_in_state", "hmm_vol_k10",
    "macro_regime_bull", "macro_regime_sdwy", "macro_regime_bear",
]
CATEG_LOGGED = ["ema_stack_bias", "structure_bias", "stoch_bias", "vwap_signal", "dir_15m",
                "sharp_move_active", "stoch_crossover_active", "stoch_flipped", "in_demand_zone",
                "smc_1h", "smc_4h", "hmm_ms_state", "hmm_vd_state", "hmm_of_state", "hmm_ps_state",
                "hmm_gd_state", "hmm_vol_state", "hmm_zdrift_state", "cg_flow_state",
                "markov_regime_daily", "markov_regime_7state", "pup_v3_hmm_state",
                "hawk_vol_regime", "kc_bo_1h", "side"]

# ---- B: reconstructed MTF battery ----
p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[(df1m.index >= "2026-04-15") & (df1m.index <= "2026-07-11")]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

def wilder_adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()

recon = {}
for frame, rule in [("5m", "5min"), ("15m", "15min"), ("1h", "1h"), ("4h", "4h")]:
    d = df1m.resample(rule).agg(AGG).dropna()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    stoch = ((c - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    pctb = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    bbw = (4 * sd20) / ma20
    d_lo, d_hi = l.rolling(20).min(), h.rolling(20).max()
    donch = (c - d_lo) / (d_hi - d_lo).replace(0, np.nan)
    e20 = c.ewm(span=20, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    kelt = (c - (e20 - 2 * atr)) / (4 * atr).replace(0, np.nan)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    mhist = (macd - macd.ewm(span=9, adjust=False).mean()) / c * 10000
    adx = wilder_adx(h, l, c)
    e8, e21, e50 = (c.ewm(span=s, adjust=False).mean() for s in (8, 21, 50))
    stack = ((e8 > e21).astype(int) + (e21 > e50).astype(int)
             - (e8 < e21).astype(int) - (e21 < e50).astype(int))
    chg = c.pct_change() * 100
    rv = chg.rolling(20).std()
    F = pd.DataFrame({f"rsi_{frame}": rsi, f"stoch_{frame}": stoch, f"pctb_{frame}": pctb,
                      f"bbw_{frame}": bbw, f"donch_{frame}": donch, f"kelt_{frame}": kelt,
                      f"macdh_{frame}": mhist, f"adx_{frame}": adx, f"emastack_{frame}": stack,
                      f"chg_{frame}_r": chg, f"rvol20_{frame}": rv})
    F.index = F.index + pd.tseries.frequencies.to_offset(rule)   # effective = bar close
    recon[frame] = F.sort_index()

pop = pop.sort_values("decision_time")
for frame, F in recon.items():
    F.index.name = "eff"
    pop = pd.merge_asof(pop, F.reset_index().sort_values("eff"),
                        left_on="decision_time", right_on="eff", direction="backward",
                        suffixes=("", f"_dup{frame}"))
    pop = pop.drop(columns=[c for c in pop.columns if c.startswith("eff")])
RECON_COLS = [c for F in recon.values() for c in F.columns]

# ---- C: session backfills ----
cg30 = pd.read_csv(f"{OUT}/cg30m_states.csv", parse_dates=["effective"]).sort_values("effective")
pop = pd.merge_asof(pop, cg30[["effective", "cg30_state"]], left_on="decision_time",
                    right_on="effective", direction="backward").drop(columns=["effective"])
sc = pd.read_csv(f"{OUT}/pup15m_sc_series_2026.csv", parse_dates=["effective"]).sort_values("effective")
pop = pd.merge_asof(pop, sc[["effective", "p_sc", "pv3_state"]], left_on="decision_time",
                    right_on="effective", direction="backward")
pop["sc_strength"] = (pop["p_sc"] - 0.5).abs()
pop["hour_utc"] = pop["decision_time"].dt.hour
gap = pop["decision_time"].diff().dt.total_seconds() / 60
pop["episode"] = (gap.isna() | (gap > 90)).cumsum()
mid = pop["decision_time"].median()

def ep_boot(sub, n_boot=1000):
    ep = sub.groupby("episode")["delta"].sum()
    boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(n_boot)]
    return float(np.mean(np.array(boots) >= 0))   # small => waiting hurts here

tests = []
NUMS = [c for c in dict.fromkeys(NUMERIC_LOGGED + RECON_COLS + ["sc_strength", "hour_utc"])
        if c in pop.columns]
for col in NUMS:
    x = pd.to_numeric(pop[col], errors="coerce")
    if x.notna().sum() < 120 or x.nunique() < 5:
        continue
    for qlo, qhi, lbl in [(0, .25, "Q1"), (.25, .5, "Q2"), (.5, .75, "Q3"), (.75, 1, "Q4")]:
        lo, hi = x.quantile(qlo), x.quantile(qhi)
        m = (x >= lo) & ((x < hi) if qhi < 1 else (x <= hi))
        sub = pop[m & x.notna()]
        if len(sub) < 60:
            continue
        tests.append((col, lbl, len(sub), sub["delta"].sum(), ep_boot(sub)))
for col in [c for c in CATEG_LOGGED + ["cg30_state", "pv3_state", "action"] if c in pop.columns]:
    for val, sub in pop.groupby(pop[col].astype(str)):
        if len(sub) < 60 or val in ("nan", ""):
            continue
        tests.append((col, f"={val}", len(sub), sub["delta"].sum(), ep_boot(sub)))

res = pd.DataFrame(tests, columns=["signal", "bucket", "n", "delta_sum", "P_wait_helps"])
res = res.sort_values("P_wait_helps")
res.to_csv(f"{OUT}/s18_rescue_sweep_all.csv", index=False)
print(f"total tests: {len(res)}  (expected null minimum P ~ {1/(len(res)+1):.4f})")
print(f"buckets with delta<0: {(res['delta_sum']<0).sum()} of {len(res)}")
print("\ntop 15 waiting-hurts candidates:")
print(res.head(15).round(3).to_string(index=False))

print("\n=== split-half confirmation for anything P<=0.02 ===")
cands = res[res["P_wait_helps"] <= 0.02]
if not len(cands):
    print("  (none reached P<=0.02)")
for _, row in cands.iterrows():
    col, lbl = row["signal"], row["bucket"]
    if lbl.startswith("="):
        m = pop[col].astype(str) == lbl[1:]
    else:
        x = pd.to_numeric(pop[col], errors="coerce")
        q = {"Q1": (0, .25), "Q2": (.25, .5), "Q3": (.5, .75), "Q4": (.75, 1)}[lbl]
        lo, hi = x.quantile(q[0]), x.quantile(q[1])
        m = (x >= lo) & ((x < hi) if q[1] < 1 else (x <= hi)) & x.notna()
    for hn, hm in [("H1", pop["decision_time"] <= mid), ("H2", pop["decision_time"] > mid)]:
        sub = pop[m & hm]
        if len(sub) < 20:
            print(f"  {col} {lbl} {hn}: n={len(sub)} too thin"); continue
        print(f"  {col} {lbl} {hn}: n={len(sub)}  delta ${sub['delta'].sum():+.2f}  "
              f"P(wait_helps)={ep_boot(sub):.3f}")
print("DONE_S18")
