"""
S15 -- Reconstruct the 4 signal families that turned out to be missing
from paper_trades.csv entirely (GARCH, macro regime Bull/Sideways/Bear,
MACD, Donchian), computed causally from BTC 1h price history using the
EXACT formulas already live in paper_trade_runner.py, for the two
blocked populations (rising+YES, crashing+NO). Then re-run the rescue
sweep including these.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/pup_v3_15m_window_sweep_20260706"

c1h = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet")["close"].astype(float)
c1h = c1h.sort_index()
hi1h = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet")["high"].astype(float).sort_index()
lo1h = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet")["low"].astype(float).sort_index()

with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)


def macro_regime_probs_at(ts):
    """Exact replica of _compute_macro_regime_probs, evaluated causally
    using only bars with index <= the last bar strictly before ts."""
    idx = c1h.index.searchsorted(ts, side="right") - 1
    if idx < 80:
        return None
    window = c1h.iloc[max(0, idx - 200):idx + 1]
    log_ret = np.log(window / window.shift(1))
    ret_24h = log_ret.rolling(24, min_periods=12).sum()
    ret_72h = log_ret.rolling(72, min_periods=36).sum()
    rv24 = log_ret.rolling(24, min_periods=12).std()
    roll_mean = log_ret.rolling(24, min_periods=12).mean()
    sharpe_24h = (roll_mean / rv24.replace(0, np.nan)).fillna(0.0)
    feat = pd.DataFrame({"ret_24h": ret_24h, "ret_72h": ret_72h, "rv24": rv24,
                        "sharpe_24h": sharpe_24h}).dropna()
    if len(feat) < 10:
        return None
    feat = feat.iloc[-80:]
    X = macro_pkg["scaler"].transform(feat.values.astype(float))
    posteriors = macro_pkg["model"].predict_proba(X)
    last = posteriors[-1]
    label_names = macro_pkg["label_names"]
    return {label_names[s]: float(last[s]) for s in range(len(last))}


def garch_ratio_at(ts):
    idx = c1h.index.searchsorted(ts, side="right") - 1
    if idx < 502:
        return np.nan
    window = np.log(c1h.iloc[idx - 500:idx + 1] / c1h.iloc[idx - 500:idx + 1].shift(1)).dropna() * 100
    try:
        from arch import arch_model
        am = arch_model(window, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_v = float(res.conditional_volatility.iloc[-1])
        omega, alpha, beta = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
        persist = alpha + beta
        lr_vol = float(np.sqrt(omega / (1 - persist))) if persist < 1 else float(window.std())
        return cond_v / lr_vol if lr_vol > 0 else np.nan
    except Exception:
        return np.nan


def macd_hist_at(ts):
    idx = c1h.index.searchsorted(ts, side="right") - 1
    if idx < 40:
        return np.nan
    window = c1h.iloc[max(0, idx - 100):idx + 1]
    macd_line = window.ewm(span=12, adjust=False).mean() - window.ewm(span=26, adjust=False).mean()
    hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    return float(hist.iloc[-1])


def donchian_pos_at(ts):
    idx = hi1h.index.searchsorted(ts, side="right") - 1
    if idx < 20:
        return np.nan
    hi = float(hi1h.iloc[idx - 19:idx + 1].max())
    lo = float(lo1h.iloc[idx - 19:idx + 1].min())
    close_now = float(c1h.iloc[idx])
    return (close_now - lo) / (hi - lo) if hi > lo else np.nan


def enrich(df):
    df = df.copy()
    macro = df["logged_at_parsed"].apply(macro_regime_probs_at)
    df["macro_bull"] = macro.apply(lambda d: d.get("Bull") if d else np.nan)
    df["macro_sideways"] = macro.apply(lambda d: d.get("Sideways") if d else np.nan)
    df["macro_bear"] = macro.apply(lambda d: d.get("Bear") if d else np.nan)
    df["garch_ratio"] = df["logged_at_parsed"].apply(garch_ratio_at)
    df["macd_hist_1h_recon"] = df["logged_at_parsed"].apply(macd_hist_at)
    df["donch_1h_recon"] = df["logged_at_parsed"].apply(donchian_pos_at)
    return df


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
        df = pd.read_csv(p, usecols=cols, low_memory=False)
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
        if "asset" not in df.columns:
            df["asset"] = "BTC"
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[full["asset"] == "BTC"]
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    return full.sort_values("logged_at_parsed")


if __name__ == "__main__":
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
    taken["yw"] = (taken["logged_at_parsed"].dt.isocalendar().year.astype(str) + "-W" +
                  taken["week"].astype(str).str.zfill(2))
    covered = taken.dropna(subset=["state3", "would_pnl"]).copy()
    covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])

    rising_yes = covered[(covered["state3"] == "rising") & (covered["side"] == "yes")]
    crashing_no = covered[(covered["state3"] == "crashing") & (covered["side"] == "no")]

    print("Reconstructing signals for rising/YES blocked population...")
    rising_yes_enriched = enrich(rising_yes)
    print("Reconstructing signals for crashing/NO blocked population...")
    crashing_no_enriched = enrich(crashing_no)

    NEW_CANDIDATES = ["macro_bull", "macro_sideways", "macro_bear", "garch_ratio",
                      "macd_hist_1h_recon", "donch_1h_recon"]

    for pop, label in [(rising_yes_enriched, "rising_yes"), (crashing_no_enriched, "crashing_no")]:
        print(f"\n=== {label}: coverage of reconstructed signals ===")
        for c in NEW_CANDIDATES:
            print(f"  {c}: {pop[c].notna().sum()}/{len(pop)} non-null")

    rising_yes_enriched.to_csv(f"{OUT}/rising_yes_enriched.csv", index=False)
    crashing_no_enriched.to_csv(f"{OUT}/crashing_no_enriched.csv", index=False)
    print("\nsaved enriched CSVs")
