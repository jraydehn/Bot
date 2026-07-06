"""
S8 -- Reconstruct GARCH, macro regime, MACD, Donchian for the ETH AGREE
population (same 4 categories that were missing from BTC's CSV too),
computed causally from ETH's OWN 1h price history. GARCH runner-side is
BTC-only (paper_trade_runner.py's _get_garch_ratio hardcodes asset!="BTC"
-> None), so there's no live ETH formula to mirror -- the GARCH(1,1)
METHOD itself is asset-agnostic, so it's refit here on ETH's own returns.
No ETH-specific macro-regime HMM exists (only BTC's was built) -- applied
BTC's macro regime model to ETH's OWN price/return features anyway as a
cross-asset "broader market regime" proxy, clearly flagged as such, not
a genuine ETH-native regime signal.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/eth_pup_rebuild_20260706"

c1h = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_ETHUSDT_1h.parquet")
close = c1h["close"].astype(float).sort_index()
hi = c1h["high"].astype(float).sort_index()
lo = c1h["low"].astype(float).sort_index()

with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)


def macro_regime_probs_at(ts):
    """BTC-trained macro regime HMM applied to ETH's OWN ret_24h/72h/rv24/
    sharpe_24h features -- a cross-asset 'broader regime' PROXY, not a
    genuine ETH-native model (none exists)."""
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 80:
        return None
    window = close.iloc[max(0, idx - 200):idx + 1]
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
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 502:
        return np.nan
    window = np.log(close.iloc[idx - 500:idx + 1] / close.iloc[idx - 500:idx + 1].shift(1)).dropna() * 100
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
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 40:
        return np.nan
    window = close.iloc[max(0, idx - 100):idx + 1]
    macd_line = window.ewm(span=12, adjust=False).mean() - window.ewm(span=26, adjust=False).mean()
    hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    return float(hist.iloc[-1])


def donchian_pos_at(ts):
    idx = hi.index.searchsorted(ts, side="right") - 1
    if idx < 20:
        return np.nan
    h = float(hi.iloc[idx - 19:idx + 1].max())
    l = float(lo.iloc[idx - 19:idx + 1].min())
    c = float(close.iloc[idx])
    return (c - l) / (h - l) if h > l else np.nan


if __name__ == "__main__":
    covered = pd.read_csv(f"{OUT}/eth_agree_pop_for_reconstruction.csv", low_memory=False)
    covered["logged_at_parsed"] = pd.to_datetime(covered["logged_at_parsed"], utc=True, errors="coerce")
    agree_pop = covered[covered["agree"]].copy()
    print(f"reconstructing for AGREE population n={len(agree_pop)}...")

    macro = agree_pop["logged_at_parsed"].apply(macro_regime_probs_at)
    agree_pop["macro_bull"] = macro.apply(lambda d: d.get("Bull") if d else np.nan)
    agree_pop["macro_sideways"] = macro.apply(lambda d: d.get("Sideways") if d else np.nan)
    agree_pop["macro_bear"] = macro.apply(lambda d: d.get("Bear") if d else np.nan)
    print("  macro regime done")
    agree_pop["garch_ratio"] = agree_pop["logged_at_parsed"].apply(garch_ratio_at)
    print("  garch done")
    agree_pop["macd_hist_1h_recon"] = agree_pop["logged_at_parsed"].apply(macd_hist_at)
    agree_pop["donch_1h_recon"] = agree_pop["logged_at_parsed"].apply(donchian_pos_at)
    print("  macd/donchian done")

    for c in ["macro_bull", "macro_sideways", "macro_bear", "garch_ratio", "macd_hist_1h_recon", "donch_1h_recon"]:
        print(f"  {c}: {agree_pop[c].notna().sum()}/{len(agree_pop)} non-null")

    agree_pop.to_csv(f"{OUT}/eth_agree_reconstructed.csv", index=False)
    print("saved")
