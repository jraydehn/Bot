"""
save_hmm3state.py — Fit and save 3-state GaussianHMM for BTC regime detection.

Features: log_ret (daily), realized_vol (20d), ret_5d (5d momentum).
States: Bear / Sideways / Bull (by ascending mean log_ret).
Output: results/hmm_3state_btc.pkl
"""
import math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

RES_DIR = Path(__file__).parent / "results"
TRAIN_START = "2021-01-01"
N_RESTARTS = 30


def fetch_btc_daily():
    df = yf.download("BTC-USD", start=TRAIN_START, auto_adjust=True, progress=False)
    close = df["Close"].squeeze().dropna()
    close.name = "close"
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    print(f"  {len(close)} daily bars  ({close.index[0].date()} → {close.index[-1].date()})")
    return close


def build_features(close):
    lr = np.log(close / close.shift(1))
    rv = lr.rolling(20, min_periods=10).std()
    r5 = np.log(close / close.shift(5))
    return pd.DataFrame({"log_ret": lr, "realized_vol": rv, "ret_5d": r5}).dropna()


def fit_best(X, n_states, n_restarts):
    best_ll, best_model = -np.inf, None
    for seed in range(n_restarts):
        m = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            random_state=seed,
            tol=1e-6,
        )
        try:
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            pass
    return best_model, best_ll


def main():
    print("Fetching BTC-USD daily data...")
    close = fetch_btc_daily()
    feats = build_features(close)
    X = feats[["log_ret", "realized_vol", "ret_5d"]].values
    n = len(X)

    print(f"\nFitting 3-state GaussianHMM ({N_RESTARTS} restarts)...")
    model, ll = fit_best(X, 3, N_RESTARTS)
    states = model.predict(X)

    # Assign labels by ascending mean log_ret
    means_lr = model.means_[:, 0]
    order = np.argsort(means_lr)  # lowest → highest return
    state_to_name = {int(order[0]): "Bear", int(order[1]): "Sideways", int(order[2]): "Bull"}

    print(f"\n  Log-likelihood: {ll:.1f}  |  n_days: {n}")
    print(f"\n  State summary:")
    print(f"  {'State':<12} {'log_ret':>8} {'vol_20d':>8} {'ret_5d':>8} {'n_days':>7} {'%hist':>6}")
    print(f"  {'-'*55}")
    for s in range(3):
        name = state_to_name[s]
        cnt  = (states == s).sum()
        print(f"  {name:<12} {model.means_[s,0]:>+8.4f} {model.means_[s,1]:>8.4f} "
              f"{model.means_[s,2]:>+8.4f} {cnt:>7,} {cnt/n*100:>5.0f}%")

    print(f"\n  Transition matrix (row=from, col=to: Bear, Sideways, Bull):")
    for s in range(3):
        from_idx = order[s]
        row = "  ".join(f"{model.transmat_[from_idx, order[j]]:.2f}" for j in range(3))
        print(f"    {state_to_name[from_idx]:<12} → [{row}]")

    # Recent 15 days
    print(f"\n  Recent 15-day labels:")
    recent_states = model.predict(X[-15:])
    for i, (dt, s) in enumerate(zip(feats.index[-15:], recent_states)):
        name = state_to_name[s]
        ret  = feats["log_ret"].iloc[-15+i]
        rv   = feats["realized_vol"].iloc[-15+i]
        r5   = feats["ret_5d"].iloc[-15+i]
        print(f"    {dt.date()}  {name:<10}  ret={ret:+.3f}  vol={rv:.4f}  r5={r5:+.3f}")

    # Check vs current ±2% threshold
    ret_20d = np.log(close / close.shift(20))
    latest_ret20 = float(ret_20d.iloc[-1])
    thresh_label = "Bull" if latest_ret20 > 0.02 else "Bear" if latest_ret20 < -0.02 else "Sideways"
    latest_hmm = state_to_name[int(recent_states[-1])]
    print(f"\n  Today: ±2% threshold={thresh_label}  |  HMM 3-state={latest_hmm}")
    print(f"  20d return={latest_ret20*100:+.2f}%")

    # BIC
    d = 3  # features
    k_params = 3*(3-1) + (3-1) + 3*d + 3*d*(d+1)//2
    bic = -2*ll + k_params * math.log(n)
    print(f"\n  BIC: {bic:.1f}  (k_params={k_params})")

    # Save
    payload = {
        "model": model,
        "state_to_name": state_to_name,
        "feature_cols": ["log_ret", "realized_vol", "ret_5d"],
        "n_states": 3,
        "train_end": str(feats.index[-1].date()),
        "log_likelihood": ll,
        "bic": bic,
    }
    out = RES_DIR / "hmm_3state_btc.pkl"
    pickle.dump(payload, open(out, "wb"), protocol=4)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
