"""
train_macro_regime_hmm.py — Build 3-state macro regime HMM for BTC.

Features (price/vol directional — independent of composite scores):
  1. ret_24h    : 24h log return sum   (short-term direction)
  2. ret_72h    : 72h log return sum   (medium-term trend persistence)
  3. rv24       : 24h realized vol     (orthogonal vol dimension)
  4. sharpe_24h : ret / rv24           (risk-adjusted direction)

These deliberately avoid all composite score indicators:
  NO MACD, Stoch, RSI, Bollinger Bands, Keltner, Williams %R, VWAP.

State interpretation (learned, not imposed):
  Bull     : ret_24h > 0, ret_72h > 0, sharpe_24h > 0
  Bear     : ret_24h < 0, ret_72h < 0, sharpe_24h < 0
  Sideways : ret ≈ 0, sharpe ≈ 0, rv24 low

Output:
  reform_results/hmm_macro_regime_btc.pkl   — trained HMM + scaler
  reform_results/hmm_macro_labels_btc.parquet — per-bar regime assignments
"""

import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

N_STATES    = 3
DATA_START  = pd.Timestamp("2024-01-01", tz="UTC")   # 1m-data floor
RANDOM_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────
def load_1h() -> pd.DataFrame:
    f = sorted(glob.glob(str(ROOT / "data/binanceus_BTCUSDT_1h_1970*.parquet")))[-1]
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.columns = df.columns.str.lower()
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly directional + vol features.
    Avoids composite score indicators (MACD, Stoch, RSI, BB, Keltner, Williams).

    Feature design rationale:
      ret_24h   : captures short-term direction (positive = recent up move)
      ret_72h   : captures medium-term direction (persistence = trend)
      sharpe_24h: risk-adjusted direction (filters noise from direction)
      rv24      : vol level (orthogonal to direction — captures vol regime)

    Expected cluster structure:
      Bull     : ret_24h > 0, ret_72h > 0, sharpe_24h > 0, rv24 moderate
      Bear     : ret_24h < 0, ret_72h < 0, sharpe_24h < 0, rv24 moderate-high
      Sideways : ret_24h ≈ 0, ret_72h ≈ 0, sharpe_24h ≈ 0, rv24 low
    """
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    # 1. 24h log return (primary direction)
    ret_24h = log_ret.rolling(24, min_periods=12).sum()

    # 2. 72h log return (trend persistence)
    ret_72h = log_ret.rolling(72, min_periods=36).sum()

    # 3. Realized vol 24h
    rv24 = log_ret.rolling(24, min_periods=12).std()

    # 4. Sharpe-like: risk-adjusted direction (annualized Sharpe at hourly freq)
    roll_mean_ret = log_ret.rolling(24, min_periods=12).mean()
    sharpe_24h    = (roll_mean_ret / rv24.replace(0, np.nan)).fillna(0.0)

    feat = pd.DataFrame({
        "ret_24h":    ret_24h,
        "ret_72h":    ret_72h,
        "rv24":       rv24,
        "sharpe_24h": sharpe_24h,
    }, index=df.index).dropna()

    return feat


# ── BIC model selection ───────────────────────────────────────────────────────
def bic(model, X, n_params):
    log_prob = model.score(X) * len(X)
    return -2 * log_prob + n_params * np.log(len(X))


def select_n_states(X_scaled: np.ndarray, n_range=(2, 6)) -> int:
    print("\n  BIC model selection:")
    print(f"  {'States':>6}  {'BIC':>12}  {'LogL':>12}")
    print("  " + "-" * 34)
    results = []
    for n in range(*n_range):
        model = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=200, random_state=RANDOM_SEED)
        model.fit(X_scaled)
        ll    = model.score(X_scaled) * len(X_scaled)
        # params: transition (n²-n), means (n×d), covars (n×d×d), startprob (n-1)
        d     = X_scaled.shape[1]
        npar  = (n*n - n) + n*d + n*d*d + (n-1)
        b     = -2 * ll + npar * np.log(len(X_scaled))
        results.append((n, b, ll))
        print(f"  {n:>6}  {b:>12.1f}  {ll:>12.1f}")
    best_n = min(results, key=lambda r: r[1])[0]
    print(f"\n  → BIC-optimal: {best_n} states")
    return best_n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading 1h BTC data...")
    df1h  = load_1h()
    print(f"  {df1h.index.min().date()} → {df1h.index.max().date()}  ({len(df1h):,} bars)")

    print("\nBuilding features...")
    feat = build_features(df1h)
    print(f"  {feat.index.min().date()} → {feat.index.max().date()}  ({len(feat):,} bars)")

    # Scale
    scaler  = StandardScaler()
    X_all   = feat.values.astype(float)
    X_scaled = scaler.fit_transform(X_all)

    # BIC selection
    best_n = select_n_states(X_scaled, n_range=(2, 6))
    # Override to N_STATES if BIC differs — keep interpretability
    if best_n != N_STATES:
        print(f"\n  BIC suggests {best_n} but using {N_STATES} for interpretability.")
    use_n = N_STATES

    # Final model
    print(f"\nFitting final {use_n}-state HMM on {len(X_scaled):,} bars...")
    model = GaussianHMM(n_components=use_n, covariance_type="full",
                        n_iter=500, random_state=RANDOM_SEED)
    model.fit(X_scaled)
    ll = model.score(X_scaled) * len(X_scaled)
    print(f"  Log-likelihood: {ll:,.1f}")

    # Viterbi labels
    labels = model.predict(X_scaled)
    feat   = feat.copy()
    feat["state"] = labels

    # ── Regime statistics ─────────────────────────────────────────────────────
    next_ret  = np.log(df1h["close"] / df1h["close"].shift(1)).shift(-1)
    next_up   = (next_ret > 0).astype(float)
    feat["next_up"] = next_up.reindex(feat.index)

    print(f"\n{'='*70}")
    print(f"  REGIME STATISTICS")
    print(f"{'='*70}")
    print(f"  {'State':>5}  {'n':>6}  {'%hrs':>6}  {'up%':>6}  "
          f"{'ret_24h':>10}  {'ret_72h':>10}  {'rv24':>8}  {'sharpe':>8}")
    print("  " + "-" * 70)

    state_map = {}
    for s in sorted(feat["state"].unique()):
        mask   = feat["state"] == s
        grp    = feat[mask]
        n      = mask.sum()
        up_pct = grp["next_up"].mean()
        r24    = grp["ret_24h"].mean()
        r72    = grp["ret_72h"].mean()
        rv     = grp["rv24"].mean()
        sh     = grp["sharpe_24h"].mean()
        pct    = n / len(feat)
        print(f"  {s:>5}  {n:>6,}  {pct:>6.1%}  {up_pct:>6.1%}  "
              f"{r24:>+10.4f}  {r72:>+10.4f}  {rv:>8.5f}  {sh:>+8.3f}")

        # Label assignment by sharpe_24h (highest = Bull, lowest = Bear)
        state_map[s] = {"n": n, "up_pct": float(up_pct), "sharpe_24h": float(sh)}

    # Sort states: Bull (highest sharpe), Bear (lowest), Sideways (middle)
    sorted_states = sorted(state_map.keys(), key=lambda s: state_map[s]["sharpe_24h"], reverse=True)
    label_names = {}
    for rank, s in enumerate(sorted_states):
        name = ["Bull", "Sideways", "Bear"][rank] if use_n == 3 else f"State{rank}"
        label_names[s] = name

    print(f"\n  Regime labels (by sharpe_24h):")
    for s, name in label_names.items():
        print(f"    State {s} → {name}  (up%={state_map[s]['up_pct']:.1%}  sharpe={state_map[s]['sharpe_24h']:+.3f})")

    # Add named label
    feat["regime"] = feat["state"].map(label_names)

    # Transition matrix
    print(f"\n  Transition matrix (rows=from, cols=to):")
    states_ord = sorted(label_names.keys())
    header = "         " + "".join(f"{label_names[s]:>10}" for s in states_ord)
    print(f"  {header}")
    for i in states_ord:
        row = f"  {label_names[i]:>8} " + "".join(f"{model.transmat_[i, j]:>10.3f}" for j in states_ord)
        print(row)

    # Monthly regime breakdown
    print(f"\n  Monthly regime distribution (recent 12 months):")
    feat["month"] = feat.index.to_period("M")
    recent = feat[feat.index >= feat.index.max() - pd.DateOffset(months=12)]
    for mo, grp in recent.groupby("month"):
        counts = grp["regime"].value_counts()
        parts  = "  ".join(f"{r}={counts.get(r, 0):3d}h" for r in ["Bull", "Sideways", "Bear"])
        print(f"    {mo}:  {parts}")

    # ── Save ──────────────────────────────────────────────────────────────────
    FEATURE_COLS = ["ret_24h", "ret_72h", "rv24", "sharpe_24h"]
    payload = {
        "model":        model,
        "scaler":       scaler,
        "n_states":     use_n,
        "label_names":  label_names,   # {state_int: "Bull"/"Sideways"/"Bear"}
        "feature_cols": FEATURE_COLS,
    }
    pkl_path = OUT_DIR / "hmm_macro_regime_btc.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved HMM → {pkl_path}")

    label_path = OUT_DIR / "hmm_macro_labels_btc.parquet"
    feat[["state", "regime"] + FEATURE_COLS].to_parquet(label_path)
    print(f"  Saved labels → {label_path}  ({len(feat):,} bars)")

    return feat, model, scaler, label_names


if __name__ == "__main__":
    feat, model, scaler, label_names = main()
