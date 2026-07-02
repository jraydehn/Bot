"""
train_hmm_vol_regime.py

Fits a Gaussian HMM on true 15m BTC log returns, then runs a two-stage
calibration process:

  Stage 1 — Train on 2025+ (captures current vol regime, not 2022 crash).
  Stage 2 — Posterior recalibration: decode scan archive, compute the
             actual per-state σ from live price moves, store those as
             the emission params used in the lognormal formula.

Rationale: the HMM's transition matrix benefits from longer history (regime
persistence), but the emission σ must reflect today's environment.  Using
scan-period price moves for σ estimation gives the best of both.

Saves:
  models/hmm_vol_regime_btc_15m.pkl   — trained HMM + recalibrated sigmas
  results/btc_scan_archive_15m.csv    — adds hmm_vol_state, hmm_sigma_raw,
                                        hmm_sigma_cal columns
  results/hmm_vol_regime_calibration.txt — calibration report

Usage:
  python3 train_hmm_vol_regime.py [--states 2|3]
"""

import argparse, math, sys, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from probability_engine import implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT_BY_ASSET

# ── Constants ──────────────────────────────────────────────────────────────────
MINS_PER_YEAR  = 525600.0
K_PUP_V2_YES   = 1.40
K_PUP_V2_NO    = 1.56
VOL_WEIGHT_BTC = REALIZED_VOL_WEIGHT_BY_ASSET.get("BTC", 0.35)
DATA           = Path(__file__).parent / "data"
MODELS         = Path(__file__).parent / "models"
RESULTS        = Path(__file__).parent / "results"

TRAIN_FROM     = "2025-01-01"   # Use 2025+ to avoid 2024 bull-run inflation
N_STATES       = 2
N_ITER         = 300
LOOKBACK_BARS  = 20             # ~5h of 15m context for state decoding
RANDOM_SEED    = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load 15m return series
# ─────────────────────────────────────────────────────────────────────────────
def load_15m_returns(min_date: str = TRAIN_FROM) -> pd.Series:
    parquet = sorted(DATA.glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
    print(f"  1m parquet: {parquet.name}")
    df = pd.read_parquet(parquet, columns=["close"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df.index >= min_date]
    c15 = df["close"].resample("15min").last().dropna()
    lr  = np.log(c15 / c15.shift(1)).dropna()
    ann = lr.std() / math.sqrt(15) * math.sqrt(MINS_PER_YEAR)
    print(f"  15m returns: {len(lr):,} bars  "
          f"({lr.index[0].date()} → {lr.index[-1].date()})  "
          f"ann_vol={ann:.1%}")
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Train HMM
# ─────────────────────────────────────────────────────────────────────────────
def train_hmm(returns: pd.Series, n_states: int) -> GaussianHMM:
    X = returns.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=N_ITER, random_state=RANDOM_SEED, tol=1e-5,
    )
    model.fit(X)
    print(f"  Converged: {model.monitor_.converged}  "
          f"log-likelihood: {model.score(X):.2f}")
    return model


def vol_ordered_states(model: GaussianHMM) -> list[int]:
    """Return state indices sorted by ascending sigma."""
    return sorted(range(model.n_components),
                  key=lambda s: float(np.sqrt(model.covars_[s, 0, 0])))


def describe_states(model: GaussianHMM, returns: pd.Series,
                    recal_sigmas=None) -> None:
    X      = returns.values.reshape(-1, 1)
    seq    = model.predict(X)
    order  = vol_ordered_states(model)
    ann_f  = math.sqrt(MINS_PER_YEAR / 15)   # per-bar → annualised

    print(f"\n  {'State':>5}  {'rank':>4}  {'μ/bar':>10}  "
          f"{'σ_train':>10}  {'σ_cal':>10}  {'ann_train':>10}  "
          f"{'ann_cal':>10}  {'freq':>6}")
    for rank, s in enumerate(order):
        mu     = float(model.means_[s, 0])
        sig_tr = float(np.sqrt(model.covars_[s, 0, 0]))
        sig_ca = recal_sigmas.get(s, sig_tr) if recal_sigmas else sig_tr
        freq   = (seq == s).mean()
        print(f"  {s:>5}  {rank:>4}  {mu*100:>+9.4f}%  "
              f"{sig_tr*100:>9.4f}%  {sig_ca*100:>9.4f}%  "
              f"{sig_tr*ann_f:>9.1%}  {sig_ca*ann_f:>9.1%}  "
              f"{freq:>5.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Decode scan archive + posterior recalibration
# ─────────────────────────────────────────────────────────────────────────────
def decode_and_recalibrate(
    model: GaussianHMM, returns_series: pd.Series
) -> tuple[list, list, list, dict]:
    """
    Decode each scan archive row, then estimate per-state σ from the
    actual price moves that occurred during the scan archive window.

    Returns (state_list, sigma_raw_list, sigma_cal_list, recal_sigmas_dict).
    sigma_raw = training-era σ for the assigned state
    sigma_cal = posterior σ estimated from scan-period price moves
    """
    sa = pd.read_csv(RESULTS / "btc_scan_archive_15m.csv")
    sa["logged_at"] = pd.to_datetime(sa["logged_at"], utc=True, format="mixed")

    order       = vol_ordered_states(model)
    rank_of     = {s: i for i, s in enumerate(order)}
    sigma_raw   = {s: float(np.sqrt(model.covars_[s, 0, 0]))
                   for s in range(model.n_components)}

    ret_idx  = returns_series.index
    ret_vals = returns_series.values

    states_out, sraw_out = [], []

    # ── Pass 1: decode every row ────────────────────────────────────────────
    for ts in sa["logged_at"]:
        pos = ret_idx.searchsorted(ts, side="right") - 1
        if pos < LOOKBACK_BARS:
            states_out.append(np.nan); sraw_out.append(np.nan)
            continue
        window    = ret_vals[pos - LOOKBACK_BARS + 1: pos + 1].reshape(-1, 1)
        vit       = model.predict(window)
        raw_state = int(vit[-1])
        states_out.append(rank_of[raw_state])
        sraw_out.append(sigma_raw[raw_state])

    sa["hmm_vol_state"] = states_out
    sa["hmm_sigma_raw"] = sraw_out

    # ── Pass 2: posterior recalibration ─────────────────────────────────────
    # Use per-expiry price moves (spot → spot_at_expiry) during the scan
    # archive window to estimate σ for each state.
    # σ_cal(state) = std(log_moves in state) / sqrt(tau_avg_minutes)
    sa_r = sa[sa["resolved_yes"].notna() & sa["hmm_vol_state"].notna()].copy()
    sa_r["log_move"] = np.log(
        sa_r["spot_at_expiry"].astype(float) / sa_r["spot"].astype(float)
    )
    # One row per expiry (use largest-tau scan = closest to window open)
    sa_exp = (sa_r.sort_values("tau_minutes", ascending=False)
                  .drop_duplicates("close_ts")
                  .copy())
    sa_exp["tau_min"] = sa_exp["tau_minutes"].astype(float)

    recal_sigmas: dict[int, float] = {}
    print("\n  Posterior recalibration (scan-period σ by HMM state):")
    for rank in sorted(sa_exp["hmm_vol_state"].dropna().unique()):
        sub  = sa_exp[sa_exp["hmm_vol_state"] == rank].dropna(subset=["log_move"])
        if len(sub) < 10:
            print(f"    Rank {rank}: too few samples ({len(sub)}), using training σ")
            recal_sigmas[rank] = sigma_raw[order[int(rank)]]
            continue
        # σ per-minute = std(log_move) / sqrt(mean_tau_minutes)
        # This annualises correctly: σ_ann = σ_per_min × sqrt(MINS_PER_YEAR)
        avg_tau = sub["tau_min"].mean()
        sigma_move_per_min = sub["log_move"].std() / math.sqrt(avg_tau)
        recal_sigmas[int(rank)] = sigma_move_per_min
        ann_cal = sigma_move_per_min * math.sqrt(MINS_PER_YEAR)
        raw_s   = sigma_raw[order[int(rank)]]
        ann_raw = raw_s / math.sqrt(15) * math.sqrt(MINS_PER_YEAR)
        print(f"    Rank {rank}: n={len(sub):4d}  "
              f"ann_raw={ann_raw:.1%}  ann_cal={ann_cal:.1%}  "
              f"avg_tau={avg_tau:.1f}m")

    # Apply calibrated sigma to every row by state rank
    scal_out = []
    for rank in states_out:
        if isinstance(rank, float) and np.isnan(rank):
            scal_out.append(np.nan)
        else:
            scal_out.append(recal_sigmas.get(int(rank), np.nan))

    sa["hmm_sigma_cal"] = scal_out
    sa.to_csv(RESULTS / "btc_scan_archive_15m.csv", index=False)
    valid = sum(1 for s in states_out if not (isinstance(s, float) and np.isnan(s)))
    print(f"\n  Decoded {valid}/{len(states_out)} rows")

    return states_out, sraw_out, scal_out, recal_sigmas


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Calibration comparison
# ─────────────────────────────────────────────────────────────────────────────
def p_yes_formula(spot, floor, tau_min, rv_ann, pm, p_up_v2,
                  vol_override_per_min=None,
                  vol_weight=VOL_WEIGHT_BTC) -> float:
    """Replicate compute_p_yes_pup_v2_15m with optional vol override."""
    if tau_min <= 0.5 or spot <= 0 or floor <= 0:
        return 0.5
    vol_realized = (vol_override_per_min if vol_override_per_min is not None
                    else rv_ann / math.sqrt(MINS_PER_YEAR))
    vol_imp   = implied_vol_from_price(pm, spot, floor, tau_min)
    vol_eff   = blend_vol(vol_realized, vol_imp, weight=vol_weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z_strike  = math.log(floor / spot) / sigma_tau
    tau_scale = math.sqrt(min(tau_min, 60.0) / 60.0)
    z_drift   = norm.ppf(float(np.clip(p_up_v2, 0.02, 0.98))) * K_PUP_V2_YES * tau_scale
    return float(np.clip(norm.cdf(z_drift - z_strike), 0.03, 0.97))


def run_calibration(n_states: int) -> None:
    sa    = pd.read_csv(RESULTS / "btc_scan_archive_15m.csv")
    sa_r  = sa[sa["resolved_yes"].notna()].copy()
    needed = ["spot", "strike", "tau_minutes", "p_market",
              "realized_vol_annual", "p_up_v2_btc",
              "hmm_vol_state", "hmm_sigma_raw", "hmm_sigma_cal"]
    sa_r = sa_r.dropna(subset=[c for c in needed if c in sa_r.columns])
    if sa_r["p_up_v2_btc"].isna().all():
        sa_r["p_up_v2_btc"] = 0.50
    print(f"\n  Calibration sample: n={len(sa_r)}")

    rows = []
    for _, row in sa_r.iterrows():
        spot    = float(row["spot"])
        floor   = float(row["strike"])
        tau     = float(row["tau_minutes"])
        pm      = float(row["p_market"])
        rv_ann  = float(row.get("realized_vol_annual", 0.30))
        p_up    = float(row.get("p_up_v2_btc", 0.50))
        sig_raw = float(row["hmm_sigma_raw"])   # per-bar (15min)
        sig_cal = float(row["hmm_sigma_cal"])   # per-minute (recalibrated)
        actual  = float(row["resolved_yes"])
        state   = int(row["hmm_vol_state"])

        # per-bar → per-minute (training σ was per 15min bar)
        sig_raw_per_min = sig_raw / math.sqrt(15)

        p_base  = p_yes_formula(spot, floor, tau, rv_ann, pm, p_up)
        p_raw   = p_yes_formula(spot, floor, tau, rv_ann, pm, p_up,
                                vol_override_per_min=sig_raw_per_min)
        p_cal   = p_yes_formula(spot, floor, tau, rv_ann, pm, p_up,
                                vol_override_per_min=sig_cal)

        rows.append({"actual": actual, "pm": pm, "state": state,
                     "p_base": p_base, "p_raw": p_raw, "p_cal": p_cal,
                     "rv_ann": rv_ann,
                     "sigma_cal_ann": sig_cal * math.sqrt(MINS_PER_YEAR)})

    df = pd.DataFrame(rows)

    def brier(p, y): return float(np.mean((p - y) ** 2))
    def logloss(p, y):
        p = np.clip(p, 1e-6, 1-1e-6)
        return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

    lines = [
        "=" * 65,
        f"BTC 15m HMM Vol Regime Calibration  (n={len(df)}, {n_states} states)",
        "=" * 65, "",
        f"  {'Model':<28} {'Brier↓':>10}  {'LogLoss↓':>10}",
        f"  {'p_market (baseline)':<28} {brier(df['pm'],     df['actual']):>10.5f}  "
        f"{logloss(df['pm'],     df['actual']):>10.5f}",
        f"  {'p_yes (realized_vol)':<28} {brier(df['p_base'], df['actual']):>10.5f}  "
        f"{logloss(df['p_base'], df['actual']):>10.5f}",
        f"  {'p_yes (HMM σ raw 2025+)':<28} {brier(df['p_raw'],  df['actual']):>10.5f}  "
        f"{logloss(df['p_raw'],  df['actual']):>10.5f}",
        f"  {'p_yes (HMM σ recalibrated)':<28} {brier(df['p_cal'],  df['actual']):>10.5f}  "
        f"{logloss(df['p_cal'],  df['actual']):>10.5f}",
        "",
        f"  Δ Brier  raw   vs base: {brier(df['p_raw'], df['actual'])-brier(df['p_base'], df['actual']):+.5f}",
        f"  Δ Brier  cal   vs base: {brier(df['p_cal'], df['actual'])-brier(df['p_base'], df['actual']):+.5f}  "
        f"({'BETTER ✓' if brier(df['p_cal'], df['actual']) < brier(df['p_base'], df['actual']) else 'WORSE ✗'})",
        "",
        "  Per-state breakdown:",
    ]

    for s in sorted(df["state"].unique()):
        sub = df[df["state"] == s]
        b_b = brier(sub["p_base"], sub["actual"])
        b_c = brier(sub["p_cal"],  sub["actual"])
        lines.append(
            f"    State {s}: n={len(sub):4d}  "
            f"YES={sub['actual'].mean():.1%}  "
            f"pm={sub['pm'].mean():.3f}  "
            f"σ_cal_ann={sub['sigma_cal_ann'].mean():.1%}  "
            f"brier_base={b_b:.4f}  brier_cal={b_c:.4f}  "
            f"{'↑ better' if b_c < b_b else '↓ worse'}"
        )

    # Edge: how much does the model disagree with market price by state?
    lines += ["", "  Model vs market divergence by state (edge signal):"]
    for s in sorted(df["state"].unique()):
        sub = df[df["state"] == s]
        cal_vs_mkt = (sub["p_cal"] - sub["pm"]).mean()
        lines.append(
            f"    State {s}: p_cal - p_market = {cal_vs_mkt:+.4f}  "
            f"(negative = model more bearish → NO edge signal)"
        )

    report = "\n".join(lines)
    print(report)
    (RESULTS / "hmm_vol_regime_calibration.txt").write_text(report)
    print(f"\n  Saved → results/hmm_vol_regime_calibration.txt")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=N_STATES)
    args = parser.parse_args()
    n_st = args.states

    print(f"=== Step 1: Load 15m returns ({TRAIN_FROM}+) ===")
    returns = load_15m_returns(TRAIN_FROM)

    print(f"\n=== Step 2: Train {n_st}-state HMM ===")
    model = train_hmm(returns, n_states=n_st)

    print(f"\n=== Step 3: Decode scan archive + posterior recalibration ===")
    states, sig_raw, sig_cal, recal = decode_and_recalibrate(model, returns)

    describe_states(model, returns, recal_sigmas=recal)

    MODELS.mkdir(exist_ok=True)
    pkl = MODELS / "hmm_vol_regime_btc_15m.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({
            "model": model, "n_states": n_st,
            "lookback_bars": LOOKBACK_BARS,
            "recal_sigmas_per_min": recal,   # rank→per-minute σ (use in formula)
            "train_from": TRAIN_FROM,
        }, f)
    print(f"\n  Saved → {pkl}")

    print(f"\n=== Step 4: Calibration test ===")
    run_calibration(n_st)

    print("\nDone.")
