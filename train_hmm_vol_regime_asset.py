"""
train_hmm_vol_regime_asset.py

Asset-parameterized version of train_hmm_vol_regime.py.
Fits a 2-state ergodic Gaussian HMM on 15m log-returns for BTC, ETH, or SOL,
then runs posterior recalibration using the asset's scan archive.

Additional outputs vs the BTC-only version:
  - Ergodicity check: eigenvalue gap, mixing time estimate
  - Sojourn time statistics per state (for semi-Markov zone calibration)

Saves:
  models/hmm_ergodic_2state_{asset_lower}_15m.pkl
  results/hmm_vol_regime_{asset_lower}_calibration.txt

Usage:
  python3 train_hmm_vol_regime_asset.py --asset ETH
  python3 train_hmm_vol_regime_asset.py --asset SOL
  python3 train_hmm_vol_regime_asset.py --asset BTC   # re-trains BTC
"""

import argparse, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

MINS_PER_YEAR = 525600.0
N_ITER        = 300
LOOKBACK_BARS = 20
RANDOM_SEED   = 42
TRAIN_FROM    = "2025-01-01"

DATA    = Path(__file__).parent / "data"
MODELS  = Path(__file__).parent / "models"
RESULTS = Path(__file__).parent / "results"

ASSET_CONFIG = {
    "BTC": {
        "ticker":       "BTCUSDT",
        "archive":      "btc_scan_archive.csv",
        "1m_glob":      "binanceus_BTCUSDT_1m_*_*.parquet",
    },
    "ETH": {
        "ticker":       "ETHUSDT",
        "archive":      "eth_scan_archive.csv",
        "1m_glob":      "binanceus_ETHUSDT_1m_2024-01-01_*.parquet",
    },
    "SOL": {
        "ticker":       "SOLUSDT",
        "archive":      "sol_scan_archive.csv",
        "1m_glob":      "binanceus_SOLUSDT_1m_2024-01-01_*.parquet",
    },
}


# ── 1. Load 15m returns ───────────────────────────────────────────────────────

def load_15m_returns(asset: str) -> pd.Series:
    cfg     = ASSET_CONFIG[asset]
    files   = sorted(DATA.glob(cfg["1m_glob"]))
    if not files:
        raise FileNotFoundError(f"No 1m parquet found for {asset}: {cfg['1m_glob']}")
    parquet = max(files, key=lambda p: p.stat().st_mtime)
    print(f"  1m parquet: {parquet.name}")
    df = pd.read_parquet(parquet, columns=["close"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[df.index >= TRAIN_FROM]
    c15 = df["close"].resample("15min").last().dropna()
    lr  = np.log(c15 / c15.shift(1)).dropna()
    ann = lr.std() / math.sqrt(15) * math.sqrt(MINS_PER_YEAR)
    print(f"  15m returns: {len(lr):,} bars  "
          f"({lr.index[0].date()} → {lr.index[-1].date()})  "
          f"ann_vol={ann:.1%}")
    return lr


# ── 2. Train HMM ──────────────────────────────────────────────────────────────

def train_hmm(returns: pd.Series) -> GaussianHMM:
    X = returns.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=2, covariance_type="diag",
        n_iter=N_ITER, random_state=RANDOM_SEED, tol=1e-5,
    )
    model.fit(X)
    print(f"  Converged: {model.monitor_.converged}  "
          f"log-likelihood: {model.score(X):.2f}")
    return model


def vol_ordered_states(model: GaussianHMM) -> list:
    return sorted(range(model.n_components),
                  key=lambda s: float(np.sqrt(model.covars_[s, 0, 0])))


# ── 3. Ergodicity check ───────────────────────────────────────────────────────

def ergodicity_check(model: GaussianHMM) -> dict:
    """
    Eigenvalue gap of the transition matrix determines how quickly the chain
    mixes (forgets its initial state).  For a 2-state ergodic chain:
      - eigenvalues are 1 and (p00 + p11 - 1)
      - gap = 1 - |second eigenvalue|
      - mixing_time ≈ 1 / gap  (in bars)
    A large gap (close to 1) = fast mixing = regimes switch often.
    A small gap (close to 0) = slow mixing = regimes are persistent.
    """
    A = model.transmat_
    eigvals = np.linalg.eigvals(A)
    eigvals_sorted = sorted(np.abs(eigvals), reverse=True)
    gap         = float(1.0 - eigvals_sorted[1])
    mixing_bars = 1.0 / gap if gap > 0 else float("inf")
    mixing_mins = mixing_bars * 15

    # Stationary distribution
    pi = model.startprob_
    # More accurately: left eigenvector of A for eigenvalue 1
    vals, vecs = np.linalg.eig(A.T)
    stat_vec = np.real(vecs[:, np.argmax(np.real(vals))])
    stat_vec = stat_vec / stat_vec.sum()

    return {
        "eigenvalue_gap":    gap,
        "mixing_bars":       mixing_bars,
        "mixing_minutes":    mixing_mins,
        "stationary":        stat_vec,
        "transmat":          A,
    }


# ── 4. Describe states ────────────────────────────────────────────────────────

def describe_states(model: GaussianHMM, returns: pd.Series,
                    recal_sigmas: dict = None) -> None:
    X     = returns.values.reshape(-1, 1)
    seq   = model.predict(X)
    order = vol_ordered_states(model)
    ann_f = math.sqrt(MINS_PER_YEAR / 15)

    print(f"\n  {'State':>5}  {'rank':>4}  {'μ/bar':>10}  "
          f"{'σ_train':>10}  {'σ_cal':>10}  "
          f"{'ann_train':>10}  {'ann_cal':>10}  {'freq':>6}")
    for rank, s in enumerate(order):
        mu     = float(model.means_[s, 0])
        sig_tr = float(np.sqrt(model.covars_[s, 0, 0]))
        sig_ca = recal_sigmas.get(rank, sig_tr) if recal_sigmas else sig_tr
        freq   = (seq == s).mean()
        print(f"  {s:>5}  {rank:>4}  {mu*100:>+9.4f}%  "
              f"{sig_tr*100:>9.4f}%  {sig_ca*100:>9.4f}%  "
              f"{sig_tr*ann_f:>9.1%}  {sig_ca*ann_f:>9.1%}  "
              f"{freq:>5.1%}")


# ── 5. Sojourn time statistics ────────────────────────────────────────────────

def sojourn_stats(model: GaussianHMM, returns: pd.Series) -> dict:
    """
    Compute empirical sojourn (run-length) distributions per state.
    Used to calibrate semi-Markov zone thresholds (early/mid/deep).
    Returns dict: rank → {mean, p25, p50, p75, p90, p95} in bars.
    """
    X   = returns.values.reshape(-1, 1)
    seq = model.predict(X)
    order   = vol_ordered_states(model)
    rank_of = {s: i for i, s in enumerate(order)}

    # Encode as ranks
    seq_rank = np.array([rank_of[s] for s in seq])

    stats_out = {}
    for rank in range(model.n_components):
        runs = []
        count = 0
        for s in seq_rank:
            if s == rank:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)

        if not runs:
            continue
        r = np.array(runs)
        stats_out[rank] = {
            "n_runs":  len(r),
            "mean":    float(r.mean()),
            "p25":     float(np.percentile(r, 25)),
            "p50":     float(np.percentile(r, 50)),
            "p75":     float(np.percentile(r, 75)),
            "p90":     float(np.percentile(r, 90)),
            "p95":     float(np.percentile(r, 95)),
            "max":     int(r.max()),
        }
    return stats_out


# ── 6. Decode scan archive + posterior recalibration ─────────────────────────

def decode_and_recalibrate(
    model: GaussianHMM, returns_series: pd.Series, asset: str
) -> tuple:
    cfg = ASSET_CONFIG[asset]
    sa  = pd.read_csv(RESULTS / cfg["archive"], low_memory=False)
    sa["logged_at"] = pd.to_datetime(sa["logged_at"], utc=True, errors="coerce")
    sa["spot"]      = pd.to_numeric(sa["spot"],      errors="coerce")
    sa["tau_minutes"] = pd.to_numeric(sa["tau_minutes"], errors="coerce")
    if "spot_at_expiry" in sa.columns:
        sa["spot_at_expiry"] = pd.to_numeric(sa["spot_at_expiry"], errors="coerce")

    order       = vol_ordered_states(model)
    rank_of     = {s: i for i, s in enumerate(order)}
    sigma_raw   = {s: float(np.sqrt(model.covars_[s, 0, 0]))
                   for s in range(model.n_components)}

    ret_idx  = returns_series.index
    ret_vals = returns_series.values

    states_out, sraw_out = [], []

    for ts in sa["logged_at"]:
        pos = ret_idx.searchsorted(ts, side="right") - 1
        if pos < LOOKBACK_BARS or pd.isna(ts):
            states_out.append(np.nan); sraw_out.append(np.nan)
            continue
        window    = ret_vals[pos - LOOKBACK_BARS + 1: pos + 1].reshape(-1, 1)
        vit       = model.predict(window)
        raw_state = int(vit[-1])
        states_out.append(rank_of[raw_state])
        sraw_out.append(sigma_raw[raw_state])

    sa["hmm_vol_state"] = states_out
    sa["hmm_sigma_raw"] = sraw_out

    # Posterior recalibration from actual scan-period price moves
    recal_sigmas: dict = {}
    if "spot_at_expiry" in sa.columns and "resolved_yes" in sa.columns:
        sa_r = sa[sa["resolved_yes"].notna() & sa["hmm_vol_state"].notna()].copy()
        sa_r["log_move"] = np.log(
            sa_r["spot_at_expiry"].astype(float) / sa_r["spot"].astype(float)
        )
        sa_exp = (sa_r.sort_values("tau_minutes", ascending=False)
                      .drop_duplicates("close_ts")
                      .copy())

        print(f"\n  Posterior recalibration ({asset} scan-period σ by state):")
        for rank in sorted(sa_exp["hmm_vol_state"].dropna().unique()):
            sub = sa_exp[sa_exp["hmm_vol_state"] == rank].dropna(subset=["log_move"])
            if len(sub) < 10:
                print(f"    Rank {int(rank)}: n={len(sub)} too few, using training σ")
                recal_sigmas[int(rank)] = sigma_raw[order[int(rank)]] / math.sqrt(15)
                continue
            avg_tau = sub["tau_minutes"].astype(float).mean()
            sigma_pm = sub["log_move"].std() / math.sqrt(avg_tau)
            recal_sigmas[int(rank)] = sigma_pm
            ann_cal = sigma_pm * math.sqrt(MINS_PER_YEAR)
            raw_s   = sigma_raw[order[int(rank)]]
            ann_raw = raw_s / math.sqrt(15) * math.sqrt(MINS_PER_YEAR)
            print(f"    Rank {int(rank)}: n={len(sub):4d}  "
                  f"ann_raw={ann_raw:.1%}  ann_cal={ann_cal:.1%}  "
                  f"avg_tau={avg_tau:.1f}m")
    else:
        print("  Skipping posterior recalibration (missing spot_at_expiry or resolved_yes)")
        for rank in range(model.n_components):
            recal_sigmas[rank] = sigma_raw[order[rank]] / math.sqrt(15)

    scal_out = [
        recal_sigmas.get(int(r), np.nan) if not (isinstance(r, float) and np.isnan(r)) else np.nan
        for r in states_out
    ]
    sa["hmm_sigma_cal"] = scal_out

    valid = sum(1 for s in states_out if not (isinstance(s, float) and np.isnan(s)))
    print(f"\n  Decoded {valid:,}/{len(states_out):,} archive rows")

    return states_out, sraw_out, scal_out, recal_sigmas


# ── 7. Print report ───────────────────────────────────────────────────────────

def print_report(asset: str, model: GaussianHMM, returns: pd.Series,
                 erg: dict, sojourn: dict, recal: dict) -> str:
    order  = vol_ordered_states(model)
    ann_f  = math.sqrt(MINS_PER_YEAR / 15)
    A      = erg["transmat"]

    lines = [
        "=" * 65,
        f"{asset} 2-State Ergodic Vol HMM  ({TRAIN_FROM}+)",
        "=" * 65,
        "",
        "Transition matrix:",
        f"  R0→R0={A[order[0],order[0]]:.4f}  R0→R1={A[order[0],order[1]]:.4f}",
        f"  R1→R0={A[order[1],order[0]]:.4f}  R1→R1={A[order[1],order[1]]:.4f}",
        "",
        f"Ergodicity:",
        f"  Eigenvalue gap:  {erg['eigenvalue_gap']:.4f}",
        f"  Mixing time:     {erg['mixing_bars']:.1f} bars  "
        f"({erg['mixing_minutes']:.0f} min  ≈ {erg['mixing_minutes']/60:.1f}h)",
        f"  Stationary:      R0={erg['stationary'][order[0]]:.1%}  "
        f"R1={erg['stationary'][order[1]]:.1%}",
        "",
        "State emissions (vol-ranked R0=low, R1=high):",
    ]

    for rank, s in enumerate(order):
        mu     = float(model.means_[s, 0])
        sig_tr = float(np.sqrt(model.covars_[s, 0, 0]))
        sig_ca = recal.get(rank, sig_tr / math.sqrt(15))
        ann_tr = sig_tr * ann_f
        ann_ca = sig_ca * math.sqrt(MINS_PER_YEAR)
        X      = returns.values.reshape(-1, 1)
        seq    = model.predict(X)
        freq   = (seq == s).mean()
        lines.append(
            f"  R{rank} (state {s}): μ={mu*100:+.4f}%/bar  "
            f"σ_train={ann_tr:.1%}ann  σ_cal={ann_ca:.1%}ann  freq={freq:.1%}"
        )

    lines += ["", "Sojourn time statistics (in 15m bars):"]
    lines.append(f"  {'Rank':<6} {'n_runs':>8} {'mean':>8} {'p25':>6} "
                 f"{'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} {'max':>6}")
    for rank, s in sojourn.items():
        lines.append(
            f"  R{rank:<5} {s['n_runs']:>8,} {s['mean']:>8.1f} "
            f"{s['p25']:>6.0f} {s['p50']:>6.0f} {s['p75']:>6.0f} "
            f"{s['p90']:>6.0f} {s['p95']:>6.0f} {s['max']:>6}"
        )

    lines += [
        "",
        "Semi-Markov zone thresholds (suggested from sojourn p50/p90):",
        f"  early = 1 – {sojourn[1]['p25']:.0f} bars  "
        f"(R1 p25 = {sojourn[1]['p25']:.0f})",
        f"  mid   = {sojourn[1]['p25']:.0f} – {sojourn[1]['p90']:.0f} bars  "
        f"(R1 p25–p90)",
        f"  deep  = {sojourn[1]['p90']:.0f}+ bars  (R1 p90+)",
    ]

    report = "\n".join(lines)
    print(report)
    out = RESULTS / f"hmm_vol_regime_{asset.lower()}_calibration.txt"
    out.write_text(report)
    print(f"\n  Saved → {out.name}")
    return report


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["BTC", "ETH", "SOL"], required=True)
    args = parser.parse_args()
    asset = args.asset

    print(f"\n{'='*65}")
    print(f"  Training 2-state ergodic HMM for {asset}")
    print(f"{'='*65}\n")

    print("=== Step 1: Load 15m returns ===")
    returns = load_15m_returns(asset)

    print("\n=== Step 2: Train HMM ===")
    model = train_hmm(returns)

    print("\n=== Step 3: Ergodicity check ===")
    erg = ergodicity_check(model)
    print(f"  Eigenvalue gap: {erg['eigenvalue_gap']:.4f}  "
          f"mixing time: {erg['mixing_bars']:.1f} bars "
          f"({erg['mixing_minutes']:.0f} min)")

    print("\n=== Step 4: Sojourn time statistics ===")
    sojourn = sojourn_stats(model, returns)
    for rank, s in sojourn.items():
        print(f"  R{rank}: mean={s['mean']:.1f}  p50={s['p50']:.0f}  "
              f"p90={s['p90']:.0f}  max={s['max']}  (bars)")

    print("\n=== Step 5: Decode archive + posterior recalibration ===")
    states, sig_raw, sig_cal, recal = decode_and_recalibrate(model, returns, asset)

    describe_states(model, returns, recal_sigmas=recal)

    print("\n=== Step 6: Full report ===")
    print_report(asset, model, returns, erg, sojourn, recal)

    MODELS.mkdir(exist_ok=True)
    pkl_path = MODELS / f"hmm_ergodic_2state_{asset.lower()}_15m.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":                   model,
            "asset":                   asset,
            "n_states":                2,
            "lookback_bars":           LOOKBACK_BARS,
            "recal_sigmas_per_min":    recal,
            "train_from":              TRAIN_FROM,
            "ergodicity":              erg,
            "sojourn_stats":           sojourn,
        }, f)
    print(f"\n  Saved → {pkl_path.name}")
    print("\nDone.")
