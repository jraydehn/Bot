"""
S2 -- Corrected efficacy/PnL analysis for the BTC hourly VWAP MTF HMM.

The first pass (train_backfill_vwap_hmm_1h.py) compared composite_p_up
(a bar-level directional signal, tightly clustered ~0.47-0.52) directly
against p_market (a strike-specific price ranging 0.02-0.99) as if they
were on the same footing -- comparing apples to oranges, producing a
nonsensical 1.3% baseline WR in the PnL section. This script fixes it by
building a proper strike-adjusted p_model via the same log-normal + drift
transform used throughout this codebase:
  z_strike = log(strike/spot) / sigma_tau      sigma_tau = vol_eff * sqrt(tau_minutes)
  p_model  = 1 - Phi(z_strike - Phi^-1(composite_p_up) * k_drift)
k_drift=0.90 matches BTC's current live YES calibration (kyes_090 reform);
applied symmetrically here as a single-model diagnostic simplification --
NOT a reproduction of BTC's real asymmetric dual YES/NO models. Sufficient
for relative state-conditional WR/edge screening, not for exact live PnL.

The trained HMM model + state backfill from the first pass are unaffected
and reused as-is (results/btc_vwap_hmm_states_1h.csv, models/hmm_vwap_mtf_btc_1h.pkl).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = "reform_results/vwap_hmm_1h_20260708"
rng = np.random.default_rng(11)

arch = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)


def parse_logged_at_mixed(series):
    def _to_utc(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_to_utc(v) for v in series], utc=True)


arch["logged_at"] = parse_logged_at_mixed(arch["logged_at"])
null_la = arch["logged_at"].isna()
if null_la.any():
    close_ts = parse_logged_at_mixed(arch.loc[null_la, "close_ts"])
    tau_min = pd.to_numeric(arch.loc[null_la, "tau_minutes"], errors="coerce").fillna(30)
    arch.loc[null_la, "logged_at"] = close_ts - pd.to_timedelta(tau_min, unit="m")

sc = pd.read_csv(f"{OUT}/../../results/btc_vwap_hmm_states_1h.csv"
                 if False else "results/btc_vwap_hmm_states_1h.csv")
sc["logged_at"] = parse_logged_at_mixed(sc["logged_at"])
arch = arch.merge(sc[["logged_at", "contract_ticker", "vwap_hmm_state"]],
                  on=["logged_at", "contract_ticker"], how="left")

for col in ["resolved_yes", "p_market", "composite_p_up", "spot", "strike", "tau_minutes", "vol_eff"]:
    arch[col] = pd.to_numeric(arch[col], errors="coerce")
arch_res = arch.dropna(subset=["resolved_yes", "p_market", "composite_p_up", "spot", "strike",
                               "tau_minutes", "vol_eff", "vwap_hmm_state"]).copy()
arch_res = arch_res[(arch_res["tau_minutes"] > 0) & (arch_res["vol_eff"] > 0)
                    & (arch_res["p_market"] > 0) & (arch_res["p_market"] < 1)]
print(f"usable resolved+state rows: {len(arch_res):,}")

K_DRIFT = 0.90
sigma_tau = arch_res["vol_eff"] * np.sqrt(arch_res["tau_minutes"])
z_strike = np.log(arch_res["strike"] / arch_res["spot"]) / sigma_tau
z_drift = norm.ppf(arch_res["composite_p_up"].clip(0.01, 0.99)) * K_DRIFT
arch_res["p_model_yes"] = np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99)

arch_res["edge_yes"] = arch_res["p_model_yes"] - arch_res["p_market"]
arch_res["edge_no"] = arch_res["p_market"] - arch_res["p_model_yes"]
arch_res["model_side"] = np.where(arch_res["edge_yes"] >= arch_res["edge_no"], "YES", "NO")

overall_yes_wr = arch_res.loc[arch_res["model_side"] == "YES", "resolved_yes"].mean()
overall_no_wr = 1 - arch_res.loc[arch_res["model_side"] == "NO", "resolved_yes"].mean()
overall_yes_be = arch_res.loc[arch_res["model_side"] == "YES", "p_market"].mean()
overall_no_be = 1 - arch_res.loc[arch_res["model_side"] == "NO", "p_market"].mean()
n_yes, n_no = (arch_res["model_side"] == "YES").sum(), (arch_res["model_side"] == "NO").sum()
print(f"\nBaseline  YES: WR={overall_yes_wr:.3f} BE={overall_yes_be:.3f} n={n_yes}  "
      f"|  NO: WR={overall_no_wr:.3f} BE={overall_no_be:.3f} n={n_no}")

print(f"\n{'State':<7} {'Side':<5} {'n':>7} {'WR':>6} {'BEven':>6} {'edge':>7} {'ΔWR':>7}  Verdict")
print("-" * 62)
results = []
for s in sorted(arch_res["vwap_hmm_state"].unique()):
    sub_s = arch_res[arch_res["vwap_hmm_state"] == s]
    for side, base_wr in [("YES", overall_yes_wr), ("NO", overall_no_wr)]:
        sub = sub_s[sub_s["model_side"] == side]
        if len(sub) < 20:
            continue
        if side == "YES":
            wr = sub["resolved_yes"].mean(); be_wr = sub["p_market"].mean()
        else:
            wr = 1 - sub["resolved_yes"].mean(); be_wr = 1 - sub["p_market"].mean()
        delta = wr - base_wr
        edge = wr - be_wr
        verdict = "BLOCK cand." if edge < -0.05 else ("BOOST cand." if edge > 0.05 and delta > 0.05 else "neutral")
        print(f"  {int(s):<5} {side:<5} {len(sub):>7} {wr:>6.3f} {be_wr:>6.3f} {edge:>+7.3f} {delta:>+7.3f}  {verdict}")
        results.append(dict(state=int(s), side=side, n=len(sub), wr=wr, be_wr=be_wr, delta=delta, edge=edge))

print("\n=== Gate candidates (|edge| > 0.03, n >= 30) ===")
candidates = [r for r in results if abs(r["edge"]) > 0.03 and r["n"] >= 30]
for r in sorted(candidates, key=lambda x: x["edge"]):
    action = "BLOCK" if r["edge"] < 0 else "BOOST"
    print(f"  State {r['state']} {r['side']:3s}: WR={r['wr']:.3f} BE={r['be_wr']:.3f} "
          f"edge={r['edge']:+.3f} ΔWR={r['delta']:+.3f} n={r['n']} -> {action}")

# ── bootstrap significance for each candidate ────────────────────────────────
print("\n=== Bootstrap significance (trade-level, n_boot=4000) ===")
arch_res["be"] = np.where(arch_res["model_side"] == "YES", arch_res["p_market"], 1 - arch_res["p_market"])
arch_res["won"] = np.where(arch_res["model_side"] == "YES", arch_res["resolved_yes"] == 1,
                           arch_res["resolved_yes"] == 0)
arch_res["trade_edge"] = arch_res["won"].astype(float) - arch_res["be"]

def boot_p(edges, n_boot=4000):
    e = edges.values; n = len(e)
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()

for r in sorted(candidates, key=lambda x: x["edge"]):
    sub = arch_res[(arch_res["vwap_hmm_state"] == r["state"]) & (arch_res["model_side"] == r["side"])]
    m, lo, hi, p = boot_p(sub["trade_edge"])
    p_report = p if r["edge"] < 0 else (1 - p)
    print(f"  State {r['state']} {r['side']}: edge_mean={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
          f"P(wrong-direction)={p_report:.4f}")

arch_res.to_csv(f"{OUT}/corrected_full_analysis.csv", index=False)
print(f"\nsaved {OUT}/corrected_full_analysis.csv")
