"""
S8 -- Sweep offset_pct + all reconstructed multi-TF indicators against the
crashing_no_gate blocked population, using the full row-level data (so PnL/
edge weight correctly by repeated scans) but significance-test with
ticker-level clustered bootstrap (per feedback_zero_lookahead_reconstruction
-- the 1172-row/92-ticker pseudo-replication bug). Any candidate is checked
for leakage into the two KNOWN real bad hours.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(331)
OUT = "reform_results/cg_hmm_20260708"

per_ticker = pd.read_csv(f"{OUT}/crashing_no_mtf_indicators.csv", low_memory=False)

b = pd.read_csv("results/blocked_trades.csv", low_memory=False)
b["logged_at"] = pd.to_datetime(b["logged_at"], format="mixed", utc=True, errors="coerce")
sub = b[b["gate_name"] == "hmm_pup_v3_crashing_no_gate"].copy()
sub["resolved_yes"] = pd.to_numeric(sub["resolved_yes"], errors="coerce")
sub["pm"] = pd.to_numeric(sub["pm"], errors="coerce")
sub["would_pnl"] = pd.to_numeric(sub["would_pnl"], errors="coerce")
sub["offset_pct"] = pd.to_numeric(sub["offset_pct"], errors="coerce")
sub = sub.dropna(subset=["resolved_yes", "pm", "would_pnl"]).copy()
sub["won"] = sub["resolved_yes"] == 0
sub["be"] = 1 - sub["pm"]
CLUSTER1 = (sub["logged_at"] >= pd.Timestamp("2026-07-07 09:00", tz="UTC")) & (sub["logged_at"] < pd.Timestamp("2026-07-07 10:00", tz="UTC"))
CLUSTER2 = (sub["logged_at"] >= pd.Timestamp("2026-07-08 07:00", tz="UTC")) & (sub["logged_at"] < pd.Timestamp("2026-07-08 08:00", tz="UTC"))
sub["bad_cluster"] = CLUSTER1 | CLUSTER2

MTF_COLS = [c for c in per_ticker.columns if any(c.startswith(p) for p in
            ["chg_", "bb_pctb_", "bb_width_", "rsi_", "stoch_", "donch_", "kc_pctb_"])]
print(f"joining {len(MTF_COLS)} reconstructed columns onto {len(sub)} rows via ticker map...")
for col in MTF_COLS + ["p_up_v3", "p_chg_1h", "p_ma6h"]:
    m = per_ticker.set_index("contract_ticker")[col].to_dict()
    sub[col] = sub["contract_ticker"].map(m)

ALL_CANDS = MTF_COLS + ["offset_pct"]
print(f"total candidates including offset_pct: {len(ALL_CANDS)}")


def ticker_boot(df, n_boot=5000):
    if df["contract_ticker"].nunique() < 5:
        return df["contract_ticker"].nunique(), np.nan, np.nan
    pt = df.groupby("contract_ticker").apply(
        lambda g: g["won"].astype(float).mean() - g["be"].mean(), include_groups=False)
    e = pt.values
    n_t = len(e)
    means = np.array([e[rng.integers(0, n_t, n_t)].mean() for _ in range(n_boot)])
    return n_t, means.mean(), (means <= 0).mean()


found = []
n_tests = 0
for feat in ALL_CANDS:
    col = sub[feat]
    nn = col.notna().sum()
    if nn < 200:  # need decent row coverage across the 92-ticker population
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for d, mask in [(">=", col >= th), ("<", col < th)]:
            n_tests += 1
            s = sub[mask.fillna(False)]
            n_t = s["contract_ticker"].nunique()
            if n_t < 15 or (sub["contract_ticker"].nunique() - n_t) < 15:
                continue
            leak = s["bad_cluster"].sum()
            wr = s["won"].mean()
            be = s["be"].mean()
            found.append({"feature": feat, "split": f"{d}{th:.4g}(q{q:.1f})", "rows": len(s),
                         "tickers": n_t, "edge": wr - be, "leak": leak, "pnl": s["would_pnl"].sum()})

fd = pd.DataFrame(found)
print(f"\n{n_tests} splits tested across {len(set(fd['feature']))} candidate features with sufficient coverage")
print(f"\ncandidates with >=20 tickers, zero bad-event leakage, positive edge:")
resc = fd[(fd["edge"] > 0) & (fd["leak"] == 0) & (fd["tickers"] >= 20)].sort_values("edge", ascending=False)
print(resc.head(20).round(4).to_string(index=False))

print(f"\n--- ticker-clustered bootstrap on top 10 (by edge, tickers>=20, zero leak) ---")
for _, r in resc.head(10).iterrows():
    feat, split = r["feature"], r["split"]
    col = sub[feat]
    if split.startswith(">="):
        thv = float(split.split("(")[0][2:]); mask = col >= thv
    else:
        thv = float(split.split("(")[0][1:]); mask = col < thv
    s = sub[mask.fillna(False)]
    n_t, edge, p = ticker_boot(s)
    print(f"  {feat} {split}: rows={len(s)} tickers={n_t}  edge={edge:+.4f}  P(<=0)={p:.4f}  pnl=${s['would_pnl'].sum():+.2f}")

sub.to_csv(f"{OUT}/crashing_no_mtf_rowlevel.csv", index=False)
print("\nDONE_S8")
