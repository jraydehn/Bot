"""
S38 -- does the optimal ITM buffer depth shift with rv_ratio regime? If
touch risk is what a buffer protects against, and touch risk scales with
rv_ratio (validated), a fixed buffer target should be "enough" in cool
conditions but insufficient in hot ones. Tests this directly using the FULL
candidate population (btc_scan_archive_15m.csv -- every scanned contract,
not just taken trades) so the buffer-depth axis isn't confounded by the
model's own selection.

Bucket candidates by (offset_pct quintile) x (rv_ratio tercile) and look at
YES-side realized edge in each cell. If the buffer/regime interaction is
real, the edge-crossover point (where a given buffer stops being "enough")
should shift outward (need a bigger buffer) as rv_ratio rises.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
c1 = px["close"]; r1m = c1.pct_change()
rv2h, rv120h = r1m.rolling(120).std(), r1m.rolling(7200).std()
rv_ratio = (rv2h / rv120h.replace(0, np.nan)).rename("rv")

arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_market", "tau_minutes",
                           "offset_pct", "resolved_yes"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "offset_pct"])
arc = arc.sort_values("logged_at")
rv_df = rv_ratio.reset_index(); rv_df.columns = ["ts", "rv"]
arc = pd.merge_asof(arc, rv_df.sort_values("ts"), left_on="logged_at", right_on="ts", direction="backward")
arc = arc.dropna(subset=["rv"])
print(f"candidate rows: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")

# YES-side only: offset_pct > 0 means spot above strike (ITM for YES)
yes_cand = arc[arc["offset_pct"] > 0].copy()
print(f"YES-side (offset>0) candidates: {len(yes_cand)}")
yes_cand["edge"] = yes_cand["resolved_yes"] - yes_cand["p_market"]

q_rv = yes_cand["rv"].quantile([1/3, 2/3])
yes_cand["rv_bucket"] = pd.cut(yes_cand["rv"], [0, q_rv[1/3], q_rv[2/3], 100],
                               labels=["cool", "mid", "hot"])

# offset buckets: fine bins near the model's typical operating range (0.02-0.15%)
bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 100]
labels = ["0-.02", ".02-.04", ".04-.06", ".06-.08", ".08-.10", ".10-.15", ".15-.20", ".20-.30", ".30+"]
yes_cand["offset_bucket"] = pd.cut(yes_cand["offset_pct"], bins, labels=labels)

def tk_edge(sub):
    g = sub.groupby("contract_ticker")["edge"].mean()
    if len(g) < 15:
        return None
    boots = [g.sample(frac=1, replace=True, random_state=i).mean() for i in range(1500)]
    return len(g), g.mean(), float(np.mean(np.array(boots) <= 0))

print("\n=== YES-side ticker-clustered ITM edge, offset bucket x rv_ratio regime ===")
print(f"{'offset':>9s} | {'cool':>28s} | {'mid':>28s} | {'hot':>28s}")
for ob in labels:
    row = f"{ob:>9s} |"
    for rvb in ["cool", "mid", "hot"]:
        sub = yes_cand[(yes_cand["offset_bucket"] == ob) & (yes_cand["rv_bucket"] == rvb)]
        r = tk_edge(sub)
        if r:
            row += f" n={r[0]:4d} edge={r[1]:+.4f} P={r[2]:.3f} |"
        else:
            row += f" {'(thin)':>26s} |"
    print(row)

print("\n=== summary: WR by offset bucket, per regime (row-level, not ticker-clustered -- directional read) ===")
summ = yes_cand.groupby(["rv_bucket", "offset_bucket"]).agg(
    n=("resolved_yes", "size"), wr=("resolved_yes", "mean"), be=("p_market", "mean")).round(3)
summ["edge"] = (summ["wr"] - summ["be"]).round(4)
print(summ.to_string())
print("DONE_S38")
