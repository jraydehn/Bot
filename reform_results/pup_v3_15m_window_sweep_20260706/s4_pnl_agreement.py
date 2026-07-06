"""
S4 -- Agree/disagree PnL test for the W=15 native-15m model against REAL
taken BTC 15m trades, mirroring the earlier p_up_v3 (hourly-model) backfill
test. This one is actually more honest than that one: wf_preds_W15.parquet
holds genuine walk-forward OOS predictions (each week's model only ever saw
data up to that week's embargo boundary), not a single full-history fit
scored retroactively.

Causal alignment: for a trade logged at time L, use the prediction at the
last bar boundary T <= L (searchsorted 'right' - 1) -- the most recent
prediction actually available at decision time.

Agreement rule (same convention as v3_agree): side=yes agrees iff p>=0.50,
side=no agrees iff p<0.50. Read-only against paper_trades_btc15m.csv;
writes only to this directory.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"

wf = pd.read_parquet(f"{OUT}/wf_preds_W15.parquet").dropna(subset=["p"])
bar_index = wf.index
p_series = wf["p"]

raw = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
raw = raw[raw["asset"] == "BTC"].copy()
raw["logged_at_parsed"] = pd.to_datetime(raw["logged_at"], format="mixed", utc=True, errors="coerce")
taken = raw[raw["decision"] == "trade"].copy()


def lookup_p(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    # stale guard: prediction bar must be within 20 min of the trade
    # (bars are 15-min apart; allow a little slack for scan-cycle lag)
    if (logged_at - bar_index[idx]) > pd.Timedelta(minutes=20):
        return np.nan
    return float(p_series.iloc[idx])


taken["p_w15"] = taken["logged_at_parsed"].apply(lookup_p)
taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")

covered = taken.dropna(subset=["p_w15"]).copy()
print(f"taken BTC 15m trades: {len(taken)}, covered by W=15 backfill: {len(covered)}")
print(f"coverage window: {covered['logged_at_parsed'].min()} -> {covered['logged_at_parsed'].max()}")

is_yes = covered["side"].str.lower() == "yes"
covered["agree"] = np.where(is_yes, covered["p_w15"] >= 0.50, covered["p_w15"] < 0.50)
covered["week"] = covered["logged_at_parsed"].dt.isocalendar().week


def breakeven(sub):
    be = np.where(sub["side"].str.lower() == "yes", sub["p_market"], 1 - sub["p_market"])
    return np.nanmean(be)


def report(sub, label):
    if len(sub) == 0:
        print(f"{label}: n=0")
        return
    wr = sub["would_win"].mean()
    pnl = sub["would_pnl"].sum()
    be = breakeven(sub)
    print(f"{label:10s} n={len(sub):4d}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  PnL=${pnl:8.2f}")


print("\n=== W=15 native-15m model, agree/disagree (p>=0.50) ===")
report(covered, "ALL")
report(covered[covered["agree"]], "AGREE")
report(covered[~covered["agree"]], "DISAGREE")

print("\n=== per-week breakdown ===")
for wk in sorted(covered["week"].unique()):
    wkdf = covered[covered["week"] == wk]
    a = wkdf[wkdf["agree"]]
    d = wkdf[~wkdf["agree"]]
    a_pnl = a["would_pnl"].sum() if len(a) else 0.0
    d_pnl = d["would_pnl"].sum() if len(d) else 0.0
    full_pnl = wkdf["would_pnl"].sum()
    print(f"  wk{wk}: agree n={len(a):3d} pnl=${a_pnl:8.2f}   disagree n={len(d):3d} pnl=${d_pnl:8.2f}   "
          f"unfiltered_total=${full_pnl:8.2f}")

covered.to_csv(f"{OUT}/btc_15m_w15_backfilled.csv", index=False)
print(f"\nwrote {OUT}/btc_15m_w15_backfilled.csv")
