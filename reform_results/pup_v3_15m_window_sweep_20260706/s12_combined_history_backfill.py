"""
S12 -- Same applicability test as s11, but stitching together the archived
paper_trades snapshots to get much more real-trade history instead of just
the current live file (which only holds ~3 weeks for hourly, ~6 for 15m
because the CSV gets rotated/archived on every major model change).

Archive chains identified by date range (verified no time overlap at each
boundary):
  HOURLY:  precal(04-07->04-15) -> pre_branched_drift(04-15->05-25)
           -> pre_regime_pup(05-25->06-16) -> paper_trades.csv(06-17->now)
  15m:     btc15m_pre_branched_drift(05-11->05-25) -> paper_trades_btc15m.csv(05-25->now)

Older hourly archives predate the 'asset' column (they're pure BTC-only
files from before multi-asset consolidation) -- treated as asset=BTC.

Caveat worth stating plainly: this stitches together several different
model/gate eras (see project_model_version_history.md) -- the SELECTION
of which trades got taken varied across eras (different gates were live
at different times). That doesn't invalidate the outcome data itself
(would_win/would_pnl are facts about the market, not the bot), but it
does mean "n trades in state X" spans populations chosen by different
rulebooks, not one consistent policy the whole time.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
NEEDED = ["logged_at", "decision", "side", "would_win", "would_pnl", "p_market", "contract_ticker"]

HOURLY_CHAIN = [
    "results/paper_trades_archive_20260415_1342_precal.csv",
    "results/paper_trades_archive_20260525_1432_pre_branched_drift.csv",
    "results/paper_trades_pre_regime_pup_20260616.csv",
    "results/paper_trades.csv",
]
M15_CHAIN = [
    "results/paper_trades_btc15m_archive_20260525_1432_pre_branched_drift.csv",
    "results/paper_trades_btc15m.csv",
]


def load_chain(paths, asset_filter):
    frames = []
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        use = [c for c in NEEDED if c in cols]
        if "asset" in cols:
            use = use + ["asset"]
        df = pd.read_csv(p, usecols=use, low_memory=False)
        if "asset" in df.columns:
            df = df[df["asset"] == "BTC"]
        else:
            df["asset"] = "BTC"
        df["_source"] = p
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["logged_at_parsed"] = pd.to_datetime(full["logged_at"], format="mixed", utc=True, errors="coerce")
    before = len(full)
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    full = full.sort_values("logged_at_parsed")
    print(f"  loaded {before} rows across {len(paths)} files -> {len(full)} after de-dup "
          f"({full['logged_at_parsed'].min()} -> {full['logged_at_parsed'].max()})")
    return full


print("HOURLY chain:")
hourly_raw = load_chain(HOURLY_CHAIN, "BTC")
print("\n15m chain:")
m15_raw = load_chain(M15_CHAIN, "BTC")

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


def prep(df):
    taken = df[df["decision"] == "trade"].copy()
    taken["state3"] = taken["logged_at_parsed"].apply(lookup_state)
    taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
    taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
    taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")
    taken["side"] = taken["side"].str.lower()
    taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
    taken["year"] = taken["logged_at_parsed"].dt.isocalendar().year
    taken["yw"] = taken["year"].astype(str) + "-W" + taken["week"].astype(str).str.zfill(2)
    covered = taken.dropna(subset=["state3", "would_pnl"]).copy()
    covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])
    return covered


hourly = prep(hourly_raw)
m15 = prep(m15_raw)


def report(covered, label):
    print(f"\n{'='*95}\n{label}  n_taken_total={len(covered)}\n{'='*95}")
    print(f"{'state':10s} {'side':5s} {'n':>5s} {'WR':>6s} {'BE':>6s} {'edge':>7s} {'PnL':>10s} "
          f"{'distinct_wks':>12s} {'worst_wk_share':>14s}")
    rows = []
    for st in ["rising", "neutral", "crashing"]:
        for side in ["yes", "no"]:
            sub = covered[(covered["state3"] == st) & (covered["side"] == side)]
            n = len(sub)
            if n == 0:
                continue
            wr = sub["would_win"].mean()
            be = sub["be"].mean()
            pnl = sub["would_pnl"].sum()
            wk_pnl = sub.groupby("yw")["would_pnl"].sum()
            n_weeks = len(wk_pnl)
            worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
            print(f"{st:10s} {side:5s} {n:5d} {wr:6.3f} {be:6.3f} {wr-be:+7.3f} {pnl:10.2f}  "
                  f"{n_weeks:12d} {worst_share:13.1f}%")
            rows.append({"state": st, "side": side, "n": n, "wr": wr, "be": be, "pnl": pnl,
                        "n_weeks": n_weeks, "worst_wk_share": worst_share})
    return pd.DataFrame(rows)


r_hourly = report(hourly, "BTC HOURLY -- combined history")
r_15m = report(m15, "BTC 15m -- combined history")

print(f"\n{'='*95}\nAPPLICABILITY VERDICT (n>=30, distinct_weeks>=4, worst_wk_share<60%)\n{'='*95}")
for label, r in [("HOURLY", r_hourly), ("15m", r_15m)]:
    for _, row in r.iterrows():
        ok_n = row["n"] >= 30
        ok_weeks = row["n_weeks"] >= 4
        ok_conc = row["worst_wk_share"] < 60
        verdict = "ACTIONABLE" if (ok_n and ok_weeks and ok_conc) else \
                 ("TOO THIN (n<30)" if not ok_n else
                  "TOO FEW WEEKS" if not ok_weeks else
                  "SINGLE-WEEK DRIVEN")
        print(f"{label:7s} {row['state']:10s} {row['side']:5s} n={row['n']:4d}  wks={row['n_weeks']:2d}  "
              f"worst_wk_share={row['worst_wk_share']:5.1f}%  ->  {verdict}")
