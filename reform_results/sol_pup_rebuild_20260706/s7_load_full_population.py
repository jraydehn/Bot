"""
S7 -- Load FULL-column rows (not just the backfill's 7 columns) for the
SOL AGREE/DISAGREE populations, joined by (logged_at, contract_ticker),
so the comprehensive_rescue sweep can test every real CSV signal.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/sol_pup_rebuild_20260706"
CHAIN = [
    "results/paper_trades_sol_archive_20260407_122844.csv",
    "results/paper_trades_sol_archive_20260407_152310.csv",
    "results/paper_trades_sol_archive_20260415_1342_precal.csv",
    "results/paper_trades_sol.csv",
]

bf = pd.read_csv(f"{OUT}/sol_pup_v1_backfilled.csv", low_memory=False)
bf["logged_at_parsed"] = pd.to_datetime(bf["logged_at"], format="mixed", utc=True, errors="coerce")
key = bf.set_index(["logged_at_parsed", "contract_ticker"])[["agree", "p_sol", "be", "would_win", "would_pnl", "yw"]]

frames = []
for p in CHAIN:
    df = pd.read_csv(p, low_memory=False)
    df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
    frames.append(df)
full = pd.concat(frames, ignore_index=True)
full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")

merged = full.set_index(["logged_at_parsed", "contract_ticker"]).join(key, how="inner", rsuffix="_bf")
merged = merged.reset_index()
print(f"merged full-column population: {len(merged)} rows (expect ~762)")

agree = merged[merged["agree"] == True].copy()
disagree = merged[merged["agree"] == False].copy()
agree.to_csv(f"{OUT}/sol_agree_full.csv", index=False)
disagree.to_csv(f"{OUT}/sol_disagree_full.csv", index=False)
print(f"AGREE: {len(agree)}  DISAGREE: {len(disagree)}")
