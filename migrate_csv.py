"""One-time migration: fix paper_trades.csv column misalignment.

Removes the stale mu_drift header entry and adds z_shift/direction_strength.
"""
import csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results" / "paper_trades.csv"
NEW_HEADER = [
    "logged_at","decision_time","contract_ticker","close_ts","spot","strike","offset_pct",
    "p_market","p_market_source","p_yes_model","z_score","vol_60m","vol_60m_model",
    "vol_implied_kalshi","vol_ratio","spread","vol_eff","structure_bias","confirmation_bias",
    "confirmation_score","no_score","obi_score","obi_raw","obi_exchanges","vol_score",
    "vwap_score","ema_stretch_score","ema_alignment","rsi_value","rsi_regime",
    "z_shift","direction_strength",
    "raw_edge","net_edge","decision","side","neutral_gate","pure_edge_gate",
    "contracts_scanned","tau_minutes","gate_blocked","kelly_fraction","bet_fraction",
    "bet_amount","bankroll","resolved_yes","would_win","would_pnl",
]

with open(CSV_PATH) as f:
    rows = list(csv.reader(f))

old_header = rows[0]
data_rows  = rows[1:]

assert "mu_drift" in old_header, "mu_drift not found — already migrated?"
assert len(old_header) == 47, f"Expected 47-col header, got {len(old_header)}"

patched = []
for row in data_rows:
    if len(row) == 46:      # old rows: insert empty z_shift, direction_strength at pos 30-31
        row = row[:30] + ["", ""] + row[30:]
    # 48-col rows (new code) are already correct
    patched.append(row)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(NEW_HEADER)
    writer.writerows(patched)

print(f"Migrated {len(patched)} rows. New header has {len(NEW_HEADER)} columns.")
