"""
analysis_gate_attribution.py  (scratch analysis — does NOT touch live trading)

Per-gate quality attribution for the BTC hourly model using blocked_trades.csv.
Dedupes to one row per (gate, contract) to kill the ~26x scan-cycle multi-count,
then scores each gate on a FLAT 1-contract stake from realized outcome + entry pm.

Metric per gate (flat unit = 1 contract, pnl in $ of notional):
  total_unit_pnl > 0  -> gate is BLOCKING net-WINNERS  -> edge left on table (BAD)
  total_unit_pnl < 0  -> gate is BLOCKING net-LOSERS   -> saving money (GOOD)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
FEE = 0.07

def unit_pnl(side, pm, ry):
    fee = FEE * min(pm, 1 - pm)
    if side == "yes":
        return (1 - pm - fee) if ry == 1 else -(pm + fee)
    else:
        return (pm - fee) if ry == 0 else -(1 - pm + fee)

b = pd.read_csv("results/blocked_trades.csv", low_memory=False)
b = b[b["asset"] == "BTC"].copy()
for c in ["pm", "net_edge", "would_pnl", "resolved_yes", "tau_minutes"]:
    b[c] = pd.to_numeric(b[c], errors="coerce")
b = b.dropna(subset=["pm", "resolved_yes", "side", "gate_name"])
b = b[b["side"].isin(["yes", "no"])]

# dedupe: one row per (gate, contract) — representative entry = median pm across cycles
ded = (b.groupby(["gate_name", "contract_ticker", "side"])
         .agg(pm=("pm", "median"),
              ry=("resolved_yes", "first"),
              net_edge=("net_edge", "median"),
              would_pnl=("would_pnl", "mean"),
              n_blocks=("pm", "size"))
         .reset_index())
ded["ry"] = ded["ry"].round().astype(int)
ded["upnl"] = ded.apply(lambda r: unit_pnl(r["side"], r["pm"], r["ry"]), axis=1)
ded["won"] = ded["upnl"] > 0

rows = []
for g, grp in ded.groupby("gate_name"):
    n = len(grp)
    wr = grp["won"].mean()
    pm_avg = grp["pm"].mean()
    # breakeven WR: avg cost basis incl fee
    be = grp.apply(lambda r: (r["pm"] if r["side"] == "yes" else 1 - r["pm"])
                              + FEE * min(r["pm"], 1 - r["pm"]), axis=1).mean()
    tot = grp["upnl"].sum()
    mean_edge = grp["upnl"].mean()
    wp = grp["would_pnl"].mean() * n  # deduped kelly-$ view (mean size * count)
    side_mix = grp["side"].value_counts().to_dict()
    rows.append(dict(gate=g, n=n, WR=wr, BE_WR=be, edge_per=mean_edge,
                     flat_pnl=tot, kelly_pnl=wp,
                     yes=side_mix.get("yes", 0), no=side_mix.get("no", 0)))

res = pd.DataFrame(rows).sort_values("flat_pnl", ascending=False)
pd.set_option("display.width", 200)
print("="*110)
print("BTC HOURLY GATE ATTRIBUTION  (deduped per gate+contract, flat 1-contract stake)")
print("  flat_pnl > 0 => gate BLOCKS WINNERS (bad / candidate to relax)")
print("  flat_pnl < 0 => gate BLOCKS LOSERS  (good / saving money)")
print("="*110)
print(f"{'gate':<34}{'n':>6}{'WR':>7}{'BE_WR':>7}{'edge/ct':>9}{'flat_$':>10}{'kelly_$':>11}  side(y/n)")
print("-"*110)
for _, r in res.iterrows():
    print(f"{r['gate']:<34}{r['n']:>6}{r['WR']:>7.1%}{r['BE_WR']:>7.1%}"
          f"{r['edge_per']:>+9.3f}{r['flat_pnl']:>+10.1f}{r['kelly_pnl']:>+11.0f}"
          f"  {int(r['yes'])}/{int(r['no'])}")
print("-"*110)
print(f"{'TOTAL blocked (flat $ if all taken)':<34}{res['n'].sum():>6}"
      f"{'':>21}{res['flat_pnl'].sum():>+10.1f}{res['kelly_pnl'].sum():>+11.0f}")
