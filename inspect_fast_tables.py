"""inspect_fast_tables.py — verify the fast p_up tables ARE calibrated per-fast-regime,
and quantify how much they differ across regimes (vs the production regime tables).
A useful regime makes the SAME (trend,rev) cell map to DIFFERENT p_up by regime."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
exec(open("build_fast_regime.py").read().split('print(f"\\n{')[0])  # setup: fr, TAB, fit_regime, build_tables, cell, base_all, SMOOTH_K, MIN_N

# --- fast regime tables (per-regime calibration, exactly as in the test) ---
feats=["ret2","ret6","ret12","rv6"]; lab=fit_regime(feats,3)
fast_tab,fast_rb=build_tables(lab)
fast_states=sorted(set(k[0] for k in fast_tab))
print("FAST regime per-state baselines (unconditional up%):", {int(s):round(fast_rb[s],4) for s in fast_states})
print("PROD regime baselines:", {r:round(TAB[r]["__baseline__"],4) for r in TAB})

# --- which (trend,rev) cells are well-populated in ALL fast regimes? compare their p_up ---
def cells_in(tab,states):
    from collections import defaultdict
    by=defaultdict(dict)
    for (rg,cl),v in tab.items(): by[cl][rg]=v
    return {cl:d for cl,d in by.items() if all(s in d for s in states)}
fast_cells=cells_in(fast_tab,fast_states)

print("\n=== sample (trend,rev) cells: p_up by FAST regime (state0/1/2) ===")
print(f"  {'cell':>8}  {'st0':>6}{'st1':>6}{'st2':>6}   spread(max-min)")
shown=0
for cl in sorted(fast_cells, key=lambda c:abs(int(c.split(',')[0]))+abs(int(c.split(',')[1]))):
    d=fast_cells[cl]; vals=[d[s] for s in fast_states]; sp=max(vals)-min(vals)
    if shown<12:
        print(f"  {cl:>8}  "+"".join(f"{v:>6.3f}" for v in vals)+f"   {sp:>6.3f}")
        shown+=1
fast_spreads=[max(d.values())-min(d.values()) for d in fast_cells.values()]

# --- production tables: same cells, p_up by Bull/Sideways/Bear ---
prod_states=["Bull","Sideways","Bear"]
prod_cells={}
for cl in fast_cells:
    vals={r:TAB[r].get(cl) for r in prod_states}
    if all(v is not None for v in vals.values()): prod_cells[cl]=vals
print("\n=== same cells: p_up by PRODUCTION regime (Bull/Sideways/Bear) ===")
print(f"  {'cell':>8}  {'Bull':>6}{'Sdwy':>6}{'Bear':>6}   spread")
shown=0
for cl in sorted(prod_cells, key=lambda c:abs(int(c.split(',')[0]))+abs(int(c.split(',')[1]))):
    d=prod_cells[cl]; vals=[d[r] for r in prod_states]; sp=max(vals)-min(vals)
    if shown<12:
        print(f"  {cl:>8}  "+"".join(f"{v:>6.3f}" for v in vals)+f"   {sp:>6.3f}")
        shown+=1
prod_spreads=[max(d.values())-min(d.values()) for d in prod_cells.values()]

print("\n"+"="*64)
print("  HOW MUCH DOES THE REGIME CHANGE p_up FOR THE SAME (trend,rev) CELL?")
print("="*64)
print(f"  FAST regime  cross-regime p_up spread:  mean={np.mean(fast_spreads):.4f}  median={np.median(fast_spreads):.4f}  (n={len(fast_spreads)} cells)")
print(f"  PROD regime  cross-regime p_up spread:  mean={np.mean(prod_spreads):.4f}  median={np.median(prod_spreads):.4f}  (n={len(prod_spreads)} cells)")
print(f"\n  Interpretation: bigger spread = the regime meaningfully RE-MAPS (trend,rev)->p_up.")
print(f"  If FAST spread << PROD spread, the fast (vol) regime barely changes p_up per cell")
print(f"  => per-regime calibration WAS done, but the vol regime doesn't differentiate DIRECTION.")
