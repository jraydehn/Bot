"""
discover_tree_gates.py

Decision tree gate discovery with MCPT validation.

Trains a shallow DecisionTreeRegressor on scan archive features, with target
edge = won_no - p_market (negative = losing NO bet). Extracts leaf rules where
mean edge is most negative (worst NO conditions → block them), then validates
via MCPT: permute outcomes 500×, retrain tree each time, check whether any
leaf on real data beats the best leaf on shuffled data.

Key distinction from the LGBM shadow model (p_gbdt):
  - LGBM: continuous score, needs threshold calibration
  - This tree: INTERPRETABLE CONJUNCTIVE RULES directly usable in runner gates

Output:
  - Top losing-leaf rules ranked by P&L improvement
  - MCPT p-value: does the best discovered rule survive overfitting?

Usage:
  python3 discover_tree_gates.py [--max-depth 4] [--min-leaf 300] [--n-perms 500]
"""
import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
RESULTS = BASE / "results"

FEE      = 0.07
FLAT_BET = 10.0

# ── Features ──────────────────────────────────────────────────────────────────
# Predictors available at entry time; no lookahead.
# Grouped by layer so the tree printout is readable.
# Signal-only features — deliberately excludes offset_pct and p_market.
# Those encode contract geometry already handled by existing pm-floor /
# highpm gates; including them lets the tree trivially rediscover "deep
# OTM YES is a bad NO bet," masking the actual signal of interest.
FEATURES = [
    # Multi-TF stochastic
    "stoch_k_5m", "stoch_k_15m", "stoch_k_1h",
    # 1h momentum
    "rsi_1h", "bp_1h", "chg_1h", "macd_hist_1h", "adx_1h",
    # 4h momentum
    "macd_hist_4h", "adx_4h",
    # Trend / structure
    "ema_stack_bias", "composite_trend", "composite_rev",
    "vwap_stretch_score", "ema_stretch_score",
    # Volume / microstructure
    "rvol_1h", "vol_score", "funding_bias", "obi_score",
    "liq_score", "liq_bias",
    # Short-term price action
    "chg_5m", "chg_10m", "chg_30m", "bp_5m",
    # Expiry context (valid signal — shorter tau = tighter window)
    "tau_minutes",
]

# Contract window: only rows that fall in a p_market range where the runner
# would actually consider a NO bet. Below 0.10 and above 0.90 are already
# hard-blocked by pm-floor / highpm gates.
PM_LO, PM_HI = 0.10, 0.90

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--max-depth",  type=int,   default=4)
parser.add_argument("--min-leaf",   type=int,   default=300,
                    help="min_samples_leaf — raise to get broader rules")
parser.add_argument("--n-perms",    type=int,   default=500)
parser.add_argument("--top-n",      type=int,   default=8,
                    help="Number of worst-edge leaves to report")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
_pq = RESULTS / "btc_scan_archive_hmm.parquet"
_cs = RESULTS / "btc_scan_archive.csv"
print("Loading archive …", end=" ", flush=True)
sa = pd.read_parquet(_pq) if _pq.exists() else pd.read_csv(_cs, low_memory=False)
sa = sa[sa["resolved_yes"].notna()].copy()
sa["p_market"] = pd.to_numeric(sa["p_market"], errors="coerce")
sa["won_no"]   = (pd.to_numeric(sa["resolved_yes"], errors="coerce") == 0).astype(float)
sa["edge"]     = sa["won_no"] - sa["p_market"]   # target: + = profitable, - = losing
print(f"{len(sa)} rows.")

# Restrict to tradeable pm window first — removes deep OTM/ITM contracts
# already handled by pm-floor / highpm gates, so the tree sees the
# interesting mid-range where signal features actually discriminate.
sa = sa[(sa["p_market"] >= PM_LO) & (sa["p_market"] <= PM_HI)].reset_index(drop=True)
print(f"Rows after pm filter [{PM_LO},{PM_HI}]: {len(sa)}")

# Coerce all feature columns
avail = []
for f in FEATURES:
    col = "stoch_k" if (f == "stoch_k_1h" and "stoch_k_1h" not in sa.columns) else f
    if col in sa.columns:
        sa[f] = pd.to_numeric(sa[col], errors="coerce")
        avail.append(f)
    else:
        avail.append(None)

USED_FEATS = [f for f in avail if f is not None]
print(f"Features: {len(USED_FEATS)}/{len(FEATURES)} available")

# Drop rows missing >30% of features
X_full = sa[USED_FEATS].copy()
row_ok  = X_full.notna().mean(axis=1) >= 0.70
sa      = sa[row_ok].reset_index(drop=True)
X_full  = X_full[row_ok].reset_index(drop=True)

# Fill remaining NaN with column median
col_med = X_full.median()
X_full  = X_full.fillna(col_med)

pm_arr  = sa["p_market"].astype(float).values
won_arr = sa["won_no"].astype(float).values
edge_arr= sa["edge"].astype(float).values
X       = X_full.values.astype(float)
pm_bins = np.floor(pm_arr * 10).astype(int)

print(f"Rows after feature filter: {len(sa)}\n")

# ── P&L helpers ───────────────────────────────────────────────────────────────

def gate_pnl(pm, won, block_mask):
    keep = ~block_mask
    p    = pm[keep]; w = won[keep]
    pnl  = np.where(w.astype(bool), (1-p)*(1-FEE), -p*(1-FEE)) * FLAT_BET
    return float(pnl.sum())


def ungated_pnl(pm, won):
    return gate_pnl(pm, won, np.zeros(len(pm), dtype=bool))

# ── Tree training + leaf extraction ───────────────────────────────────────────

def train_tree(X, edge, max_depth, min_leaf):
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=0,
    )
    tree.fit(X, edge)
    return tree


def extract_leaf_gates(tree, X, pm, won, feature_names, top_n=10):
    """Return list of (delta, mask, rule_str) for the worst-edge leaves."""
    leaf_ids  = tree.apply(X)
    base_pnl  = ungated_pnl(pm, won)
    leaves    = []

    # Get decision path text for rule extraction
    tree_text = export_text(tree, feature_names=feature_names)

    for leaf_id in np.unique(leaf_ids):
        mask     = leaf_ids == leaf_id
        n_leaf   = mask.sum()
        mean_edge= float(won[mask].mean() - pm[mask].mean())
        if mean_edge >= 0:
            continue                       # profitable leaf — keep trading it
        gated_pnl = gate_pnl(pm, won, mask)
        delta     = gated_pnl - base_pnl
        wr        = float(won[mask].mean())
        bkev      = float(pm[mask].mean())
        leaves.append((delta, n_leaf, wr, bkev, mask, mean_edge, leaf_id))

    leaves.sort(key=lambda x: -x[0])      # highest P&L improvement first
    return leaves[:top_n]


def rule_for_leaf(tree, leaf_id, feature_names):
    """Extract the decision path for a given leaf node as a readable string."""
    from sklearn.tree import _tree
    t         = tree.tree_
    node_feat = t.feature
    threshold = t.threshold
    children_left  = t.children_left
    children_right = t.children_right

    # Walk from root to find path to this leaf
    def find_path(node, path):
        if children_left[node] == _tree.TREE_LEAF:
            if node == leaf_id:
                return path
            return None
        left  = find_path(children_left[node],
                          path + [(feature_names[node_feat[node]], "<=", threshold[node])])
        if left is not None:
            return left
        return find_path(children_right[node],
                         path + [(feature_names[node_feat[node]], ">",  threshold[node])])

    path = find_path(0, [])
    if path is None:
        return "(rule not found)"
    parts = [f"{feat} {op} {thr:.2f}" for feat, op, thr in path]
    return "  AND  ".join(parts)


# ── MCPT ──────────────────────────────────────────────────────────────────────

def best_leaf_delta(tree, X, pm, won):
    """Best P&L improvement across all losing leaves on this tree."""
    leaf_ids = tree.apply(X)
    base     = ungated_pnl(pm, won)
    best     = 0.0
    for lid in np.unique(leaf_ids):
        mask  = leaf_ids == lid
        if float(won[mask].mean() - pm[mask].mean()) >= 0:
            continue
        delta = gate_pnl(pm, won, mask) - base
        if delta > best:
            best = delta
    return best


def run_mcpt(X, pm_arr, won_arr, pm_bins, real_best_delta,
             max_depth, min_leaf, n_perms):
    rng  = np.random.default_rng(42)
    perm_best = []
    print(f"  Running {n_perms} permutations …", end=" ", flush=True)
    for i in range(n_perms):
        won_p = won_arr.copy()
        for b in range(11):
            idx = np.where(pm_bins == b)[0]
            if len(idx) > 1:
                vals = won_p[idx]; rng.shuffle(vals); won_p[idx] = vals
        edge_p = won_p - pm_arr
        t      = train_tree(X, edge_p, max_depth, min_leaf)
        perm_best.append(best_leaf_delta(t, X, pm_arr, won_p))
        if (i + 1) % 100 == 0:
            print(f"{i+1}", end=" ", flush=True)
    print("done.")

    pd_arr = np.array(perm_best)
    p_val  = float((pd_arr >= real_best_delta).mean())
    pct5   = float(np.percentile(pd_arr, 5))
    pct50  = float(np.percentile(pd_arr, 50))
    pct95  = float(np.percentile(pd_arr, 95))
    return p_val, pd_arr, pct5, pct50, pct95


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Training tree (max_depth={args.max_depth}, min_leaf={args.min_leaf}) …",
      end=" ", flush=True)
tree = train_tree(X, edge_arr, args.max_depth, args.min_leaf)
print(f"done.  Leaves: {tree.get_n_leaves()}")

leaves = extract_leaf_gates(tree, X, pm_arr, won_arr, USED_FEATS, args.top_n)
real_best_delta = leaves[0][0] if leaves else 0.0

print(f"\n{'='*72}")
print(f"  TOP LOSING LEAVES  (max_depth={args.max_depth}, min_leaf={args.min_leaf})")
print(f"{'='*72}\n")

base = ungated_pnl(pm_arr, won_arr)
print(f"  Baseline (all bets): ${base:+,.2f}\n")

for rank, (delta, n, wr, bkev, mask, mean_edge, lid) in enumerate(leaves, 1):
    rule = rule_for_leaf(tree, lid, USED_FEATS)
    pnl_blocked = gate_pnl(pm_arr, won_arr, mask)
    print(f"  Leaf #{rank}  Δ=${delta:+,.2f}  n={n}  WR={wr:.1%}  bkev={bkev:.1%}  "
          f"edge={mean_edge:+.1%}")
    print(f"    Rule : {rule}")
    print()

# MCPT
print(f"{'─'*72}")
print(f"  MCPT  (n={args.n_perms} permutations)")
print(f"  H0: best discovered leaf improvement ≤ what random data produces")
print(f"{'─'*72}\n")

p_val, pd_arr, pct5, pct50, pct95 = run_mcpt(
    X, pm_arr, won_arr, pm_bins,
    real_best_delta, args.max_depth, args.min_leaf, args.n_perms,
)
sig = p_val < 0.05

print(f"\n  Real best leaf Δ  : ${real_best_delta:+,.2f}")
print(f"  Null p5/p50/p95   : ${pct5:+,.2f} / ${pct50:+,.2f} / ${pct95:+,.2f}")
print(f"  p-value           : {p_val:.3f}  ({'SIGNIFICANT ✓' if sig else 'not significant ✗'})")

if sig:
    print(f"\n  ✓ The best discovered rule survives MCPT — genuine signal.")
    print(f"    Top candidate for paper_trade_runner.py:")
    if leaves:
        d, n, wr, bkev, mask, me, lid = leaves[0]
        print(f"      Rule : {rule_for_leaf(tree, lid, USED_FEATS)}")
        print(f"      Stats: n={n}  WR={wr:.1%}  bkev={bkev:.1%}  Δ=${d:+,.2f}")
else:
    print(f"\n  ✗ No rule survives MCPT — tree is fitting noise.")
    print(f"    Try --min-leaf {args.min_leaf * 2} or collect more data before using tree rules.")

print(f"\n{'='*72}")
