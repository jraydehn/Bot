"""
simulate_price_movement_gates.py

Gate simulations based on price-movement correlation analysis.
Uses flat-bankroll PnL so magnitudes are comparable across scenarios.

Gates tested:
  1h  — vol_eff quartile (BTC/ETH/SOL)
  1h  — adx_1h threshold (ETH, SOL)
  1h  — squeeze_1h gate (ETH, SOL)
  15m — consec_dir_15m gate (BTC, ETH, SOL)  [small n — interpret carefully]
  15m — upper_wick_15m gate (BTC, ETH, SOL)
  15m — ls_long_pct gate (ETH)
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path("results")
FLAT_BET = 25.0   # fixed bet size for all simulations

SEP  = "=" * 72
SEP2 = "-" * 56


# ── helpers ─────────────────────────────────────────────────────────────────

def flat_pnl(side: str, p_market: float, win: bool) -> float:
    if side == "yes":
        return FLAT_BET * (1.0 / p_market - 1.0) if win else -FLAT_BET
    else:
        p_no = 1.0 - p_market
        return FLAT_BET * (1.0 / p_no - 1.0) if win else -FLAT_BET


def prep(path: Path, side_col: str = "side") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df["p_market"]     = pd.to_numeric(df["p_market"],     errors="coerce")
    df = df[df["resolved_yes"].notna() & df["p_market"].notna()].copy()
    df["side"]     = df[side_col].str.strip().str.lower() if side_col in df.columns else "yes"
    df["win"]      = ((df["side"] == "yes") & (df["resolved_yes"] == 1)) | \
                     ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    df["flat_pnl"] = df.apply(
        lambda r: flat_pnl(r["side"], r["p_market"], r["win"])
        if pd.notna(r["p_market"]) and r["p_market"] not in (0, 1) else 0.0,
        axis=1,
    )
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    return df


def stats(df: pd.DataFrame, label: str = ""):
    n   = len(df)
    w   = int(df["win"].sum())
    l   = n - w
    wr  = w / n * 100 if n else 0
    pnl = df["flat_pnl"].sum()
    return n, w, l, wr, pnl


def print_stats(label, n, w, l, wr, pnl, extra=""):
    sign = "+" if pnl >= 0 else ""
    print(f"  {label:<34} n={n:>5}  W={w:>4}  L={l:>4}  WR={wr:>5.1f}%  PnL={sign}${pnl:>8.2f}{extra}")


def gate_report(df_all: pd.DataFrame, mask: pd.Series, label: str,
                block_direction: str = "both", side_filter: str = None,
                rescue_features: list = None):
    """
    mask: rows that would be BLOCKED by the gate.
    block_direction: 'yes' | 'no' | 'both' — which side the gate applies to.
    side_filter: only block this side.
    """
    if side_filter:
        mask = mask & (df_all["side"] == side_filter)

    blocked  = df_all[mask]
    kept     = df_all[~mask]
    base     = stats(df_all)
    blk      = stats(blocked)
    kpt      = stats(kept)
    delta    = kpt[4] - base[4]
    sign     = "+" if delta >= 0 else ""

    print(f"\n  ── {label} ──")
    print_stats("Baseline (all)", *base)
    print_stats("Blocked by gate", *blk)
    print_stats("Kept after gate", *kpt, extra=f"  Δ={sign}${delta:.2f}")

    # Rescue analysis: within blocked trades, find conditions where they WIN
    if rescue_features and len(blocked) >= 10:
        wins_in_blocked   = blocked[blocked["win"]]
        losses_in_blocked = blocked[~blocked["win"]]
        if len(wins_in_blocked) > 3:
            print(f"\n    Rescue candidates (blocked W={blk[1]}, L={blk[2]}):")
            for feat in rescue_features:
                if feat not in df_all.columns:
                    continue
                col = pd.to_numeric(blocked[feat], errors="coerce")
                if col.notna().sum() < 5:
                    continue
                win_mean  = pd.to_numeric(wins_in_blocked[feat],   errors="coerce").mean()
                loss_mean = pd.to_numeric(losses_in_blocked[feat], errors="coerce").mean()
                if pd.notna(win_mean) and pd.notna(loss_mean) and abs(win_mean - loss_mean) > 0.05:
                    print(f"      {feat:<28} win_mean={win_mean:+.3f}  loss_mean={loss_mean:+.3f}")

    return delta


def threshold_sweep(df_all: pd.DataFrame, feature: str, side_filter: str,
                    thresholds, block_when: str = "below",
                    label: str = ""):
    """Sweep thresholds for a feature and print best ones."""
    if feature not in df_all.columns:
        print(f"  [skip] {feature} not in data")
        return

    col = pd.to_numeric(df_all[feature], errors="coerce")
    sub = df_all[col.notna()].copy()
    col_sub = col[col.notna()]
    if len(sub) < 20:
        print(f"  [skip] {feature}: only {len(sub)} non-null rows")
        return

    print(f"\n  Sweep: {label or feature}  (side={side_filter or 'all'}, "
          f"block_when={block_when}, n={len(sub)})")
    print(f"  {'Threshold':>12}  {'Blk':>5}  {'BlkW':>5}  {'BlkL':>5}  "
          f"{'BlkWR':>6}  {'BlkPnL':>9}  {'KeptPnL':>9}  {'Delta':>9}")
    print("  " + "-" * 72)

    results = []
    for thr in thresholds:
        if block_when == "below":
            mask = col_sub < thr
        elif block_when == "above":
            mask = col_sub > thr
        elif block_when == "eq":
            mask = col_sub == thr
        else:
            continue

        if side_filter:
            mask = mask & (sub["side"] == side_filter)

        blk = sub[mask]
        kpt = sub[~mask]
        if len(blk) == 0:
            continue

        blk_n, blk_w, blk_l, blk_wr, blk_pnl = stats(blk)
        kpt_n, kpt_w, kpt_l, kpt_wr, kpt_pnl = stats(kpt)
        base_pnl = sub["flat_pnl"].sum()
        delta = kpt_pnl - base_pnl
        results.append((thr, blk_n, blk_w, blk_l, blk_wr, blk_pnl, kpt_pnl, delta))
        sign_d = "+" if delta >= 0 else ""
        sign_b = "+" if blk_pnl >= 0 else ""
        sign_k = "+" if kpt_pnl >= 0 else ""
        print(f"  {thr:>12}  {blk_n:>5}  {blk_w:>5}  {blk_l:>5}  "
              f"{blk_wr:>5.1f}%  {sign_b}${blk_pnl:>7.2f}  "
              f"{sign_k}${kpt_pnl:>7.2f}  {sign_d}${delta:>7.2f}")

    if results:
        best = max(results, key=lambda x: x[7])
        print(f"\n  Best threshold = {best[0]}  (Δ={'+' if best[7]>=0 else ''}${best[7]:.2f})")


# ── Load data ────────────────────────────────────────────────────────────────

def load_1h(asset: str) -> pd.DataFrame:
    p = {"BTC": "paper_trades.csv", "ETH": "paper_trades_eth.csv",
         "SOL": "paper_trades_sol.csv"}[asset]
    return prep(RESULTS / p)


def load_15m(asset: str) -> pd.DataFrame:
    p = {"BTC": "paper_trades_btc15m.csv", "ETH": "paper_trades_eth15m.csv",
         "SOL": "paper_trades_sol15m.csv"}[asset]
    df = prep(RESULTS / p)
    # For 15m, we only have side for trade rows. For pass rows, infer from model.
    # Only analyse rows where we have a clear side decision.
    return df[df["side"].isin(["yes", "no"])].copy()


# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("  GATE SIMULATION — Price Movement Correlations")
print(f"  Flat bet = ${FLAT_BET:.0f}  |  All dollar figures in flat-bet terms")
print(SEP)

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  1h BTC — vol_eff Kelly multiplier analysis")
print("  (vol_eff = vol efficiency score; higher = more decisive outcomes)")
print("─" * 72)

btc = load_1h("BTC")
btc["vol_eff"] = pd.to_numeric(btc["vol_eff"], errors="coerce")
btc_v = btc[btc["vol_eff"].notna()].copy()

q25, q50, q75 = btc_v["vol_eff"].quantile([0.25, 0.50, 0.75]).values
print(f"\n  vol_eff quartiles: Q1={q25:.6f}  Q2={q50:.6f}  Q3={q75:.6f}")
print(f"\n  {'Quartile':<20} {'n':>5}  {'W':>4}  {'L':>4}  {'WR':>6}  {'PnL':>10}  {'PnL/trade':>10}")
print("  " + "-" * 65)
for qlabel, qmask in [
    ("Q1 (lowest vol_eff)", btc_v["vol_eff"] <= q25),
    ("Q2",                  (btc_v["vol_eff"] > q25) & (btc_v["vol_eff"] <= q50)),
    ("Q3",                  (btc_v["vol_eff"] > q50) & (btc_v["vol_eff"] <= q75)),
    ("Q4 (highest vol_eff)",btc_v["vol_eff"] > q75),
]:
    sub = btc_v[qmask]
    n, w, l, wr, pnl = stats(sub)
    sign = "+" if pnl >= 0 else ""
    print(f"  {qlabel:<20} {n:>5}  {w:>4}  {l:>4}  {wr:>5.1f}%  "
          f"{sign}${pnl:>8.2f}  {'+' if pnl/n>=0 else ''}${pnl/n:>8.3f}")

# vol_eff block low: block trades where vol_eff is very low (indecisive outcomes)
threshold_sweep(btc_v, "vol_eff", None,
    thresholds=[q25 * 0.5, q25, q25 * 1.5, q50, q50 * 1.2],
    block_when="below",
    label="1h BTC — block when vol_eff LOW (indecisive outcome expected)")

threshold_sweep(btc_v, "vol_eff", "yes",
    thresholds=[q25, q50],
    block_when="below",
    label="1h BTC YES only — block when vol_eff LOW")

threshold_sweep(btc_v, "vol_eff", "no",
    thresholds=[q25, q50],
    block_when="below",
    label="1h BTC NO only — block when vol_eff LOW")

# ════════════════════════════════════════════════════════════════════════════
for asset in ("ETH", "SOL"):
    print(f"\n{'─'*72}")
    print(f"  1h {asset} — vol_eff + adx_1h + squeeze_1h simulations")
    print("─" * 72)

    df1h = load_1h(asset)
    df1h["vol_eff"]    = pd.to_numeric(df1h["vol_eff"], errors="coerce")
    df1h["adx_1h"]     = pd.to_numeric(df1h["adx_1h"], errors="coerce")
    df1h["squeeze_1h"] = pd.to_numeric(df1h["squeeze_1h"], errors="coerce")

    v = df1h[df1h["vol_eff"].notna()].copy()
    q25v, q50v = v["vol_eff"].quantile([0.25, 0.50]).values
    threshold_sweep(v, "vol_eff", None, [q25v, q50v], "below",
                    label=f"1h {asset} — block low vol_eff")
    threshold_sweep(v, "vol_eff", "yes", [q25v, q50v], "below",
                    label=f"1h {asset} YES — block low vol_eff")

    adx = df1h[df1h["adx_1h"].notna()].copy()
    if len(adx) >= 30:
        threshold_sweep(adx, "adx_1h", "yes",
            thresholds=[20, 25, 30, 35, 40],
            block_when="above",
            label=f"1h {asset} YES — block when adx_1h HIGH (strong trend = overbought)")
        threshold_sweep(adx, "adx_1h", "yes",
            thresholds=[20, 25, 30, 35],
            block_when="below",
            label=f"1h {asset} YES — block when adx_1h LOW (weak trend)")
        threshold_sweep(adx, "adx_1h", "no",
            thresholds=[20, 25, 30, 35, 40],
            block_when="above",
            label=f"1h {asset} NO — block when adx_1h HIGH")

    sq = df1h[df1h["squeeze_1h"].notna()].copy()
    if len(sq) >= 30:
        # squeeze_1h = 1 when active, 0 otherwise
        gate_report(sq, sq["squeeze_1h"] == 1, f"1h {asset} — block YES when squeeze active",
                    side_filter="yes",
                    rescue_features=["composite_p_up", "stoch_k", "ema_stack_bias",
                                     "composite_rev", "composite_trend", "vwap_distance_pct"])
        gate_report(sq, sq["squeeze_1h"] == 0, f"1h {asset} — block YES when squeeze NOT active",
                    side_filter="yes",
                    rescue_features=["composite_p_up", "stoch_k", "ema_stack_bias"])
        gate_report(sq, sq["squeeze_1h"] == 1, f"1h {asset} — block NO when squeeze active",
                    side_filter="no",
                    rescue_features=["composite_rev", "composite_trend", "stoch_k"])

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  15m GATES  (small n — interpret carefully)")
print("─" * 72)

for asset in ("BTC", "ETH", "SOL"):
    df15 = load_15m(asset)

    for feat, thresholds_above, thresholds_below, sides, note in [
        ("consec_dir_15m",
         [1, 2, 3],  # block YES when many consecutive bullish bars (mean reversion)
         [-1, -2, -3],  # block NO when many consecutive bearish bars
         ["yes", "no"],
         "block YES on bullish streak / NO on bearish streak"),
        ("upper_wick_15m",
         [0.10, 0.20, 0.30, 0.40],  # block YES when large upper wick
         [],
         ["yes"],
         "block YES when large upper wick (price reversal signal)"),
        ("ls_long_pct",
         [],
         [],
         ["yes"],
         "YES only by ls_long_pct level"),
    ]:
        col_data = pd.to_numeric(df15[feat], errors="coerce") if feat in df15.columns else pd.Series()
        sub = df15[col_data.notna()].copy() if len(col_data.dropna()) >= 10 else pd.DataFrame()
        if sub.empty:
            print(f"\n  [skip] 15m {asset} {feat}: insufficient data")
            continue

        n_trades = len(sub[sub.get("decision","") == "trade"]) if "decision" in sub.columns else 0
        print(f"\n{'─'*56}")
        print(f"  15m {asset} — {feat}  (n_with_feature={len(sub)}, "
              f"n_actual_trades={n_trades}, NOTE: small sample)")

        if "yes" in sides and thresholds_above:
            threshold_sweep(sub, feat, "yes", thresholds_above, "above",
                            label=f"15m {asset} YES — block when {feat} HIGH")
        if "yes" in sides and thresholds_below:
            threshold_sweep(sub, feat, "yes", thresholds_below, "below",
                            label=f"15m {asset} YES — block when {feat} LOW (bearish streak)")
        if "no" in sides and thresholds_above:
            threshold_sweep(sub, feat, "no", thresholds_below, "below",
                            label=f"15m {asset} NO — block when {feat} LOW")

        # For ls_long_pct: analyze by quartile
        if feat == "ls_long_pct":
            col = pd.to_numeric(sub[feat], errors="coerce")
            q50 = col.quantile(0.50)
            print(f"\n  15m {asset} ls_long_pct median={q50:.2f}")
            for side in ("yes", "no"):
                s = sub[sub["side"] == side]
                if len(s) < 5:
                    continue
                lo = s[pd.to_numeric(s[feat], errors="coerce") < q50]
                hi = s[pd.to_numeric(s[feat], errors="coerce") >= q50]
                if len(lo) >= 3 and len(hi) >= 3:
                    n1,w1,l1,wr1,p1 = stats(lo)
                    n2,w2,l2,wr2,p2 = stats(hi)
                    print(f"  {side.upper()} ls<{q50:.1f}: n={n1} WR={wr1:.1f}% PnL={'+' if p1>=0 else ''}${p1:.2f}  |  "
                          f"ls≥{q50:.1f}: n={n2} WR={wr2:.1f}% PnL={'+' if p2>=0 else ''}${p2:.2f}")

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  COMBINED 1h — vol_eff block below Q1 across all assets")
print(SEP)

frames = []
for asset in ("BTC", "ETH", "SOL"):
    d = load_1h(asset)
    d["asset"] = asset
    frames.append(d)
combined = pd.concat(frames, ignore_index=True)
combined["vol_eff"] = pd.to_numeric(combined["vol_eff"], errors="coerce")
cv = combined[combined["vol_eff"].notna()].copy()
gq25, gq50 = cv["vol_eff"].quantile([0.25, 0.50]).values
print(f"\n  Global Q1={gq25:.6f}  Q2={gq50:.6f}  n={len(cv)}")
threshold_sweep(cv, "vol_eff", None, [gq25, gq50], "below",
                label="Combined 1h — block ALL trades with low vol_eff")
threshold_sweep(cv, "vol_eff", "yes", [gq25, gq50], "below",
                label="Combined 1h YES — block low vol_eff")
threshold_sweep(cv, "vol_eff", "no", [gq25, gq50], "below",
                label="Combined 1h NO — block low vol_eff")
