#!/usr/bin/env python3
"""
simulate_liq_wall.py  (v3 — vectorized)

Proximity-to-key-level signals, computed via full-series pre-computation
(no per-row resample loops).  Runs all 6 asset×TF combos in seconds.

Signals at each trade timestamp:
  above_ema{20,50,100,200}_1h  : spot above each 1h EMA
  ema_stack_liq                : # EMAs below spot (0-4)
  above_vwap                   : spot above daily VWAP
  above_pivot_pp               : spot above prev-day pivot PP
  nearest_res_dist_pct         : % to nearest level above spot
  nearest_sup_dist_pct         : % to nearest level below spot
  n_res_within_X               : count of resistance levels within X% above
  on_key_level                 : within 0.3% of any key level

Run: python3 simulate_liq_wall.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
CSVS = {
    "BTC_15m": ("results/paper_trades_btc15m.csv", "BTCUSDT", "floor_strike", "15m"),
    "ETH_15m": ("results/paper_trades_eth15m.csv", "ETHUSDT", "floor_strike", "15m"),
    "SOL_15m": ("results/paper_trades_sol15m.csv", "SOLUSDT", "floor_strike", "15m"),
    "BTC_1h":  ("results/paper_trades.csv",         "BTCUSDT", "strike",       "1h"),
    "ETH_1h":  ("results/paper_trades_eth.csv",      "ETHUSDT", "strike",       "1h"),
    "SOL_1h":  ("results/paper_trades_sol.csv",      "SOLUSDT", "strike",       "1h"),
}
DATA_DIR    = Path("data")
EMA_PERIODS = [20, 50, 100, 200]
CLOSE_PCT   = 0.003   # "on key level" threshold

SEP  = "=" * 78
SEP2 = "-" * 78

# ── Parquet loader ─────────────────────────────────────────────────────────────
_cache: dict[str, pd.DataFrame] = {}

def load_1m(sym: str) -> pd.DataFrame:
    if sym not in _cache:
        files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
        df = pd.read_parquet(files[-1])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        _cache[sym] = df.sort_index()
    return _cache[sym]


# ── Vectorized indicator pre-computation for one symbol ────────────────────────

def build_indicator_series(sym: str) -> pd.DataFrame:
    """
    Pre-compute EMA / VWAP / Pivot series aligned to 1-minute bars.
    Returns a DataFrame indexed by UTC timestamp with columns:
      ema20, ema50, ema100, ema200  (from 1h close, forward-filled to 1m)
      vwap                          (daily VWAP, forward-filled)
      pivot_pp, pivot_r1, pivot_r2, pivot_s1, pivot_s2  (prev-day, forward-filled)
    """
    df = load_1m(sym)

    # 1h OHLCV
    df_1h = df.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])

    # EMAs on 1h close
    ema_1h = pd.DataFrame(index=df_1h.index)
    for p in EMA_PERIODS:
        ema_1h[f"ema{p}"] = df_1h["close"].ewm(span=p, adjust=False).mean()

    # Forward-fill to 1m (each 1m bar gets the EMA from the current/last completed hour)
    ema_1m = ema_1h.reindex(df.index, method="ffill")

    # Daily VWAP (reset each calendar day)
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_date"] = df.index.normalize()
    df["_cum_tpv"] = df.groupby("_date")["_tp"].transform(lambda s: (s * df.loc[s.index, "volume"]).cumsum())
    df["_cum_vol"] = df.groupby("_date")["volume"].transform("cumsum")
    df["vwap"] = df["_cum_tpv"] / df["_cum_vol"]

    # Daily pivots: computed from previous day's H/L/C, available from day-open
    df_1d = df.resample("1D").agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    df_1d["pp"] = (df_1d["high"] + df_1d["low"] + df_1d["close"]) / 3
    df_1d["r1"] = 2 * df_1d["pp"] - df_1d["low"]
    df_1d["r2"] = df_1d["pp"] + (df_1d["high"] - df_1d["low"])
    df_1d["s1"] = 2 * df_1d["pp"] - df_1d["high"]
    df_1d["s2"] = df_1d["pp"] - (df_1d["high"] - df_1d["low"])
    # Shift by 1 day so we use *previous* day's pivots
    pivots_1d = df_1d[["pp", "r1", "r2", "s1", "s2"]].shift(1)
    # Forward-fill to 1m
    pivots_1m = pivots_1d.reindex(df.index, method="ffill")

    result = pd.concat([ema_1m, df[["vwap"]], pivots_1m], axis=1)
    return result


# ── Timestamp parser (handles mixed tz-naive / tz-aware CSVs) ─────────────────

def parse_ts(s):
    try:
        t = pd.Timestamp(s)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except Exception:
        return pd.NaT


# ── Enrich trades DataFrame with signals ──────────────────────────────────────

def enrich(trades: pd.DataFrame, sym: str, ind: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade row, merge the nearest indicator snapshot at/before trade time.
    """
    ts = trades["ts"]
    spot = trades["spot"].values

    # merge_asof needs sorted keys
    left = trades[["ts", "spot"]].copy().reset_index(drop=True)
    left = left.sort_values("ts")
    right = ind.copy()
    right.index.name = "ts"
    right = right.reset_index()

    merged = pd.merge_asof(left, right, on="ts", direction="backward")

    # Restore original order
    merged = merged.set_index(left.index).reindex(trades.index)
    spot = trades["spot"].values

    sigs = pd.DataFrame(index=trades.index)

    for p in EMA_PERIODS:
        col = f"ema{p}"
        if col in merged.columns:
            ema_vals = merged[col].values
            sigs[f"above_ema{p}_1h"] = np.where(spot > ema_vals, 1, -1).astype(float)
            sigs[f"ema{p}_dist_pct"]  = (spot - ema_vals) / ema_vals * 100
        else:
            sigs[f"above_ema{p}_1h"] = np.nan
            sigs[f"ema{p}_dist_pct"]  = np.nan

    # EMA stack: count of EMAs below spot
    ema_cols = [f"ema{p}" for p in EMA_PERIODS if f"ema{p}" in merged.columns]
    if ema_cols:
        ema_mat = merged[ema_cols].values
        sigs["ema_stack_liq"] = (ema_mat < spot[:, None]).sum(axis=1).astype(float)
    else:
        sigs["ema_stack_liq"] = np.nan

    # VWAP
    if "vwap" in merged.columns:
        vwap = merged["vwap"].values
        sigs["above_vwap"]    = np.where(spot > vwap, 1, -1).astype(float)
        sigs["vwap_dist_pct"] = (spot - vwap) / vwap * 100
    else:
        sigs["above_vwap"]    = np.nan
        sigs["vwap_dist_pct"] = np.nan

    # Pivot PP
    if "pp" in merged.columns:
        pp = merged["pp"].values
        r1 = merged["r1"].values
        r2 = merged["r2"].values
        s1 = merged["s1"].values
        s2 = merged["s2"].values

        sigs["above_pivot_pp"]   = np.where(spot > pp, 1, -1).astype(float)
        sigs["pivot_pp_dist_pct"] = (spot - pp) / pp * 100

        # Nearest pivot resistance / support
        piv_res = np.where(r1 > spot, r1, np.where(r2 > spot, r2, np.where(pp > spot, pp, np.inf)))
        piv_sup = np.where(s1 < spot, s1, np.where(s2 < spot, s2, np.where(pp < spot, pp, -np.inf)))
        sigs["nearest_pivot_res_pct"] = np.where(
            np.isfinite(piv_res), (piv_res - spot) / spot * 100, 999.0
        )
        sigs["nearest_pivot_sup_pct"] = np.where(
            np.isfinite(piv_sup), (spot - piv_sup) / spot * 100, 999.0
        )
    else:
        for c in ["above_pivot_pp", "pivot_pp_dist_pct", "nearest_pivot_res_pct", "nearest_pivot_sup_pct"]:
            sigs[c] = np.nan

    # Nearest level across ALL sources (EMA + VWAP + pivots)
    all_level_cols = ema_cols + (["vwap"] if "vwap" in merged.columns else []) + \
                     (["pp", "r1", "r2", "s1", "s2"] if "pp" in merged.columns else [])
    if all_level_cols:
        lvl_mat = merged[all_level_cols].values   # shape (N, K)
        above   = lvl_mat - spot[:, None]          # positive = above spot
        below   = spot[:, None] - lvl_mat          # positive = below spot

        above_pos = np.where(above > 0, above, np.inf)
        below_pos = np.where(below > 0, below, np.inf)

        nearest_res = np.nanmin(above_pos, axis=1)
        nearest_sup = np.nanmin(below_pos, axis=1)

        sigs["nearest_res_dist_pct"] = np.where(
            np.isfinite(nearest_res), nearest_res / spot * 100, 999.0
        )
        sigs["nearest_sup_dist_pct"] = np.where(
            np.isfinite(nearest_sup), nearest_sup / spot * 100, 999.0
        )

        for thresh in [0.5, 1.0, 2.0]:
            mask_res = (above > 0) & (above / spot[:, None] * 100 <= thresh)
            mask_sup = (below > 0) & (below / spot[:, None] * 100 <= thresh)
            sigs[f"n_res_within_{thresh}pct"] = mask_res.sum(axis=1).astype(float)
            sigs[f"n_sup_within_{thresh}pct"] = mask_sup.sum(axis=1).astype(float)

        sigs["on_key_level"] = (
            (np.abs(lvl_mat - spot[:, None]) / spot[:, None] <= CLOSE_PCT).any(axis=1)
        ).astype(float)
    else:
        for c in ["nearest_res_dist_pct", "nearest_sup_dist_pct",
                  "n_res_within_0.5pct", "n_res_within_1.0pct", "n_res_within_2.0pct",
                  "n_sup_within_0.5pct", "n_sup_within_1.0pct", "n_sup_within_2.0pct",
                  "on_key_level"]:
            sigs[c] = np.nan

    return pd.concat([trades.reset_index(drop=True), sigs.reset_index(drop=True)], axis=1)


# ── Load + enrich all CSVs ─────────────────────────────────────────────────────

def load_and_enrich() -> pd.DataFrame:
    print("  Pre-computing indicators …", end="", flush=True)
    indicators: dict[str, pd.DataFrame] = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        indicators[sym] = build_indicator_series(sym)
        print(f" {sym[:3]}✓", end="", flush=True)
    print()

    enriched = []
    for key, (csv_path, sym, strike_col, tf) in CSVS.items():
        asset = key.split("_")[0]
        try:
            d = pd.read_csv(csv_path, low_memory=False)
        except FileNotFoundError:
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        d["asset"] = asset
        d["_sym"]  = sym
        d["_tf"]   = tf
        if strike_col in d.columns and strike_col != "strike":
            d = d.rename(columns={strike_col: "strike"})

        if "decision" in d.columns:
            mask = (d["decision"] == "trade") | d["decision"].isna()
        else:
            mask = pd.Series(True, index=d.index)

        d = d[mask & d["would_pnl"].notna()].copy()
        for col in ["would_win", "would_pnl", "spot", "strike", "p_market"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        if "bet_amount" not in d.columns:
            d["bet_amount"] = 1.0
        else:
            d["bet_amount"] = pd.to_numeric(d["bet_amount"], errors="coerce")

        d["ts"] = d["logged_at"].map(parse_ts)
        d = d.dropna(subset=["ts", "spot"]).reset_index(drop=True)

        print(f"  Enriching {key:<10} ({len(d):>4} trades) …", end=" ", flush=True)
        d = enrich(d, sym, indicators[sym])
        print("done")
        enriched.append(d)

    return pd.concat(enriched, ignore_index=True)


# ── Analysis helpers ───────────────────────────────────────────────────────────

def report_gate(df, blocked, label, indent="  "):
    kept   = df[~blocked]
    blk    = df[blocked]
    n_b    = int(blocked.sum())
    wr_b   = blk["would_win"].mean()  if n_b          else float("nan")
    wr_k   = kept["would_win"].mean() if len(kept)     else float("nan")
    pnl_b  = df["would_pnl"].sum()
    pnl_k  = kept["would_pnl"].sum() if len(kept)     else 0.0
    delta  = pnl_k - pnl_b
    wb     = int(blk["would_win"].sum()) if n_b else 0
    lb     = n_b - wb
    print(f"{indent}{label:<36}  "
          f"blk={n_b:>3}(W{wb}/L{lb})  "
          f"WR_blk={wr_b:>5.1%}  "
          f"WR_kept={wr_k:>5.1%}  "
          f"PnL_kept=${pnl_k:>+8.2f}  "
          f"delta=${delta:>+8.2f}")


def section_correlations(df):
    signals = [
        "above_ema20_1h","above_ema50_1h","above_ema100_1h","above_ema200_1h",
        "ema_stack_liq","above_vwap","above_pivot_pp",
        "nearest_res_dist_pct","nearest_sup_dist_pct",
        "nearest_pivot_res_pct","nearest_pivot_sup_pct",
        "vwap_dist_pct","pivot_pp_dist_pct",
        "ema200_dist_pct","ema100_dist_pct","ema50_dist_pct","ema20_dist_pct",
        "n_res_within_0.5pct","n_res_within_1.0pct","n_res_within_2.0pct",
        "n_sup_within_0.5pct","n_sup_within_1.0pct","n_sup_within_2.0pct",
        "on_key_level",
    ]
    yes = df[df["side"] == "yes"]
    no  = df[df["side"] == "no"]
    print(f"\n{SEP}")
    print(f"  SIGNAL CORRELATIONS with would_win")
    print(SEP)
    print(f"\n  {'Signal':<30}  {'r_YES':>7}  {'N_yes':>6}  {'r_NO':>7}  {'N_no':>6}  {'r_ALL':>7}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}")
    rows = []
    for sig in signals:
        if sig not in df.columns:
            continue
        r_yes = yes[sig].corr(yes["would_win"]) if len(yes) > 5 else float("nan")
        r_no  = no[sig].corr(no["would_win"])   if len(no) > 5  else float("nan")
        r_all = df[sig].corr(df["would_win"])
        rows.append((abs(r_yes or 0) + abs(r_no or 0), sig, r_yes,
                     len(yes.dropna(subset=[sig])), r_no,
                     len(no.dropna(subset=[sig])), r_all))
    rows.sort(reverse=True)
    for _, sig, r_yes, n_yes, r_no, n_no, r_all in rows:
        flag = " ◄" if (abs(r_yes or 0) > 0.08 or abs(r_no or 0) > 0.08) else ""
        print(f"  {sig:<30}  {r_yes:>+7.3f}  {n_yes:>6}  {r_no:>+7.3f}  {n_no:>6}  {r_all:>+7.3f}{flag}")


def section_buckets(df):
    print(f"\n{SEP}")
    print(f"  OUTCOME BY SIGNAL VALUE — key binary signals")
    print(SEP)

    binary_sigs = [
        ("above_ema200_1h", "Spot above EMA200 1h"),
        ("above_ema100_1h", "Spot above EMA100 1h"),
        ("above_ema50_1h",  "Spot above EMA50 1h"),
        ("above_vwap",      "Spot above daily VWAP"),
        ("above_pivot_pp",  "Spot above pivot PP"),
        ("ema_stack_liq",   "EMA stack (0-4 EMAs below)"),
    ]

    for side in ["yes", "no"]:
        sub = df[df["side"] == side]
        print(f"\n  Side={side.upper()}  N={len(sub)}  WR={sub['would_win'].mean():.1%}  "
              f"PnL=${sub['would_pnl'].sum():+.2f}")
        print(f"  {'Signal':<30}  {'Val':>3}  {'N':>5}  {'WR':>7}  {'PnL':>9}  {'vs base':>8}")
        print(f"  {SEP2}")
        base_wr = sub["would_win"].mean()
        for col, label in binary_sigs:
            if col not in sub.columns:
                continue
            for val in sorted(sub[col].dropna().unique()):
                grp = sub[sub[col] == val]
                if len(grp) < 5:
                    continue
                wr   = grp["would_win"].mean()
                pnl  = grp["would_pnl"].sum()
                diff = wr - base_wr
                flag = " ◄◄" if abs(diff) > 0.10 else (" ◄" if abs(diff) > 0.05 else "")
                print(f"  {label:<30}  {int(val):>3}  {len(grp):>5}  {wr:>6.1%}  "
                      f"${pnl:>+8.2f}  {diff:>+7.1%}{flag}")

    # Nearest resistance bucket for YES
    print(f"\n  NEAREST RESISTANCE (all levels) — YES trades")
    yes = df[df["side"] == "yes"].copy()
    if "nearest_res_dist_pct" in yes.columns:
        yes["res_bucket"] = pd.cut(
            yes["nearest_res_dist_pct"].clip(0, 5),
            bins=[0, 0.2, 0.5, 1.0, 2.0, 5.0],
            labels=["<0.2%", "0.2-0.5%", "0.5-1%", "1-2%", ">2%"],
        )
        grp = yes.groupby("res_bucket", observed=True).agg(
            N=("would_win", "count"), WR=("would_win", "mean"), PnL=("would_pnl", "sum")
        )
        base_wr = yes["would_win"].mean()
        for bkt, row in grp.iterrows():
            diff = row["WR"] - base_wr
            flag = " ◄◄" if abs(diff) > 0.10 else (" ◄" if abs(diff) > 0.05 else "")
            print(f"  {str(bkt):>10}  N={int(row['N']):>4}  WR={row['WR']:.1%}  "
                  f"PnL=${row['PnL']:>+8.2f}  {diff:>+7.1%}{flag}")

    # Nearest support bucket for both sides
    print(f"\n  NEAREST SUPPORT (all levels) — by side")
    for side in ["yes", "no"]:
        sub = df[df["side"] == side].copy()
        if "nearest_sup_dist_pct" not in sub.columns:
            continue
        sub["sup_bucket"] = pd.cut(
            sub["nearest_sup_dist_pct"].clip(0, 5),
            bins=[0, 0.2, 0.5, 1.0, 2.0, 5.0],
            labels=["<0.2%", "0.2-0.5%", "0.5-1%", "1-2%", ">2%"],
        )
        grp = sub.groupby("sup_bucket", observed=True).agg(
            N=("would_win", "count"), WR=("would_win", "mean"), PnL=("would_pnl", "sum")
        )
        base_wr = sub["would_win"].mean()
        print(f"  {side.upper()} (base WR={base_wr:.1%}):")
        for bkt, row in grp.iterrows():
            diff = row["WR"] - base_wr
            flag = " ◄◄" if abs(diff) > 0.10 else (" ◄" if abs(diff) > 0.05 else "")
            print(f"    {str(bkt):>10}  N={int(row['N']):>4}  WR={row['WR']:.1%}  "
                  f"PnL=${row['PnL']:>+8.2f}  {diff:>+7.1%}{flag}")


def section_gates(df):
    print(f"\n{SEP}")
    print(f"  GATE SWEEPS — all assets combined")
    print(f"  Baseline: N={len(df)}  WR={df['would_win'].mean():.1%}  "
          f"PnL=${df['would_pnl'].sum():+.2f}")
    print(SEP)

    yes = df[df["side"] == "yes"]
    no  = df[df["side"] == "no"]

    # YES gates
    print(f"\n  YES gates  N={len(yes)}  WR={yes['would_win'].mean():.1%}  "
          f"PnL=${yes['would_pnl'].sum():+.2f}")
    print(f"  {'Gate':<36}  blk        WR_blk   WR_kept   PnL_kept      delta")
    print(f"  {SEP2}")
    report_gate(yes, pd.Series(False, index=yes.index), "BASELINE")
    for col, val, label in [
        ("above_ema200_1h", -1, "YES: below EMA200 1h"),
        ("above_ema100_1h", -1, "YES: below EMA100 1h"),
        ("above_ema50_1h",  -1, "YES: below EMA50 1h"),
        ("above_vwap",      -1, "YES: below daily VWAP"),
        ("above_pivot_pp",  -1, "YES: below pivot PP"),
        ("ema_stack_liq",    0, "YES: 0 EMAs below spot"),
        ("ema_stack_liq",    1, "YES: <=1 EMA below spot"),
    ]:
        if col not in yes.columns:
            continue
        blocked = yes[col] == val
        if blocked.sum() >= 5:
            report_gate(yes, blocked, label)

    print(f"\n  YES: nearest resistance close above")
    for thresh in [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if "nearest_res_dist_pct" not in yes.columns:
            break
        blocked = yes["nearest_res_dist_pct"] <= thresh
        if 5 <= blocked.sum() <= len(yes) - 5:
            report_gate(yes, blocked, f"YES: nearest_res <= {thresh:.2f}%")

    print(f"\n  YES: multiple resistance levels clustered above")
    for col in ["n_res_within_0.5pct", "n_res_within_1.0pct", "n_res_within_2.0pct"]:
        if col not in yes.columns:
            continue
        for n in [1, 2, 3, 4]:
            blocked = yes[col] >= n
            if 5 <= blocked.sum() <= len(yes) - 5:
                report_gate(yes, blocked, f"YES: {col} >= {n}")

    # NO gates
    print(f"\n  NO gates  N={len(no)}  WR={no['would_win'].mean():.1%}  "
          f"PnL=${no['would_pnl'].sum():+.2f}")
    print(f"  {'Gate':<36}  blk        WR_blk   WR_kept   PnL_kept      delta")
    print(f"  {SEP2}")
    report_gate(no, pd.Series(False, index=no.index), "BASELINE")
    for col, val, label in [
        ("above_ema200_1h",  1, "NO: above EMA200 1h"),
        ("above_ema100_1h",  1, "NO: above EMA100 1h"),
        ("above_ema50_1h",   1, "NO: above EMA50 1h"),
        ("above_vwap",       1, "NO: above daily VWAP"),
        ("above_pivot_pp",   1, "NO: above pivot PP"),
        ("ema_stack_liq",    4, "NO: all 4 EMAs below spot"),
        ("ema_stack_liq",    3, "NO: 3+ EMAs below spot"),
    ]:
        if col not in no.columns:
            continue
        blocked = no[col] == val
        if blocked.sum() >= 5:
            report_gate(no, blocked, label)

    print(f"\n  NO: support close below (price may bounce)")
    for thresh in [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if "nearest_sup_dist_pct" not in no.columns:
            break
        blocked = no["nearest_sup_dist_pct"] <= thresh
        if 5 <= blocked.sum() <= len(no) - 5:
            report_gate(no, blocked, f"NO: nearest_sup <= {thresh:.2f}%")

    # Best combined YES + NO gate combos
    print(f"\n  COMBINED GATE SWEEPS (YES + NO)")
    print(f"  {'Combo':<55}  blk        WR_blk   WR_kept   PnL_kept      delta")
    print(f"  {SEP2}")
    report_gate(df, pd.Series(False, index=df.index), "BASELINE (no gates)")

    def make_blocked(d, col, val):
        if col.endswith("_le"):
            real = col[:-3]
            return (d[real] <= val) if real in d.columns else pd.Series(False, index=d.index)
        return (d[col] == val) if col in d.columns else pd.Series(False, index=d.index)

    yes_gates = [
        ("above_ema200_1h", -1, "YES<EMA200"),
        ("above_vwap",      -1, "YES<VWAP"),
        ("above_pivot_pp",  -1, "YES<PP"),
        ("nearest_res_dist_pct_le", 0.5, "YES_res<=0.5%"),
        ("nearest_res_dist_pct_le", 1.0, "YES_res<=1.0%"),
    ]
    no_gates = [
        ("above_ema200_1h",  1, "NO>EMA200"),
        ("above_vwap",       1, "NO>VWAP"),
        ("above_pivot_pp",   1, "NO>PP"),
        ("nearest_sup_dist_pct_le", 0.5, "NO_sup<=0.5%"),
        ("nearest_sup_dist_pct_le", 1.0, "NO_sup<=1.0%"),
    ]
    combos = []
    for ycol, yval, ylabel in yes_gates:
        for ncol, nval, nlabel in no_gates:
            yblk = (df["side"] == "yes") & make_blocked(df, ycol, yval)
            nblk = (df["side"] == "no")  & make_blocked(df, ncol, nval)
            combined = yblk | nblk
            if 10 <= combined.sum() <= len(df) - 10:
                kept  = df[~combined]
                delta = kept["would_pnl"].sum() - df["would_pnl"].sum()
                combos.append((delta, combined, f"{ylabel} + {nlabel}"))
    combos.sort(reverse=True)
    for delta, combined, label in combos[:20]:
        report_gate(df, combined, label)


def section_by_asset(df):
    print(f"\n{SEP}")
    print(f"  PER-ASSET BREAKDOWN")
    print(SEP)

    top_yes = [
        ("above_ema200_1h", -1, "below EMA200"),
        ("above_vwap",      -1, "below VWAP"),
        ("above_pivot_pp",  -1, "below Pivot PP"),
        ("ema_stack_liq",    0, "0 EMAs below"),
    ]
    top_no = [
        ("above_ema200_1h",  1, "above EMA200"),
        ("above_vwap",       1, "above VWAP"),
        ("above_pivot_pp",   1, "above Pivot PP"),
        ("ema_stack_liq",    4, "all 4 EMAs below"),
    ]
    for asset in ["BTC", "ETH", "SOL"]:
        sub = df[df["asset"] == asset]
        print(f"\n  {asset}  N={len(sub)}  WR={sub['would_win'].mean():.1%}  "
              f"PnL=${sub['would_pnl'].sum():+.2f}")
        for side, gates in [("yes", top_yes), ("no", top_no)]:
            s = sub[sub["side"] == side]
            if s.empty:
                continue
            print(f"    {side.upper()} N={len(s)}  WR={s['would_win'].mean():.1%}")
            for col, val, label in gates:
                if col not in s.columns:
                    continue
                blocked = s[col] == val
                if blocked.sum() >= 3:
                    report_gate(s, blocked, f"      {label}", indent="")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Liquidity-Level Proximity Gate Simulation  (v3 — vectorized)")
    print("  Signals: EMA20/50/100/200 (1h) | VWAP (daily) | Prev-day Pivots")
    print("  All signals fire on EVERY trade via merge_asof")
    print(SEP)

    df = load_and_enrich()

    print(f"\n  Total resolved trades: {len(df)}")
    for tf in ["15m", "1h"]:
        sub = df[df["_tf"] == tf]
        if sub.empty:
            continue
        print(f"  [{tf}] N={len(sub)}  "
              f"BTC={(sub.asset=='BTC').sum()}  "
              f"ETH={(sub.asset=='ETH').sum()}  "
              f"SOL={(sub.asset=='SOL').sum()}  "
              f"WR={sub['would_win'].mean():.1%}  PnL=${sub['would_pnl'].sum():+.2f}")
    print(f"\n  COMBINED: WR={df['would_win'].mean():.1%}  PnL=${df['would_pnl'].sum():+.2f}")

    for tf_label, subset in [("15m", df[df["_tf"] == "15m"]),
                              ("1h",  df[df["_tf"] == "1h"]),
                              ("ALL", df)]:
        if subset.empty:
            continue
        print(f"\n{'='*78}")
        print(f"  ── TIMEFRAME: {tf_label}  N={len(subset)}  "
              f"WR={subset['would_win'].mean():.1%}  "
              f"PnL=${subset['would_pnl'].sum():+.2f}")
        print(f"{'='*78}")
        section_correlations(subset)
        section_buckets(subset)
        section_gates(subset)
        section_by_asset(subset)

    print(f"\n{SEP}\n  done\n{SEP}\n")


if __name__ == "__main__":
    main()
