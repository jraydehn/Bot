"""
BTC 15m gate backtest: 5 scenarios with HMM regime backfill.
Scenarios:
  0 — Baseline (no new gates)
  1 — Gate A only (block NO when stoch_k_15m>=80 AND ema_bias==1, no rescue)
  2 — Gate A with rescue (allow through when bp_15m<0.35 AND dir_15m==-1)
  3 — HMM regimes only (replace markov labels in regime gates)
  4 — Gate A rescue + HMM regimes
"""

import pandas as pd
import numpy as np
import pickle

DATA_DIR = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data"
RESULTS_DIR = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results"

KELLY_MULT = 0.30
KELLY_CAP = 0.06
BANKROLL = 1000
EDGE_THRESH = 0.04

# ── Step 1: Load HMM models ─────────────────────────────────────────────────
print("Loading HMM models...")
with open(f"{RESULTS_DIR}/hmm_3state_btc_1h.pkl", "rb") as f:
    hmm_1h_bundle = pickle.load(f)
with open(f"{RESULTS_DIR}/hmm_3state_btc_15m.pkl", "rb") as f:
    hmm_15m_bundle = pickle.load(f)

model_1h = hmm_1h_bundle["model"]
stn_1h   = hmm_1h_bundle["state_to_name"]   # int -> 'Bull'/'Bear'/'Sideways'
feats_1h = hmm_1h_bundle["feature_cols"]    # ['log_ret','realized_vol','ret_5bar']

model_15m = hmm_15m_bundle["model"]
stn_15m   = hmm_15m_bundle["state_to_name"]
feats_15m = hmm_15m_bundle["feature_cols"]

print(f"  1h  state map: {stn_1h}  features: {feats_1h}")
print(f"  15m state map: {stn_15m}  features: {feats_15m}")


def compute_hmm_features(df_close):
    """Given a series of close prices (indexed by timestamp), compute HMM features."""
    c = df_close.copy()
    log_ret      = np.log(c / c.shift(1))
    realized_vol = log_ret.rolling(20, min_periods=10).std()
    ret_5bar     = np.log(c / c.shift(5))
    feat = pd.DataFrame({
        "log_ret":      log_ret,
        "realized_vol": realized_vol,
        "ret_5bar":     ret_5bar,
    }, index=c.index)
    return feat.dropna()


# ── 1h HMM prediction ───────────────────────────────────────────────────────
print("\nBuilding 1h HMM regime series...")
df_1h = pd.read_parquet(f"{DATA_DIR}/binanceus_BTCUSDT_1h_1970-01-01_2026-05-28.parquet")
df_1h = df_1h[~df_1h.index.duplicated(keep="last")]
df_1h = df_1h.sort_index()
# Drop the dummy 1970 row if present
df_1h = df_1h[df_1h.index >= pd.Timestamp("2020-01-01", tz="UTC")]

feat_1h = compute_hmm_features(df_1h["close"])
states_1h = model_1h.predict(feat_1h[feats_1h].values)
hmm_regime_1h = pd.Series(
    [stn_1h[s] for s in states_1h],
    index=feat_1h.index,
    name="hmm_regime_1h"
)
print(f"  1h regime series: {len(hmm_regime_1h)} bars")
print(f"  1h distribution:\n{hmm_regime_1h.value_counts()}")


# ── 15m HMM prediction (resample 1m → 15m) ──────────────────────────────────
print("\nBuilding 15m HMM regime series (resample from 1m)...")
df_1m = pd.read_parquet(f"{DATA_DIR}/binanceus_BTCUSDT_1m_1970-01-01_2026-05-28.parquet")
df_1m = df_1m[~df_1m.index.duplicated(keep="last")]
df_1m = df_1m.sort_index()
df_1m = df_1m[df_1m.index >= pd.Timestamp("2020-01-01", tz="UTC")]

# Resample to 15m; use 'label=left, closed=left' so bar timestamp = open of bar
df_15m = df_1m["close"].resample("15min", label="left", closed="left").last().dropna()

feat_15m = compute_hmm_features(df_15m)
states_15m = model_15m.predict(feat_15m[feats_15m].values)
hmm_regime_15m = pd.Series(
    [stn_15m[s] for s in states_15m],
    index=feat_15m.index,
    name="hmm_regime_15m"
)
print(f"  15m regime series: {len(hmm_regime_15m)} bars")
print(f"  15m distribution:\n{hmm_regime_15m.value_counts()}")


# ── Step 2: Load scan archive and join HMM regimes ───────────────────────────
print("\nLoading scan archive...")
df = pd.read_csv(f"{RESULTS_DIR}/btc_scan_archive_15m.csv")
df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True)
print(f"  Raw rows: {len(df)}")

# Dedup: keep latest logged_at per contract_ticker
df = df.sort_values("logged_at").drop_duplicates(subset="contract_ticker", keep="last")
print(f"  After dedup: {len(df)}")

# Filter: resolved_yes not null
df = df[df["resolved_yes"].notna()]
print(f"  After resolved filter: {len(df)}")

# For merge_asof we need the archive sorted by logged_at and the regime series
# as DataFrames with 'timestamp' key.
df = df.sort_values("logged_at").reset_index(drop=True)

# Build left DataFrames for merge_asof
regime_1h_df  = hmm_regime_1h.reset_index()
regime_1h_df.columns = ["bar_ts", "hmm_regime_1h"]

regime_15m_df = hmm_regime_15m.reset_index()
regime_15m_df.columns = ["bar_ts", "hmm_regime_15m"]

# We want "most recent completed bar before logged_at"
# For 1h: bar at ts represents open_time, it's "complete" at ts + 1h
# So we want the bar whose open_time + 1h <= logged_at, i.e. open_time <= logged_at - 1h
# Equivalently: shift index right by 1 bar period so bar at T is available at T + period
regime_1h_df_shifted = regime_1h_df.copy()
regime_1h_df_shifted["bar_ts"] = regime_1h_df_shifted["bar_ts"] + pd.Timedelta(hours=1)

regime_15m_df_shifted = regime_15m_df.copy()
regime_15m_df_shifted["bar_ts"] = regime_15m_df_shifted["bar_ts"] + pd.Timedelta(minutes=15)

archive_ts = pd.DataFrame({"logged_at": df["logged_at"]})

merged_1h = pd.merge_asof(
    archive_ts.sort_values("logged_at"),
    regime_1h_df_shifted.sort_values("bar_ts").rename(columns={"bar_ts": "logged_at"}),
    on="logged_at",
    direction="backward"
)
merged_15m = pd.merge_asof(
    archive_ts.sort_values("logged_at"),
    regime_15m_df_shifted.sort_values("bar_ts").rename(columns={"bar_ts": "logged_at"}),
    on="logged_at",
    direction="backward"
)

df = df.sort_values("logged_at").reset_index(drop=True)
df["hmm_regime_1h"]  = merged_1h["hmm_regime_1h"].values
df["hmm_regime_15m"] = merged_15m["hmm_regime_15m"].values

print(f"\nHMM regime null counts after join:")
print(f"  hmm_regime_1h  nulls: {df['hmm_regime_1h'].isna().sum()}")
print(f"  hmm_regime_15m nulls: {df['hmm_regime_15m'].isna().sum()}")


# ── Step 3: Regime label comparison ─────────────────────────────────────────
print("\n" + "="*60)
print("REGIME LABEL COMPARISON (old rolling-return vs HMM)")
print("="*60)

# The archive doesn't have old rolling-return regime columns, so we can only compare
# HMM 1h vs HMM 15m distributions, and note that
# Note: scan archive has no markov columns logged — we work with HMM only.
# We'll also compare HMM 1h vs HMM 15m agreement.

print("\nHMM 1h distribution (archive rows):")
print(df["hmm_regime_1h"].value_counts(dropna=False))
print("\nHMM 15m distribution (archive rows):")
print(df["hmm_regime_15m"].value_counts(dropna=False))

# Cross-tab
print("\n1h vs 15m cross-tab:")
print(pd.crosstab(df["hmm_regime_1h"], df["hmm_regime_15m"], margins=True))

# Agreement between 1h and 15m
match = (df["hmm_regime_1h"] == df["hmm_regime_15m"]).sum()
total_valid = df[["hmm_regime_1h","hmm_regime_15m"]].dropna().shape[0]
print(f"\n1h==15m agreement: {match}/{total_valid} = {match/total_valid*100:.1f}%")


# ── Step 4: Build betting model ──────────────────────────────────────────────
def kelly_bet(edge, pm):
    """Fractional Kelly sizing, capped at KELLY_CAP * BANKROLL."""
    # p = prob of win (edge = p - cost)
    # For YES: win prob = pm_model_yes; q=1-p; b = (1-pm)/pm (odds)
    # Kelly = p - q/b = p - (1-p)*pm/(1-pm)
    # We use a simpler version: kelly = edge * KELLY_MULT
    kelly_f = edge * KELLY_MULT
    kelly_f = min(kelly_f, KELLY_CAP)
    return max(kelly_f * BANKROLL, 0)


def calc_pnl(row, side):
    """Calculate PnL for a given bet side."""
    pm = row["p_market"]
    fee = 0.07 * min(pm, 1 - pm)
    won = (row["resolved_yes"] == 1.0)

    if side == "YES":
        bet = kelly_bet(row["edge_yes"], pm)
        if won:
            return bet * (1 - pm - fee)
        else:
            return -bet * (pm + fee)
    else:  # NO
        bet = kelly_bet(row["edge_no"], pm)
        if won:
            return -bet * (1 - pm + fee)
        else:
            return bet * (pm - fee)


# Compute edges
df["edge_yes"] = df["p_model_yes"] - df["p_market"]
df["edge_no"]  = df["p_market"]    - df["p_model_no"]

# Determine base side
def get_side(row):
    ey = row["edge_yes"]
    en = row["edge_no"]
    if en > ey and en > EDGE_THRESH:
        return "NO"
    elif ey > en and ey > EDGE_THRESH:
        return "YES"
    return None

df["base_side"] = df.apply(get_side, axis=1)
df = df[df["base_side"].notna()].reset_index(drop=True)
print(f"\nRows with base edge: {len(df)}")
print(f"  Base YES: {(df['base_side']=='YES').sum()}")
print(f"  Base NO:  {(df['base_side']=='NO').sum()}")


# ── Step 5: Simulate each scenario ──────────────────────────────────────────
def simulate(df, label, gate_a=False, gate_a_rescue=False, use_hmm=False):
    """
    gate_a: block NO when stoch_k_15m>=80 AND ema_bias==1
    gate_a_rescue: allow through (rescue) when bp_15m<0.35 AND dir_15m==-1
    use_hmm: apply regime gates using HMM labels instead of old labels
              (since archive has no old labels, this is reflected in
               markov_1h_bear_gate and markov_15m_bear_gate logic)
    """
    results = []

    for _, row in df.iterrows():
        side = row["base_side"]
        blocked = False
        block_reason = None

        # --- Gate A: block NO when stoch_k_15m>=80 AND ema_bias==1 ---
        if gate_a and side == "NO":
            triggered = (row["stoch_k_15m"] >= 80) and (row["ema_bias"] == 1)
            if triggered:
                if gate_a_rescue:
                    # Rescue: allow through if bp_15m < 0.35 AND dir_15m == -1
                    rescued = (row["bp_15m"] < 0.35) and (row["dir_15m"] == -1)
                    if not rescued:
                        blocked = True
                        block_reason = "gate_a"
                else:
                    blocked = True
                    block_reason = "gate_a"

        # --- Regime gates (Scenario 3/4): use HMM labels ---
        # markov_1h_bear_gate: block YES when 1h regime == 'Bear'
        # markov_15m_bear_gate: block YES when 15m regime == 'Bear' AND composite_p_up > 0.488
        if use_hmm and side == "YES" and not blocked:
            regime_1h  = row["hmm_regime_1h"]
            regime_15m = row["hmm_regime_15m"]
            cpu = row.get("composite_p_up", np.nan)

            if regime_1h == "Bear":
                blocked = True
                block_reason = "hmm_1h_bear"
            elif regime_15m == "Bear" and not np.isnan(cpu) and cpu > 0.488:
                blocked = True
                block_reason = "hmm_15m_bear"

        if not blocked:
            pnl = calc_pnl(row, side)
            won = (row["resolved_yes"] == 1.0) if side == "YES" else (row["resolved_yes"] == 0.0)
            results.append({
                "side": side,
                "pnl": pnl,
                "won": won,
                "blocked": False,
                "block_reason": None,
            })
        else:
            # Blocked: record as blocked with the outcome for analysis
            won = (row["resolved_yes"] == 1.0) if side == "YES" else (row["resolved_yes"] == 0.0)
            pnl_counterfactual = calc_pnl(row, side)
            results.append({
                "side": side,
                "pnl": 0,
                "won": won,
                "blocked": True,
                "block_reason": block_reason,
                "pnl_counterfactual": pnl_counterfactual,
            })

    rdf = pd.DataFrame(results)
    traded = rdf[~rdf["blocked"]]
    blocked_df = rdf[rdf["blocked"]]

    n_trades = len(traded)
    n_yes = (traded["side"] == "YES").sum()
    n_no  = (traded["side"] == "NO").sum()
    wins  = traded["won"].sum()
    wr    = wins / n_trades if n_trades > 0 else 0
    total_pnl = traded["pnl"].sum()

    # Blocked analysis per reason
    block_summary = {}
    for reason in blocked_df["block_reason"].unique():
        sub = blocked_df[blocked_df["block_reason"] == reason]
        wins_blocked   = sub["won"].sum()
        losses_blocked = (~sub["won"]).sum()
        pnl_saved = -sub["pnl_counterfactual"].sum() if "pnl_counterfactual" in sub.columns else 0
        block_summary[reason] = {
            "n": len(sub),
            "wins_blocked": int(wins_blocked),
            "losses_blocked": int(losses_blocked),
            "pnl_saved": float(pnl_saved)
        }

    return {
        "label": label,
        "n_trades": n_trades,
        "n_yes": int(n_yes),
        "n_no":  int(n_no),
        "wr":    wr,
        "total_pnl": total_pnl,
        "block_summary": block_summary,
    }


print("\n" + "="*60)
print("SCENARIO SIMULATION RESULTS")
print("="*60)

scenarios = [
    dict(label="S0 Baseline",                gate_a=False, gate_a_rescue=False, use_hmm=False),
    dict(label="S1 Gate A only (no rescue)",  gate_a=True,  gate_a_rescue=False, use_hmm=False),
    dict(label="S2 Gate A + rescue",          gate_a=True,  gate_a_rescue=True,  use_hmm=False),
    dict(label="S3 HMM regimes only",         gate_a=False, gate_a_rescue=False, use_hmm=True),
    dict(label="S4 Gate A rescue + HMM",      gate_a=True,  gate_a_rescue=True,  use_hmm=True),
]

results_list = []
baseline_pnl = None

for sc in scenarios:
    res = simulate(df, **sc)
    results_list.append(res)
    if baseline_pnl is None:
        baseline_pnl = res["total_pnl"]

print(f"\n{'Scenario':<35} {'N':>5} {'YES':>5} {'NO':>5} {'WR%':>7} {'PnL':>8} {'Delta':>8}")
print("-" * 80)
for res in results_list:
    delta = res["total_pnl"] - baseline_pnl
    delta_str = f"+${delta:.2f}" if delta >= 0 else f"-${abs(delta):.2f}"
    print(f"{res['label']:<35} {res['n_trades']:>5} {res['n_yes']:>5} {res['n_no']:>5} "
          f"{res['wr']*100:>6.1f}% ${res['total_pnl']:>7.2f} {delta_str:>8}")

print()
print("="*60)
print("GATE IMPACT BREAKDOWN (wins blocked / losses blocked / PnL saved)")
print("="*60)
for res in results_list[1:]:
    if res["block_summary"]:
        print(f"\n{res['label']}:")
        for reason, bs in res["block_summary"].items():
            print(f"  Gate [{reason}]: n={bs['n']}, "
                  f"wins_blocked={bs['wins_blocked']}, "
                  f"losses_blocked={bs['losses_blocked']}, "
                  f"pnl_saved=${bs['pnl_saved']:.2f}")
    else:
        print(f"\n{res['label']}: no gates fired")

# ── Deep dive: Gate A trigger analysis ──────────────────────────────────────
print("\n" + "="*60)
print("GATE A TRIGGER DEEP DIVE")
print("="*60)

no_trades = df[df["base_side"] == "NO"].copy()
print(f"\nBase NO trades: {len(no_trades)}")

gate_a_trigger = no_trades[(no_trades["stoch_k_15m"] >= 80) & (no_trades["ema_bias"] == 1)]
print(f"Gate A triggered: {len(gate_a_trigger)}")

if len(gate_a_trigger) > 0:
    # WR of triggered trades (NO wins = resolved_yes == 0)
    triggered_wr = (gate_a_trigger["resolved_yes"] == 0).mean()
    all_no_wr    = (no_trades["resolved_yes"] == 0).mean()
    print(f"  WR (NO) in triggered:  {triggered_wr*100:.1f}%")
    print(f"  WR (NO) in all NO:     {all_no_wr*100:.1f}%")

    # Rescue subset
    rescued = gate_a_trigger[(gate_a_trigger["bp_15m"] < 0.35) & (gate_a_trigger["dir_15m"] == -1)]
    blocked_only = gate_a_trigger[~((gate_a_trigger["bp_15m"] < 0.35) & (gate_a_trigger["dir_15m"] == -1))]
    print(f"\n  Rescued (bp<0.35 & dir==-1): {len(rescued)}")
    if len(rescued) > 0:
        print(f"    Rescued WR (NO): {(rescued['resolved_yes']==0).mean()*100:.1f}%")
    print(f"  Blocked (no rescue):         {len(blocked_only)}")
    if len(blocked_only) > 0:
        print(f"    Blocked WR (NO): {(blocked_only['resolved_yes']==0).mean()*100:.1f}%")
        pnl_blocked = blocked_only.apply(lambda r: calc_pnl(r, "NO"), axis=1)
        print(f"    PnL of blocked (counterfactual): ${pnl_blocked.sum():.2f}")

# ── HMM regime gate analysis ─────────────────────────────────────────────────
print("\n" + "="*60)
print("HMM REGIME GATE ANALYSIS (Scenarios 3/4)")
print("="*60)

yes_trades = df[df["base_side"] == "YES"].copy()
print(f"\nBase YES trades: {len(yes_trades)}")

hmm_1h_block = yes_trades[yes_trades["hmm_regime_1h"] == "Bear"]
print(f"\nBlocked by HMM 1h Bear: {len(hmm_1h_block)}")
if len(hmm_1h_block) > 0:
    wr_yes = (hmm_1h_block["resolved_yes"] == 1).mean()
    print(f"  WR (YES) in blocked: {wr_yes*100:.1f}%")
    print(f"  WR (YES) all YES:    {(yes_trades['resolved_yes']==1).mean()*100:.1f}%")
    pnl_cf = hmm_1h_block.apply(lambda r: calc_pnl(r, "YES"), axis=1)
    print(f"  PnL saved: ${-pnl_cf.sum():.2f}")

# 15m bear gate (after 1h bear gate)
remaining_yes = yes_trades[yes_trades["hmm_regime_1h"] != "Bear"]
cpu_valid = remaining_yes[remaining_yes["composite_p_up"].notna()]
hmm_15m_block = cpu_valid[
    (cpu_valid["hmm_regime_15m"] == "Bear") & (cpu_valid["composite_p_up"] > 0.488)
]
print(f"\nBlocked by HMM 15m Bear + cpu>0.488: {len(hmm_15m_block)}")
if len(hmm_15m_block) > 0:
    wr_yes = (hmm_15m_block["resolved_yes"] == 1).mean()
    print(f"  WR (YES) in blocked: {wr_yes*100:.1f}%")
    pnl_cf = hmm_15m_block.apply(lambda r: calc_pnl(r, "YES"), axis=1)
    print(f"  PnL saved: ${-pnl_cf.sum():.2f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
