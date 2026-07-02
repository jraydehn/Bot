"""
analyze_hmm_vol_state_otm_itm.py

Backfills hmm_vol_state (R0=low-vol, R1=high-vol) into btc_scan_archive.csv
(the 1h runner archive), then breaks down profitability by:
  - R0 vs R1
  - YES OTM / YES ITM / NO OTM / NO ITM within each state

Model: models/hmm_ergodic_2state_btc_15m.pkl
  R0: σ_ann=32%, self-transition=0.978 → mean residence ~11h (near-absorbing)
  R1: σ_ann=88%, self-transition=0.923 → mean residence ~3h (near-absorbing)

Prior aggregate result (from runner comment):
  R0=$0.84/trade  R1=$0.04/trade (near-zero, -$2.69 May+)

This script recovers the finer OTM/ITM breakdown that was lost from context.

Run: python3 analyze_hmm_vol_state_otm_itm.py [--quick] [--save]
  --quick : sample 50k rows from scan archive (fast dev run)
  --save  : write backfilled hmm_vol_state back to btc_scan_archive.csv
"""
import argparse, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
MODELS  = BASE / "models"
RESULTS = BASE / "results"
DATA    = BASE / "data"

PKL_1H  = MODELS / "hmm_ergodic_2state_btc_15m.pkl"   # 2-state ergodic HMM

LOOKBACK_BARS = 20   # 20 × 15m = 5h of context per decode

FEE_RATE  = 0.07
BANKROLL  = 1_000.0
KELLY_MUL = 0.30
KELLY_CAP = 0.06

# --- load model ---------------------------------------------------------------

def load_hmm():
    pkg   = pickle.load(open(PKL_1H, "rb"))
    model = pkg["model"]
    order = sorted(range(model.n_components),
                   key=lambda s: float(np.sqrt(model.covars_[s, 0, 0])))
    rank_of = {s: i for i, s in enumerate(order)}
    print(f"  [hmm] Loaded {PKL_1H.name}  ({model.n_components} states)")
    for rank, s in enumerate(order):
        mu  = float(model.means_[s, 0])
        sig = float(np.sqrt(model.covars_[s, 0, 0]))
        ann = sig * math.sqrt(525600 / 15)
        tm  = model.transmat_[s, s]
        print(f"    R{rank}: μ={mu*100:+.4f}%/bar  σ_ann={ann:.1%}  "
              f"self-trans={tm:.3f}  mean_residence={1/(1-tm+1e-9):.0f} bars "
              f"(~{1/(1-tm+1e-9)*15/60:.1f}h)")
    return model, rank_of


# --- decode scan archive ------------------------------------------------------

def decode_archive(model, rank_of, df_sa: pd.DataFrame,
                   returns_series: pd.Series) -> pd.Series:
    """Assign R0/R1 rank to every row in df_sa via Viterbi on a lookback window."""
    ret_idx  = returns_series.index
    ret_vals = returns_series.values
    states   = []
    for ts in df_sa["logged_at"]:
        pos = ret_idx.searchsorted(ts, side="right") - 1
        if pos < LOOKBACK_BARS:
            states.append(np.nan)
            continue
        window    = ret_vals[pos - LOOKBACK_BARS + 1: pos + 1].reshape(-1, 1)
        vit       = model.predict(window)
        raw_state = int(vit[-1])
        states.append(rank_of[raw_state])
    return pd.Series(states, index=df_sa.index, name="hmm_vol_state")


def load_15m_returns() -> pd.Series:
    parquet = sorted(DATA.glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))
    if not parquet:
        parquet = sorted(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"))
    if not parquet:
        raise FileNotFoundError("No BTCUSDT 1m parquet found in data/")
    print(f"  Loading {parquet[-1].name} ...")
    df = pd.read_parquet(parquet[-1], columns=["close"])
    df.index = pd.to_datetime(df.index, utc=True)
    c15 = df["close"].resample("15min").last().dropna()
    lr  = np.log(c15 / c15.shift(1)).dropna()
    print(f"  15m returns: {len(lr):,} bars  "
          f"({lr.index[0].date()} → {lr.index[-1].date()})")
    return lr


# --- PnL helpers --------------------------------------------------------------

def contract_pnl(side: str, pm: float, bet_amount: float, resolved_yes: int) -> float:
    won = (resolved_yes == 1 and side == "yes") or (resolved_yes == 0 and side == "no")
    if side == "yes":
        return bet_amount * (1 - pm) / pm if won else -bet_amount
    else:
        return bet_amount * pm / (1 - pm) if won else -bet_amount


def kelly_bet(edge: float, risk: float) -> float:
    n = min(edge / risk * KELLY_MUL, KELLY_CAP) * BANKROLL / risk
    return max(n, 0.0)


# --- segment stats printer ----------------------------------------------------

def seg_stats(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"  {label:<40}  n=0"
    wr  = df["won"].mean()
    pnl = df["pnl"].sum()
    ppt = df["pnl"].mean()
    bkev = 1 / (1 + df["pnl_if_win"].mean() / df["bet_amount"].abs().mean()) if len(df) else 0.5
    return (f"  {label:<40}  n={len(df):>5}  WR={wr:.1%}  "
            f"PnL=${pnl:>+8,.0f}  $/trade={ppt:>+7.2f}  bkev={bkev:.1%}")


# --- main analysis ------------------------------------------------------------

def run(quick: bool = False, save: bool = False) -> None:
    print("\n=== Step 1: Load HMM ===")
    model, rank_of = load_hmm()

    print("\n=== Step 2: Load scan archive ===")
    sa_path = RESULTS / "btc_scan_archive.csv"
    if not sa_path.exists():
        print(f"  ERROR: {sa_path} not found")
        return

    usecols = ["logged_at", "contract_ticker", "spot", "strike", "offset_pct",
               "p_market", "tau_minutes", "resolved_yes", "spot_at_expiry",
               "hmm_vol_state"]
    # gracefully handle missing columns
    peek = pd.read_csv(sa_path, nrows=0)
    usecols = [c for c in usecols if c in peek.columns]

    if quick:
        df = pd.read_csv(sa_path, usecols=usecols, nrows=50_000, low_memory=False)
        print(f"  Quick mode: {len(df):,} rows")
    else:
        print(f"  Loading full archive (may take a minute)...")
        df = pd.read_csv(sa_path, usecols=usecols, low_memory=False)
        print(f"  {len(df):,} rows")

    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed")
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df["p_market"]     = pd.to_numeric(df["p_market"],     errors="coerce")
    df["tau_minutes"]  = pd.to_numeric(df["tau_minutes"],  errors="coerce")
    df["spot"]         = pd.to_numeric(df["spot"],         errors="coerce")
    df["strike"]       = pd.to_numeric(df["strike"],       errors="coerce")
    if "offset_pct" in df.columns:
        df["offset_pct"] = pd.to_numeric(df["offset_pct"], errors="coerce")

    # resolved only
    df_r = df[df["resolved_yes"].notna()].copy()
    print(f"  Resolved: {len(df_r):,}")

    print("\n=== Step 3: Decode HMM states ===")
    # Use existing column if already filled; backfill missing
    need_decode = True
    if "hmm_vol_state" in df_r.columns:
        filled = df_r["hmm_vol_state"].notna().sum()
        print(f"  Pre-filled hmm_vol_state: {filled}/{len(df_r)} rows")
        if filled / len(df_r) > 0.8:
            need_decode = False
            print("  Sufficient pre-fill — skipping decode")

    if need_decode:
        print("  Loading 15m returns for decode...")
        returns = load_15m_returns()
        df_r["hmm_vol_state"] = decode_archive(model, rank_of, df_r, returns)
        filled = df_r["hmm_vol_state"].notna().sum()
        print(f"  Decoded {filled}/{len(df_r)} rows")

        if save and not quick:
            df.loc[df_r.index, "hmm_vol_state"] = df_r["hmm_vol_state"]
            df.to_csv(sa_path, index=False)
            print(f"  Saved hmm_vol_state back to {sa_path.name}")

    df_r = df_r[df_r["hmm_vol_state"].notna()].copy()
    df_r["hmm_vol_state"] = df_r["hmm_vol_state"].astype(int)
    print(f"  State distribution: {df_r['hmm_vol_state'].value_counts().to_dict()}")

    # --- build simple Kelly trade simulation ----------------------------------
    # Use best-edge scan-by-scan approach (no gate stack) to measure raw regime edge
    print("\n=== Step 4: Compute Kelly PnL per scan cycle ===")
    MIN_EDGE = 0.005

    # Derive YES probability from lognormal formula (simplified — vol_eff not in archive)
    # Use p_market directly as baseline for edge; model p_yes from tau+offset
    from scipy.stats import norm

    rows = []
    for ts, grp in df_r.groupby("logged_at"):
        best_edge, best_r = MIN_EDGE, None
        for _, row in grp.iterrows():
            pm  = float(row["p_market"])
            tau = float(row["tau_minutes"])
            ry  = int(row["resolved_yes"])
            st  = int(row["hmm_vol_state"])
            op  = float(row.get("offset_pct", np.nan))
            fee = FEE_RATE * min(pm, 1 - pm)

            if not (0.10 <= pm <= 0.90 and 5 <= tau <= 150):
                continue

            # Determine ITM/OTM: YES-ITM = pm >= 0.50, YES-OTM = pm < 0.50
            # For NO: NO-ITM = pm < 0.50 (contract likely to resolve NO), NO-OTM = pm >= 0.50
            yes_type = "YES_ITM" if pm >= 0.50 else "YES_OTM"
            no_type  = "NO_ITM"  if pm < 0.50  else "NO_OTM"

            # Simple edge: use vol_eff proxy as lognormal p_yes if available,
            # else use a simple reversion proxy
            # Since we don't have vol_eff in the archive, use p_market as benchmark
            # and compute edge from the model vs market
            # For this analysis: treat contracts at face value — each resolved_yes is the truth
            # Measure: how does WR/PnL differ by state + contract type?

            for side, bet_type in [("yes", yes_type), ("no", no_type)]:
                edge = (pm - fee) if side == "no" else (1 - pm - fee)
                risk = pm if side == "yes" else (1 - pm)
                if edge > best_edge and risk > 0:
                    won = (ry == 1 and side == "yes") or (ry == 0 and side == "no")
                    bet = kelly_bet(edge, risk)
                    if side == "yes":
                        pnl = bet * (1 - pm) / pm if won else -bet
                        pnl_if_win = bet * (1 - pm) / pm
                    else:
                        pnl = bet * pm / (1 - pm) if (not won) else -bet
                        pnl_if_win = bet * pm / (1 - pm)
                    best_edge = edge
                    best_r = {
                        "ts": ts, "side": side, "bet_type": bet_type,
                        "pm": pm, "edge": edge, "won": won,
                        "bet": bet, "pnl": pnl, "pnl_if_win": pnl_if_win,
                        "bet_amount": bet, "state": st, "tau": tau, "offset_pct": op,
                    }
        if best_r:
            rows.append(best_r)

    trades = pd.DataFrame(rows)
    print(f"  {len(trades)} simulated trades")

    # ── Full breakdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  R0 vs R1 — all bet types")
    print(f"{'='*72}")
    for state, label in [(0, "R0 (low-vol,  σ=32%, ~11h residence)"),
                         (1, "R1 (high-vol, σ=88%,  ~3h residence)")]:
        sub = trades[trades["state"] == state]
        print(f"\n  [{label}]  total trades={len(sub)}")
        print(seg_stats(sub, "  ALL"))
        for bt in ["YES_OTM", "YES_ITM", "NO_OTM", "NO_ITM"]:
            print(seg_stats(sub[sub["bet_type"] == bt], f"  {bt}"))

    print(f"\n{'='*72}")
    print("  Cross-tab: state × bet_type")
    print(f"{'='*72}")
    for bt in ["YES_OTM", "YES_ITM", "NO_OTM", "NO_ITM"]:
        sub = trades[trades["bet_type"] == bt]
        print(f"\n  [{bt}]")
        for state, label in [(0, "R0"), (1, "R1")]:
            ss = sub[sub["state"] == state]
            print(seg_stats(ss, f"    {label}"))

    # ── p_market buckets by state ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  p_market buckets by state (0.10 steps)")
    print(f"{'='*72}")
    buckets = [(0.10,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),
               (0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90)]
    for state, label in [(0, "R0"), (1, "R1")]:
        ss = trades[trades["state"] == state]
        print(f"\n  [{label}]")
        for lo, hi in buckets:
            bk = ss[(ss["pm"] >= lo) & (ss["pm"] < hi)]
            if len(bk) >= 5:
                print(seg_stats(bk, f"    pm=[{lo:.2f},{hi:.2f})"))

    print(f"\n{'='*72}")
    print("  SUMMARY — what to gate based on state")
    print(f"{'='*72}")
    for state, label in [(0, "R0 (low-vol)"), (1, "R1 (high-vol)")]:
        ss = trades[trades["state"] == state]
        print(f"\n  {label}:")
        for bt in ["YES_OTM", "YES_ITM", "NO_OTM", "NO_ITM"]:
            bsub = ss[ss["bet_type"] == bt]
            if len(bsub) < 5:
                print(f"    {bt}: n={len(bsub)} (too few)")
                continue
            wr  = bsub["won"].mean()
            ppt = bsub["pnl"].mean()
            verdict = "ALLOW" if ppt > 0 else "CAUTION" if ppt > -1 else "BLOCK"
            print(f"    {bt}: WR={wr:.1%}  $/trade={ppt:>+.2f}  → {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Sample 50k rows (fast dev run)")
    parser.add_argument("--save",  action="store_true",
                        help="Write hmm_vol_state back to btc_scan_archive.csv")
    args = parser.parse_args()
    run(quick=args.quick, save=args.save)
