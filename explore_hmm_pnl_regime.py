"""
P&L Regime HMM — explore and train.

Observation: rolling 10-trade normalized edge (realized WR - market BE).
States: 0 = edge-active, 1 = edge-degraded.
Action: 0.5x Kelly when state=1.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from pathlib import Path

WINDOW   = 10    # rolling trade window
N_STATES = 2
MODEL_PATH = Path("models/hmm_pnl_regime_btc.pkl")

# ── load paper trades ────────────────────────────────────────────────────────
pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt["logged_at"]    = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
pt["would_pnl"]    = pd.to_numeric(pt["would_pnl"], errors="coerce")
pt["resolved_yes"] = pd.to_numeric(pt["resolved_yes"], errors="coerce")
pt["p_market"]     = pd.to_numeric(pt["p_market"], errors="coerce")
pt["bet_amount"]   = pd.to_numeric(pt["bet_amount"], errors="coerce")

btc = pt[pt["contract_ticker"].str.contains("KXBTCD", na=False)].copy()
btc = btc[(btc["decision"]=="trade") & btc["resolved_yes"].notna() & btc["would_pnl"].notna()].copy()
btc = btc.sort_values("logged_at").reset_index(drop=True)
print(f"BTC resolved trades: {len(btc)}  "
      f"({btc['logged_at'].min().date()} → {btc['logged_at'].max().date()})")

# ── build observation sequence ───────────────────────────────────────────────
# Two features per rolling window:
#   f1: rolling edge = mean(resolved_yes) - mean(p_market)  → is model finding edge?
#   f2: rolling P&L z-score vs all-time mean/std             → absolute magnitude signal
btc["win"] = (btc["resolved_yes"] == 1).astype(float)
btc["roll_wr"]  = btc["win"].rolling(WINDOW).mean()
btc["roll_be"]  = btc["p_market"].rolling(WINDOW).mean()
btc["roll_edge"] = btc["roll_wr"] - btc["roll_be"]
btc["roll_pnl"]  = btc["would_pnl"].rolling(WINDOW).sum()

# Per-trade normalized P&L (return on amount risked)
btc["pnl_norm"] = btc["would_pnl"] / btc["bet_amount"].clip(lower=1.0)
btc["roll_pnl_norm"] = btc["pnl_norm"].rolling(WINDOW).mean()

seq = btc.dropna(subset=["roll_edge","roll_pnl_norm"]).copy().reset_index(drop=True)
print(f"Rolling {WINDOW}-trade sequence: n={len(seq)}")

X_raw = seq[["roll_edge","roll_pnl_norm"]].values

# ── scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── train HMM ────────────────────────────────────────────────────────────────
best_model, best_ll = None, -np.inf
for seed in range(20):
    try:
        m = GaussianHMM(n_components=N_STATES, covariance_type="full",
                        n_iter=200, random_state=seed)
        m.fit(X)
        ll = m.score(X)
        if ll > best_ll:
            best_ll, best_model = ll, m
    except Exception:
        pass

model = best_model
states = model.predict(X)
seq["hmm_state"] = states

print(f"\nLog-likelihood: {best_ll:.2f}")
print(f"\nTransition matrix:")
print(model.transmat_.round(3))

# ── identify which state is "degraded" ───────────────────────────────────────
state_pnl = {s: seq[seq["hmm_state"]==s]["roll_pnl"].mean() for s in range(N_STATES)}
degraded_state = min(state_pnl, key=lambda s: state_pnl[s])
active_state   = 1 - degraded_state

print(f"\nState means (raw):")
for s in range(N_STATES):
    sub = seq[seq["hmm_state"]==s]
    m_edge = sub["roll_edge"].mean()
    m_pnl  = sub["roll_pnl"].mean()
    n      = len(sub)
    label  = "DEGRADED" if s==degraded_state else "active  "
    print(f"  St{s} [{label}] n={n:3d}  roll_edge={m_edge:+.3f}  roll_pnl=${m_pnl:+.1f}")

# Transition probabilities from active to degraded and vice versa
p_ad = model.transmat_[active_state, degraded_state]
p_dd = model.transmat_[degraded_state, degraded_state]
p_da = model.transmat_[degraded_state, active_state]
p_aa = model.transmat_[active_state, active_state]

mean_dur_active   = 1 / (1 - p_aa) if p_aa < 1 else float("inf")
mean_dur_degraded = 1 / (1 - p_dd) if p_dd < 1 else float("inf")
print(f"\nTransition probabilities:")
print(f"  Active→Degraded:   {p_ad:.3f}  (mean active duration  = {mean_dur_active:.1f} trades)")
print(f"  Degraded→Active:   {p_da:.3f}  (mean degraded duration = {mean_dur_degraded:.1f} trades)")

# ── walk-forward validation ──────────────────────────────────────────────────
mid = len(seq)//2
seq_train = seq.iloc[:mid].copy()
seq_test  = seq.iloc[mid:].copy()

m_tr = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=200, random_state=0)
m_tr.fit(X[:mid])
states_test = m_tr.predict(X[mid:])

# Map degraded state (lowest roll_pnl mean in train)
tr_pnl = {s: seq_train[seq_train["hmm_state"]==s]["roll_pnl"].mean() for s in range(N_STATES)}
deg_tr = min(tr_pnl, key=lambda s: tr_pnl[s])
act_tr = 1 - deg_tr

# Evaluate on test: P&L when active vs degraded
seq_test = seq_test.copy()
seq_test["wf_state"] = states_test
# Map test states to same semantics (check which test state has lower mean)
test_pnl = {s: seq_test[seq_test["wf_state"]==s]["roll_pnl"].mean() for s in range(N_STATES)}
deg_te = min(test_pnl, key=lambda s: test_pnl[s])
act_te = 1 - deg_te

print(f"\nWalk-forward (train=first {mid}, test=last {len(seq)-mid}):")
for s in range(N_STATES):
    label = "DEGRADED" if s==deg_tr else "active  "
    lbl_te = "DEGRADED" if s==deg_te else "active  "
    sub_tr = seq_train[seq_train["hmm_state"]==s]
    sub_te = seq_test[seq_test["wf_state"]==s]
    print(f"  St{s} train[{label}] n={len(sub_tr):3d} pnl=${sub_tr['roll_pnl'].mean():+.1f}  "
          f"test[{lbl_te}]  n={len(sub_te):3d} pnl=${sub_te['roll_pnl'].mean():+.1f}")

# ── Kelly impact simulation ───────────────────────────────────────────────────
# Per actual trade: when state=degraded apply 0.5x Kelly (so bet half as much)
# Map each trade to a state using the full-data model
seq_indexed = seq.set_index(seq.index)
btc_with_state = btc.copy()
btc_with_state = btc_with_state.loc[seq.index].copy()
btc_with_state["hmm_state"] = seq["hmm_state"].values

baseline_pnl = btc_with_state["would_pnl"].sum()
dampened_pnl = 0.0
for _, row in btc_with_state.iterrows():
    mult = 0.5 if row["hmm_state"] == degraded_state else 1.0
    dampened_pnl += row["would_pnl"] * mult

print(f"\nKelly impact simulation (0.5x in degraded state):")
print(f"  Baseline P&L:  ${baseline_pnl:+,.2f}")
print(f"  Dampened P&L:  ${dampened_pnl:+,.2f}")
print(f"  Delta:         ${dampened_pnl-baseline_pnl:+,.2f}")

degraded_pnl = btc_with_state[btc_with_state["hmm_state"]==degraded_state]["would_pnl"].sum()
active_pnl   = btc_with_state[btc_with_state["hmm_state"]==active_state]["would_pnl"].sum()
degraded_n   = (btc_with_state["hmm_state"]==degraded_state).sum()
active_n     = (btc_with_state["hmm_state"]==active_state).sum()
print(f"\n  Active  state trades: n={active_n},   sum_pnl=${active_pnl:+,.2f}")
print(f"  Degraded state trades: n={degraded_n}, sum_pnl=${degraded_pnl:+,.2f}")

# ── timeline printout ─────────────────────────────────────────────────────────
print(f"\nState timeline (10-trade windows, D=degraded A=active):")
seq["date_str"] = seq["logged_at"].dt.strftime("%m/%d %H:%M")
prev_date = None
for _, row in seq.iterrows():
    date = str(row["logged_at"].date())
    label = "D" if row["hmm_state"]==degraded_state else "A"
    if date != prev_date:
        print(f"\n  {date}: ", end="")
        prev_date = date
    print(label, end="")
print()

# ── save model ────────────────────────────────────────────────────────────────
pkg = dict(
    model=model,
    scaler=scaler,
    n_states=N_STATES,
    degraded_state=degraded_state,
    active_state=active_state,
    window=WINDOW,
    features=["roll_edge","roll_pnl_norm"],
    kelly_mult_degraded=0.5,
)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nModel saved → {MODEL_PATH}")
print("Done.")
