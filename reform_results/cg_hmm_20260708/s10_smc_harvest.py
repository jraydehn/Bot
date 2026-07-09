"""
S10 -- Harvest the 3 unused states (Markup=0, Markdown=1, Divergence=3) of the
already-live hmm_smc_phases.pkl (only State 2/ChoCH is currently gated).

Reconstructs the state sequence causally over BTC's full 1h history using the
EXACT production get_smc_signals() (smc_signals.py) + the EXACT trained model's
sequential buffer-decode (deque maxlen=24, predict only once len>=3) -- NOT a
point-in-time re-decode, matching how the live runner actually calls it.
Zero-lookahead: only completed 1h/4h bars used at every point (matches the
hourly runner's dominant convention elsewhere -- p_up_v3, Keltner-done, CG
decoder all use completed-bars-only).

Joins the reconstructed state to the REAL combined paper_trades archive chain
(same 4-file chain used for the p_up_v3 zero-lookahead re-validation) so the
signal check itself is on real taken trades, not synthetic candidates.
"""
import pickle
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from smc_signals import get_smc_signals

OUT = "reform_results/cg_hmm_20260708"

with open("hmm_smc_phases.pkl", "rb") as f:
    pkg = pickle.load(f)
model, scaler, FEAT_COLS = pkg["model"], pkg["scaler"], pkg["feat_cols"]
print(f"model: {pkg['n_states']} states, feats={FEAT_COLS}")

close1h_full = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet").sort_index()
print(f"1h history: {len(close1h_full)} bars {close1h_full.index.min()} -> {close1h_full.index.max()}")

_bos_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


def smc_vec_at(idx):
    """idx = integer position into close1h_full; only bars [0, idx] used (completed, causal)."""
    df_1h = close1h_full.iloc[max(0, idx - 600):idx + 1]
    if len(df_1h) < 60:
        return None
    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    if len(df_4h) < 20:
        return None
    spot = float(df_1h["close"].iloc[-1])
    try:
        r = get_smc_signals(df_1h, df_4h, spot)
    except Exception:
        return None
    return [
        _bos_map.get(r.bos_4h, 0.0), _bos_map.get(r.bos_1h, 0.0),
        float(r.choch_4h), float(r.choch_1h),
        float(min(20.0, max(0.0, r.nearest_supply_pct)) if r.nearest_supply_pct is not None else 10.0),
        float(min(20.0, max(0.0, r.nearest_demand_pct)) if r.nearest_demand_pct is not None else 10.0),
        float(r.n_supply_zones - r.n_demand_zones),
        float(r.in_supply_zone), float(r.in_demand_zone),
    ]


# ── decode over the full history, matching the live buffer-based predict ──
STEP_HOURS = 1
START_IDX = 700  # warmup for the 800-bar zone lookback inside get_smc_signals
indices = list(range(START_IDX, len(close1h_full), STEP_HOURS))
print(f"decoding {len(indices)} hourly points (this recomputes SMC zones every step -- slow)...")

from collections import deque
buf = deque(maxlen=24)
rows = []
for n, idx in enumerate(indices):
    vec = smc_vec_at(idx)
    if vec is None:
        continue
    scaled = scaler.transform([vec])[0]
    buf.append(scaled)
    if len(buf) < 3:
        continue
    state = int(model.predict(np.array(list(buf)))[-1])
    rows.append({"ts": close1h_full.index[idx], "smc_state": state})
    if n % 500 == 0:
        print(f"  {n}/{len(indices)}  ts={close1h_full.index[idx]}  state={state}")

sv = pd.DataFrame(rows)
sv.to_csv(f"{OUT}/smc_hmm_states_full.csv", index=False)
print(f"\ndecoded {len(sv)} states. distribution:")
print(sv["smc_state"].value_counts().sort_index())
print("DONE_S10")
