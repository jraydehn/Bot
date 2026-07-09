"""
S6 -- z_drift at multiple windows (1h/2h/3h/6h/9h/12h/24h), exact live formula:
mean over resolved rows with decision_time in [T-w, T) of
  log(spot_at_expiry/spot) / (realized_vol_annual * sqrt(tau/525600))
CAUSALITY: a row enters trade T's window ONLY if its close_ts <= T (resolved
by then) -- the live runner gets this for free (unresolved rows have no
spot_at_expiry yet); a naive archive reconstruction would NOT (final values
are backfilled) and would smuggle in outcomes that postdate the trade.

Steps: (1) reconstruct all windows for the full NO book, (2) PARITY-check
the reconstructed 6h against the logged z_drift_6h, (3) per-window gate
threshold curves on the full book (is 6h even the best window?), (4)
cross-window structure inside the z6<0.55 bucket.
"""
import warnings
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1049)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]
WINDOWS = [1, 2, 3, 6, 9, 12, 24]

# ── archive event list ─────────────────────────────────────────────────────
def parse_mixed(s):
    def _u(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_u(v) for v in s], utc=True)

sa = pd.read_csv("results/sol_scan_archive_15m.csv", low_memory=False,
                 usecols=["logged_at", "close_ts", "spot", "realized_vol_annual",
                          "tau_minutes", "spot_at_expiry"])
for c in ["spot", "realized_vol_annual", "tau_minutes", "spot_at_expiry"]:
    sa[c] = pd.to_numeric(sa[c], errors="coerce")
sa["dts"] = parse_mixed(sa["logged_at"])
sa["cts"] = parse_mixed(sa["close_ts"])
sa = sa.dropna(subset=["dts", "cts", "spot", "realized_vol_annual", "tau_minutes", "spot_at_expiry"])
sa = sa[(sa["spot"] > 0) & (sa["realized_vol_annual"] > 0) & (sa["tau_minutes"] > 0)
        & (sa["spot_at_expiry"] > 0)]
sigma = sa["realized_vol_annual"] * np.sqrt(sa["tau_minutes"] / 525600.0)
sa["z"] = np.log(sa["spot_at_expiry"] / sa["spot"]) / sigma.replace(0, np.nan)
sa = sa.dropna(subset=["z"]).sort_values("dts").reset_index(drop=True)
D = sa["dts"].values.astype("datetime64[ns]")
C = sa["cts"].values.astype("datetime64[ns]")
Z = sa["z"].values
print(f"archive z-events: {len(sa)}  {sa['dts'].min()} -> {sa['dts'].max()}")

no = pd.read_csv(f"{OUT}/no_book_reconstructed.csv", low_memory=False)
no["logged_at_p"] = pd.to_datetime(no["logged_at_p"], utc=True)
no["week"] = no["logged_at_p"].dt.to_period("W-FRI").astype(str)


def zdrift_at(T, w_hours):
    Tn = np.datetime64(T.tz_convert("UTC").tz_localize(None))
    lo = np.searchsorted(D, Tn - np.timedelta64(w_hours, "h"), side="left")
    hi = np.searchsorted(D, Tn, side="left")
    if hi <= lo:
        return np.nan
    seg_c = C[lo:hi]
    seg_z = Z[lo:hi]
    ok = seg_c <= Tn                     # resolved BY trade time -- causality guard
    if ok.sum() < 3:
        return np.nan
    return float(seg_z[ok].mean())


print("reconstructing windows for the full NO book...")
for w in WINDOWS:
    no[f"zw_{w}h"] = no["logged_at_p"].apply(lambda t: zdrift_at(t, w))
    print(f"  {w}h: coverage {no[f'zw_{w}h'].notna().sum()}/{len(no)}")

# ── parity check vs logged 6h ─────────────────────────────────────────────
both = no.dropna(subset=["z_drift_6h", "zw_6h"])
corr = both["z_drift_6h"].corr(both["zw_6h"])
mae = (both["z_drift_6h"] - both["zw_6h"]).abs().mean()
print(f"\nPARITY 6h recon vs logged: n={len(both)}  corr={corr:.3f}  MAE={mae:.4f}")
if corr < 0.85:
    print("  !! parity weak -- treat cross-window results as approximate, findings on the")
    print("     RECONSTRUCTED 6h must match the logged-6h gate result before trusting others")
    m_r = (both["zw_6h"] < 0.55)
    d = both[m_r]
    print(f"  recon-6h<0.55 bucket: n={len(d)} edge={d['tedge'].mean():+.4f} "
          f"(logged-6h gave -0.0467) -- sanity anchor")

# ── per-window gate curves (full book) ────────────────────────────────────
def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means >= 0).mean()

print(f"\n=== per-window gate curves: block NO when zw < thr ===")
print(f"{'win':>4} {'thr':>6} {'n':>5} {'eps':>4} {'ep_edge':>8} {'P_pos':>7} {'$':>10} {'streak':>6} "
      f"{'complement_edge':>16}")
best = {}
for w in WINDOWS:
    col = no[f"zw_{w}h"]
    if col.notna().sum() < 400:
        continue
    rows = []
    for q in [0.3, 0.4, 0.5, 0.6, 0.7]:
        thr = col.quantile(q)
        m = (col < thr).fillna(False)
        d = no[m]
        if len(d) < 100:
            continue
        ne, ee, pp = ep_stats(d)
        comp = no[~m & col.notna()]
        sh = int(d["contract_ticker"].isin(STREAK).sum())
        rows.append((thr, len(d), ne, ee, pp, d["would_pnl"].sum(), sh,
                     comp["tedge"].mean()))
    for thr, n, ne, ee, pp, pnl, sh, ce in rows:
        print(f"{w:>3}h {thr:>6.2f} {n:>5} {ne:>4} {ee:>+8.4f} {pp:>7.4f} {pnl:>+10.2f} {sh:>4}/4 {ce:>+16.4f}")
    # best = deepest significant ep_edge with 4/4 streak
    cands = [r for r in rows if r[4] <= 0.05 and r[6] == 4]
    if cands:
        best[w] = min(cands, key=lambda r: r[3])

print("\nbest qualifying threshold per window (P<=0.05, streak 4/4):")
for w, r in best.items():
    print(f"  {w}h: thr={r[0]:.2f}  n={r[1]}  ep_edge={r[3]:+.4f}  P={r[4]:.4f}  $={r[5]:+.2f}  "
          f"complement={r[7]:+.4f}")

# ── cross-window structure inside the logged-6h bucket ───────────────────
print(f"\n=== cross-window structure inside z_drift_6h(logged)<0.55 bucket ===")
bucket = no[(no["z_drift_6h"] < 0.55).fillna(False)].copy()
streak_mask = bucket["contract_ticker"].isin(STREAK)
def ep2(d, n_boot=3000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()
for w in [1, 2, 3]:
    col = bucket[f"zw_{w}h"]
    if col.notna().sum() < 200:
        print(f"  zw_{w}h coverage {col.notna().sum()} -- thin")
        continue
    med = col.median()
    for lab, mk in [(f"zw_{w}h high (>= {med:.2f})", col >= med),
                    (f"zw_{w}h low (< {med:.2f})", col < med)]:
        d = bucket[mk.fillna(False)]
        ne, ee, pn = ep2(d)
        sh = int((mk.fillna(False) & streak_mask).sum())
        print(f"  {lab}: n={len(d)} eps={ne} ep_edge={ee:+.4f} P(<=0)={pn:.4f} streak={sh}/4 "
              f"$={d['would_pnl'].sum():+.2f}")

no.to_csv(f"{OUT}/no_book_zwindows.csv", index=False)
print("DONE_S6")
