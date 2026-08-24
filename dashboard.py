"""
Live paper-trading dashboard — dark mode — multi-asset (BTC, ETH, SOL).

Run with:
    python3 -m streamlit run dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

PST = ZoneInfo("America/Los_Angeles")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR     = Path(__file__).parent / "results"
REFRESH_SECONDS = 60
DISPLAY_FROM    = "2026-05-16 06:56:42"   # hide trades before this UTC time (dashboard cleared May 15 11:56 PM PDT — ETH 15m z_drift sim complete, BTC 15m branched model live)
ASSET_DISPLAY_FROM = {
    # [cleared 2026-07-28 per user request — fresh read from the last hourly
    # runner restart (fee-true accounting + HMM-state archive logging live;
    # decision logic unchanged). Prior cutoff was 2026-06-18 06:18:28.]
    # [2026-08-15 FINAL per user: BTC hourly paper = niche v1 book, displayed
    # FROM THE PROMOTION START (08-13 22:31 UTC). NOTE: the niche runner was
    # frozen 08-14 03:14 -> 08-15 16:45 UTC (watchdog cron was TCC-blocked),
    # so the display will be sparse until it resumes trading — that is
    # correct, not missing data.]
    # [2026-08-22 cleared per user request — fresh read from the
    # btc_mkv_sideways_gate deploy (gate live in production paper +
    # niche v1 + refresh runners ~03:40 UTC 08-23; cross-asset frozen
    # rule, BTC-confirmed by the 08-22 niche run −$1,177 35/35-Sideways).
    # Prior cutoff: 2026-08-13 22:31:00 (niche v1 promotion start).]
    "BTC": "2026-08-23 03:40:00",
    # [2026-08-19 per user: ETH hourly paper seat = YES-favorite book
    # (eth_hourly_fav_runner, model-free bias, promoted at the post-crash
    # fleet restart 01:50 UTC — model routes null x2, prior hourly book
    # failing). Old production book relabeled 'benchmark' in load_trades.
    # Prior cutoff: 2026-07-28 22:15:13.]
    "ETH": "2026-08-19 01:50:00",
    "SOL": "2026-07-28 22:15:13",
    "BTC_OLD": "2026-07-21 17:27:13",  # cleared 2026-07-21 — ported 9 structural/infra bug
                                    # fixes (schema drift, degenerate ms/vd/of HMM decode,
                                    # per-runner stop-loss isolation, markov_yf NameError,
                                    # missing gated-cycle marker row, p_up_v2/p_up_v3/pup15m
                                    # shadow logging, etc.) into the frozen pre-July-1 file —
                                    # gate/rescue/sizing logic is untouched, only correctness
                                    # bugs unrelated to the model itself. Want a clean read
                                    # isolated from the pre-fix period's HMM-decode etc. noise.
}
ASSET_DISPLAY_FROM_15M = {
    "BTC": "2026-08-14 03:06:00",  # cleared 2026-08-14 per user request — fresh run of the
                                    # then-promoted DUAL v2 paper book (prod g+k 12 kelly +
                                    # SHADOW/mktanchor flat $100). [2026-08-17] seat REVERTED
                                    # to DUAL v1 (prod + mkt-fav k1.8 flat) — shadow arm bled
                                    # -$911/144; cutoff NOT moved (user did not ask to clear;
                                    # display spans both eras, arm swap visible in the flat
                                    # trades). Prior cutoffs: 2026-08-12 16:55:51 (DUAL v1),
                                    # 2026-07-28 06:53:02 (fee-audit gate package).
    # [cleared 2026-08-18 per user request — fresh run of the PROMOTED
    # COMBO replica (NOtrio+YESknife joined the paper strategy at READ #1;
    # runner restarted 03:18 UTC). Prior cutoff: 2026-08-12 19:05:00.]
    "ETH": "2026-08-18 03:18:00",  # cleared 2026-08-12 per user request — fresh run of
                                    # the PROMOTED paper book (exact-replica decision path
                                    # live 19:05 UTC: production model + 5 survivor gates,
                                    # c4cd6a7). First trade shown: 19:23 UTC YES @0.395.
                                    # Prior: 2026-07-29 18:25:54 (slope-feature
                                    # model live (replaced inverted blend chain, commit
                                    # 163276d). Prior: 2026-07-28 06:53:02 (fee-audit gate
                                    # package live (3 band blocks + yes_dipvol + no_slope
                                    # gates + fee-aware Kelly). Prior: 2026-07-14 22:01:15.
                                    # [older note: 2026-07-14 retrained lgbm_15m_eth.pkl deployed
                                    # (train split extended thru 2025-07, val thru 2026-01);
                                    # want a clean read isolated from the pre-retrain model
                                    # [restored 2026-07-17 per user request after an
                                    # unprompted clear for the eth_yes_kelly_damp removal —
                                    # see project_eth15m_yes_kelly_undamp_20260717.md and
                                    # feedback_dashboard_clear_only_when_asked.md]
    "SOL": "2026-08-12 18:00:00",  # cleared 2026-08-12 per user request — fresh run of
                                    # the PROMOTED paper model (slope + zd65+path+SW became
                                    # the decision path 18:00 UTC, commit 0f9f2b2).
                                    # Prior: 2026-08-12 17:20:00 (candidate-staging clear).
                                    # (15m falls back to ASSET_DISPLAY_FROM when absent here, which
                                    # would otherwise incorrectly inherit the hourly-only 07-07 clear)
}

# Supplement paper_trades with live_trades for gap periods when paper logging was broken.
# Dedup logic in load_trades() prevents double-counting dual-mode rows already in paper_trades.
ASSET_LIVE_CSV = {
    "BTC": RESULTS_DIR / "live_trades.csv",
    "SOL": RESULTS_DIR / "live_trades_sol.csv",
}
LIVE_SUPPLEMENT_FROM = "2026-06-18 00:00:00"

ASSET_CSV = {
    "BTC": RESULTS_DIR / "paper_trades.csv",
    "ETH": RESULTS_DIR / "paper_trades_eth.csv",
    "SOL": RESULTS_DIR / "paper_trades_sol.csv",
    # [2026-07-12] Isolated pre-July-1 BTC hourly model, run standalone to test the
    # user's "revert to old model" hypothesis without contaminating the current
    # model's shared CSV/scan-archive/blocked-trades files. See
    # project_btc_hourly_pre_july1_revert_test_20260712.md.
    "BTC_OLD": RESULTS_DIR / "paper_trades_btc_OLD_pre_july1_test.csv",
}

ASSET_CSV_15M = {
    "BTC": RESULTS_DIR / "paper_trades_btc15m.csv",
    "ETH": RESULTS_DIR / "paper_trades_eth15m.csv",
    "SOL": RESULTS_DIR / "paper_trades_sol15m.csv",
}

ASSET_SPOT_SOURCES = {
    "BTC": [
        ("coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot",      lambda r: float(r.json()["data"]["amount"])),
        ("kraken",   "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",   lambda r: float(r.json()["result"]["XXBTZUSD"]["c"][0])),
        ("bitstamp", "https://www.bitstamp.net/api/v2/ticker/btcusd/",        lambda r: float(r.json()["last"])),
        ("gemini",   "https://api.gemini.com/v1/pubticker/btcusd",            lambda r: float(r.json()["last"])),
    ],
    "ETH": [
        ("coinbase", "https://api.coinbase.com/v2/prices/ETH-USD/spot",      lambda r: float(r.json()["data"]["amount"])),
        ("kraken",   "https://api.kraken.com/0/public/Ticker?pair=XETHZUSD", lambda r: float(r.json()["result"]["XETHZUSD"]["c"][0])),
        ("bitstamp", "https://www.bitstamp.net/api/v2/ticker/ethusd/",        lambda r: float(r.json()["last"])),
        ("gemini",   "https://api.gemini.com/v1/pubticker/ethusd",            lambda r: float(r.json()["last"])),
    ],
    "SOL": [
        ("coinbase", "https://api.coinbase.com/v2/prices/SOL-USD/spot",      lambda r: float(r.json()["data"]["amount"])),
        ("kraken",   "https://api.kraken.com/0/public/Ticker?pair=SOLUSD",   lambda r: float(r.json()["result"]["SOLUSD"]["c"][0])),
    ],
}

st.set_page_config(
    page_title="Kalshi Trader",
    page_icon="₿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Dark mode CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background-color: #0e0e0e !important;
    color: #e0e0e0 !important;
}}
[data-testid="stSidebar"] {{ background-color: #141414 !important; }}
[data-testid="stHeader"]  {{ background-color: #0e0e0e !important; }}
[data-testid="stMetric"] {{
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{ color: #888 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700; }}
hr {{ border-color: #2a2a2a !important; }}
[data-testid="stDataFrame"] {{ background-color: #1a1a1a !important; border-radius: 10px; }}
thead tr th {{ background-color: #1f1f1f !important; color: #aaa !important; font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #333 !important; }}
.js-plotly-plot {{ background-color: transparent !important; }}
[data-testid="stExpander"] {{ background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; }}
.stCaption {{ color: #666 !important; }}
[data-testid="stTabs"] button {{ color: #888 !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: #4da6ff !important; border-bottom-color: #4da6ff !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# [2026-08-15] Live update WITHOUT page reloads. The old mechanism was a
# <meta http-equiv="refresh"> tag — a full BROWSER reload every 60s (scroll
# reset, flicker, tab bounce). Replaced with a st.fragment timer: every
# REFRESH_SECONDS it checks the data files' mtimes and, ONLY when something
# actually changed, triggers an in-place app rerun over the websocket —
# Streamlit diffs the elements, so scroll position and the active tab are
# preserved and nothing repaints unless its content changed. Quiet periods
# cause zero re-renders at all.
# ---------------------------------------------------------------------------
@st.fragment(run_every=REFRESH_SECONDS)
def _live_update():
    import glob as _g, os as _os
    _files = (_g.glob(str(RESULTS_DIR / "paper_trades*.csv"))
              + _g.glob(str(RESULTS_DIR / "*backfill*.csv")))
    try:
        _sig = max(_os.path.getmtime(_f) for _f in _files)
    except ValueError:
        return
    if st.session_state.get("_data_sig") is None:
        st.session_state["_data_sig"] = _sig
    elif _sig != st.session_state["_data_sig"]:
        st.session_state["_data_sig"] = _sig
        st.rerun(scope="app")

_live_update()


# ---------------------------------------------------------------------------
# Data loaders (cached per asset)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=55)
def fetch_spot(asset: str) -> dict:
    sources = ASSET_SPOT_SOURCES.get(asset, [])
    prices = {}
    for name, url, parser in sources:
        try:
            r = requests.get(url, timeout=6)
            prices[name] = parser(r)
        except Exception:
            pass
    avg = sum(prices.values()) / len(prices) if prices else None
    return {"prices": prices, "avg": avg}


@st.cache_data(ttl=55)
def _load_csv(path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")
    df = df[df["logged_at"].notna()]
    for col in ["resolved_yes", "would_win"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)
    return df


_MONTH_NUM = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
              'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

def _ticker_to_close_ts(ticker: str):
    """Parse close timestamp from Kalshi hourly ticker (KXBTCD/KXSOLD/KXETHD-YYMMMDDHH-TSTRIKE).
    close_ts = ticker date + (ticker_hour + 4) hours UTC."""
    import re, datetime as dt
    m = re.match(r'KX\w+-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T', str(ticker))
    if not m:
        return None
    yy, mmm, dd, hh = m.groups()
    month = _MONTH_NUM.get(mmm)
    if month is None:
        return None
    base = dt.datetime(2000 + int(yy), month, int(dd), 0, 0, tzinfo=dt.timezone.utc)
    return base + dt.timedelta(hours=int(hh) + 4)


@st.cache_data(ttl=55)
def _load_live_supplement(path) -> pd.DataFrame:
    """Load live_trades hourly contracts only, mapped to paper_trades schema for gap-filling."""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")
    df = df[df["logged_at"].notna()].copy()
    # Keep only hourly contracts (KXSOLD / KXETHD) — exclude 15m tickers already in paper_trades_*15m.csv
    df = df[~df["contract_ticker"].str.contains("15M", na=False)]
    df = df.rename(columns={"live_pnl": "would_pnl"})
    # Derive would_win from P&L sign (works for both YES and NO bets; resolved_yes alone is wrong for NO)
    pnl_num = pd.to_numeric(df["would_pnl"], errors="coerce")
    df["would_win"] = pnl_num.apply(lambda x: True if x > 0 else (False if x < 0 else pd.NA))
    df["decision"]  = "trade"
    df["timeframe"] = "1h"
    # Compute tau_minutes from ticker close_ts (live_trades CSVs don't store it)
    logged_utc = df["logged_at"].dt.tz_convert("UTC")
    def _tau(row):
        close = _ticker_to_close_ts(row["contract_ticker"])
        if close is None:
            return float("nan")
        return (close - row["_logged_utc"]).total_seconds() / 60.0
    df["_logged_utc"] = logged_utc
    df["tau_minutes"] = df.apply(_tau, axis=1)
    df = df.drop(columns=["_logged_utc"])
    return df


def load_trades(asset: str) -> pd.DataFrame:
    df_1h  = _load_csv(ASSET_CSV.get(asset))
    df_15m = _load_csv(ASSET_CSV_15M.get(asset))

    if not df_1h.empty:
        df_1h["timeframe"] = "1h"
    # [2026-08-15 FINAL] BTC hourly PAPER = niche v1 (promoted 08-13).
    # Production trades relabeled 'benchmark' (rows stay for signal
    # panels); niche trades appended, normalized like _load_csv.
    if asset == "BTC" and not df_1h.empty:
        try:
            df_1h.loc[df_1h["decision"] == "trade", "decision"] = "benchmark"
            _nv = pd.read_csv(RESULTS_DIR / "paper_trades_btc_hourly_niche.csv",
                              low_memory=False)
            _nv["logged_at"] = pd.to_datetime(
                _nv["logged_at"], format="mixed", utc=True,
                errors="coerce").dt.tz_convert("America/Los_Angeles")
            _nv = _nv[_nv["logged_at"].notna()]
            _nv = _nv.rename(columns={"p_model": "p_yes_model"})
            _nv["side"] = "yes"
            _nv["decision"] = "trade"
            _nv["timeframe"] = "1h"
            _nv["bet_amount"] = pd.to_numeric(_nv.get("stake"),
                                              errors="coerce").fillna(100.0)
            _nv["net_edge"] = _nv.get("fee_adj_edge")
            df_1h = pd.concat([df_1h, _nv], ignore_index=True, sort=False)
        except Exception:
            pass
    # [2026-08-19] ETH hourly PAPER = YES-favorite book (promoted at the
    # post-crash restart per user — model routes null x2, prior book
    # failing). Same pattern as BTC/niche above; the fav book is
    # model-free, so appended rows have no p_yes_model/net_edge.
    if asset == "ETH" and not df_1h.empty:
        try:
            df_1h.loc[df_1h["decision"] == "trade", "decision"] = "benchmark"
            _fv = pd.read_csv(RESULTS_DIR / "paper_trades_eth_hourly_fav.csv",
                              low_memory=False)
            _fv["logged_at"] = pd.to_datetime(
                _fv["logged_at"], format="mixed", utc=True,
                errors="coerce").dt.tz_convert("America/Los_Angeles")
            _fv = _fv[_fv["logged_at"].notna()]
            # [2026-08-19] live-fill accounting: unfilled attempts (filled=0)
            # are recorded in the book but are not trades — exclude here.
            if "filled" in _fv.columns:
                _fv = _fv[pd.to_numeric(_fv["filled"], errors="coerce") == 1]
            _fv["side"] = "yes"
            _fv["decision"] = "trade"
            _fv["timeframe"] = "1h"
            _fv["bet_amount"] = pd.to_numeric(_fv.get("stake"),
                                              errors="coerce").fillna(100.0)
            df_1h = pd.concat([df_1h, _fv], ignore_index=True, sort=False)
        except Exception:
            pass
    if not df_15m.empty:
        # Normalise 15m column names to match hourly dashboard expectations
        df_15m = df_15m.rename(columns={
            "p_model_15m": "p_yes_model",
            "floor_strike": "strike",
        })
        df_15m["timeframe"] = "15m"
        df_15m["net_edge"]  = df_15m.get("raw_edge", pd.NA)

        # [2026-07-20] Dedup live + paper-twin duplicate rows. When a live process
        # and its paper twin both run for the same asset (see
        # feedback_parallel_paper_runner), each independently evaluates every
        # contract and logs its own row to the SAME paper_trades_{asset}15m.csv --
        # e.g. ETH 15m since its 07-18 go-live. These are two real, distinct
        # decisions (one real money, one simulated), not a data-corruption bug,
        # but showing both inflates the dashboard's trade count / PnL ~2x.
        # Merge rows within the same (contract_ticker, decision, side) group that
        # land within 90s of the previous row into one "session", keeping the
        # is_live=1 row (the real fill) when both exist. Adjacency-based (not a
        # fixed time-grid floor) so a live/twin pair landing right on a minute
        # boundary still merges correctly; 90s is well under the 5-min scan
        # cadence (LOOP_INTERVAL_SEC), so genuinely distinct re-scans of an
        # unresolved contract never collapse together. is_live is blank on rows
        # logged before this column existed (pre-2026-07-20) -- those can't be
        # tagged retroactively, so this only cleanly prefers the live row going
        # forward (older rows still merge, just without a live/paper preference).
        _key_cols = [c for c in ["contract_ticker", "decision", "side"] if c in df_15m.columns]
        # [2026-08-12] BTC DUAL paper book: the mkt-fav arm (flat $100,
        # kelly_fraction=0) can legitimately trade the SAME contract/side
        # in the same scan as the production arm — key the dedup on the
        # flat-stake fingerprint so those two rows never merge. Live/twin
        # duplicate pairs are both kelly-sized, so their dedup is unchanged.
        if {"bet_amount", "kelly_fraction"} <= set(df_15m.columns):
            _flat15 = ((pd.to_numeric(df_15m["bet_amount"], errors="coerce") == 100.0)
                       & (pd.to_numeric(df_15m["kelly_fraction"], errors="coerce") == 0.0))
            df_15m["_flat"] = _flat15.fillna(False)
            _key_cols.append("_flat")
        if _key_cols:
            _ts = pd.to_datetime(df_15m["logged_at"], errors="coerce", utc=True, format="mixed")
            _is_live_col = df_15m["is_live"] if "is_live" in df_15m.columns \
                else pd.Series(0, index=df_15m.index)
            _live_num = pd.to_numeric(_is_live_col, errors="coerce").fillna(0)
            df_15m = df_15m.assign(_ts=_ts, _live=_live_num) \
                            .sort_values(_key_cols + ["_ts"])
            # dropna=False: "side" is blank/NaN for every "pass" decision row --
            # groupby's default drop of NaN keys would otherwise exclude ALL pass
            # rows from grouping, defeating the dedup for the far more common
            # pass-decision duplicates.
            _gap = df_15m.groupby(_key_cols, dropna=False)["_ts"].diff()
            _new_session = _gap.isna() | (_gap > pd.Timedelta(seconds=90))
            df_15m["_session"] = _new_session.cumsum()
            df_15m = df_15m.sort_values("_live", ascending=False)
            df_15m = df_15m.drop_duplicates(subset=_key_cols + ["_session"], keep="first")
            df_15m = df_15m.drop(columns=["_ts", "_live", "_session"], errors="ignore")
        df_15m = df_15m.drop(columns=["_flat"], errors="ignore")

    frames = [f for f in [df_1h, df_15m] if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.sort_values("logged_at").reset_index(drop=True)

    # For ETH/SOL: supplement with live_trades for Jun18+ to fill the paper_trades gap
    if asset in ASSET_LIVE_CSV:
        live_supp = _load_live_supplement(ASSET_LIVE_CSV[asset])
        if not live_supp.empty:
            cutoff_ts = pd.Timestamp(LIVE_SUPPLEMENT_FROM, tz="UTC").tz_convert("America/Los_Angeles")
            live_supp = live_supp[live_supp["logged_at"] >= cutoff_ts].copy()
            # Exclude tickers already recorded in paper_trades for this window (avoids double-counting dual-mode rows)
            existing_tickers = set(df[df["logged_at"] >= cutoff_ts]["contract_ticker"].dropna())
            live_supp = live_supp[~live_supp["contract_ticker"].isin(existing_tickers)]
            if not live_supp.empty:
                df = pd.concat([df, live_supp], ignore_index=True, sort=False)
                df = df.sort_values("logged_at").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _parse_win(val) -> bool:
    """Parse would_win stored as True/False, 1/0, 1.0/0.0, or their string forms."""
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "1.0")

def stat_card(col, label, value, color="#ffffff"):
    col.markdown(f"""
    <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;
                padding:20px 24px;margin-bottom:12px;">
      <div style="color:#666;font-size:0.68rem;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div style="color:{color};font-size:2rem;font-weight:800;line-height:1;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def _f(val, default=float("nan")):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fmt(val, fmt, fallback="—"):
    v = _f(val)
    return fmt.format(v) if v == v else fallback


# ---------------------------------------------------------------------------
# Per-asset dashboard renderer
# ---------------------------------------------------------------------------

INDICATORS = [
    ("composite_trend",   "Composite Trend Score (4h)"),
    ("composite_rev",     "Composite Reversion Score (1h/15m)"),
    ("confirmation_score","Confirmation Score (from composite)"),
    ("no_score",          "NO Score (from composite)"),
    ("obi_score",         "Order Book Imbalance"),
    ("funding_bias",      "Funding Rate Bias"),
    ("vol_score",         "Vol Regime Score"),
    ("vwap_score",        "VWAP Score"),
    ("ema_stretch_score", "EMA Stretch Score"),
    ("vpin_score",        "VPIN Score"),
]

TABLE_STYLE = [{"selector": "th", "props": [
    ("background-color", "#141414"), ("color", "#777"),
    ("font-size", "0.68rem"), ("letter-spacing", "0.08em"),
    ("text-transform", "uppercase"), ("border-bottom", "1px solid #333"),
]}]


def _to_numeric_ind(series):
    num = pd.to_numeric(series, errors="coerce")
    return num.apply(lambda v: 0 if pd.isna(v) else (1 if v > 0 else (-1 if v < 0 else 0))).astype(int)


def _alignment(ind_val, side):
    if ind_val == 0:
        return "Neutral"
    return "Aligned" if (ind_val > 0 and side == "yes") or (ind_val < 0 and side == "no") else "Misaligned"


def color_wr(val):
    try:
        v = float(str(val).rstrip("%")) / 100 if "%" in str(val) else float(val)
        if v >= 0.6:  return "color: #00c076; font-weight: 700"
        if v >= 0.5:  return "color: #7ec8a0"
        if v >= 0.4:  return "color: #f0a500"
        return "color: #ff4d4d"
    except Exception:
        return ""


def color_alignment(val):
    if val == "Aligned":    return "color: #00c076"
    if val == "Misaligned": return "color: #ff4d4d"
    return "color: #888"


def render_indicator_stats(resolved: pd.DataFrame):
    """Render indicator alignment win-rate tables for a resolved trades DataFrame."""
    if resolved.empty:
        st.markdown("<p style='color:#555;padding:20px 0;'>No resolved trades yet.</p>", unsafe_allow_html=True)
        return

    for col, label in INDICATORS:
        if col not in resolved.columns:
            st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:12px 0 2px 0;'>{label}</div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#555;font-size:0.75rem;padding:4px 0 8px 0;'>No data yet.</div>", unsafe_allow_html=True)
            continue

        merged = resolved.copy()
        merged["_ind"]   = _to_numeric_ind(merged[col])
        merged["_win"]   = merged["would_win_bool"].astype(int)
        merged["_side"]  = merged["side"].str.lower()
        merged["_align"] = merged.apply(lambda r: _alignment(r["_ind"], r["_side"]), axis=1)

        if merged["_ind"].eq(0).all():
            st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:12px 0 2px 0;'>{label}</div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#555;font-size:0.75rem;padding:4px 0 8px 0;'>All neutral.</div>", unsafe_allow_html=True)
            continue

        rows = []
        for side_label, side_filter in [("YES", "yes"), ("NO", "no"), ("ALL", None)]:
            sub = merged if side_filter is None else merged[merged["_side"] == side_filter]
            if sub.empty:
                continue
            for align in ["Aligned", "Neutral", "Misaligned"]:
                a = sub[sub["_align"] == align]
                if a.empty:
                    continue
                t = len(a); w = int(a["_win"].sum())
                rows.append({"Side": side_label, "Alignment": align, "Trades": t, "Wins": w, "Losses": t - w, "Win Rate": w / t})

        if not rows:
            continue

        tbl = pd.DataFrame(rows)
        styled_ind = (
            tbl.style
            .applymap(color_alignment, subset=["Alignment"])
            .applymap(color_wr,        subset=["Win Rate"])
            .format({"Win Rate": "{:.1%}"})
            .set_properties(**{"background-color": "#1a1a1a", "color": "#ddd", "border-color": "#2a2a2a"})
            .set_table_styles(TABLE_STYLE)
            .hide(axis="index")
        )
        st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:16px 0 4px 0;'>{label}</div>", unsafe_allow_html=True)
        st.dataframe(styled_ind, use_container_width=True, height=min(38 * len(tbl) + 48, 320))


def render_asset(asset: str, csv_key: str = None, spot_asset: str = None, label: str = None):
    """csv_key/spot_asset let a synthetic asset (e.g. "BTC_OLD") reuse a real
    asset's spot-price source and composite-scorer display logic while loading
    trades from its own dedicated CSV via ASSET_CSV[asset]."""
    csv_key    = csv_key or asset
    spot_asset = spot_asset or asset
    display_label = label or asset

    spot_data = fetch_spot(spot_asset)
    spot      = spot_data["avg"]
    prices    = spot_data["prices"]

    # Spot bar
    spot_cols = st.columns(len(prices) + 1)
    with spot_cols[0]:
        st.metric(f"{display_label} (avg)", f"${spot:,.2f}" if spot else "—")
    for i, (name, price) in enumerate(prices.items()):
        spot_cols[i + 1].metric(name.capitalize(), f"${price:,.2f}")

    st.markdown("<hr style='margin:14px 0 12px 0;'>", unsafe_allow_html=True)

    # Load data
    df_all = load_trades(csv_key)

    if df_all.empty:
        st.markdown("<p style='color:#666;text-align:center;padding:30px 0;'>No trades logged yet.</p>", unsafe_allow_html=True)
        return

    # Apply display cutoff — per-timeframe override takes precedence over per-asset
    def _row_cutoff(row):
        tf = row.get("timeframe", "1h")
        if tf == "15m" and asset in ASSET_DISPLAY_FROM_15M:
            return pd.Timestamp(ASSET_DISPLAY_FROM_15M[asset], tz="UTC").tz_convert("America/Los_Angeles")
        return pd.Timestamp(ASSET_DISPLAY_FROM.get(asset, DISPLAY_FROM), tz="UTC").tz_convert("America/Los_Angeles")
    cutoff_1h  = pd.Timestamp(ASSET_DISPLAY_FROM.get(asset, DISPLAY_FROM), tz="UTC").tz_convert("America/Los_Angeles")
    cutoff_15m = pd.Timestamp(ASSET_DISPLAY_FROM_15M.get(asset, ASSET_DISPLAY_FROM.get(asset, DISPLAY_FROM)), tz="UTC").tz_convert("America/Los_Angeles")
    df = df_all[
        ((df_all["timeframe"] == "15m") & (df_all["logged_at"] >= cutoff_15m)) |
        ((df_all["timeframe"] != "15m") & (df_all["logged_at"] >= cutoff_1h))
    ].copy()

    if df.empty:
        st.markdown("<p style='color:#666;text-align:center;padding:30px 0;'>No trades in display window.</p>", unsafe_allow_html=True)
        return

    trades   = df[(df["decision"] == "trade") & (df["contract_ticker"].fillna("").str.strip() != "")].copy()
    resolved = trades.dropna(subset=["would_win"]).copy()
    pending  = trades[trades["would_win"].isna()].copy()

    if not resolved.empty:
        resolved["would_win_bool"] = resolved["would_win"].apply(_parse_win)
        resolved["would_pnl_num"]  = pd.to_numeric(resolved["would_pnl"], errors="coerce")
        win_rate = resolved["would_win_bool"].mean()
        net_pnl  = resolved["would_pnl_num"].sum()
        wins     = resolved["would_win_bool"].sum()
        losses   = (~resolved["would_win_bool"]).sum()
    else:
        win_rate = net_pnl = wins = losses = 0

    pnl_color = "#00c076" if net_pnl >= 0 else "#ff4d4d"
    wr_color  = "#00c076" if win_rate >= 0.5 else "#ff4d4d"
    pnl_sign  = "+" if net_pnl >= 0 else ""

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)
    stat_card(r1c1, "Total Trades",  len(trades),              "#4da6ff")
    stat_card(r1c2, "Pending",       len(pending),              "#f0a500")
    stat_card(r1c3, "Resolved",      len(resolved),             "#ffffff")
    stat_card(r2c1, "Win Rate",      f"{win_rate:.1%}" if resolved.shape[0] else "—", wr_color)
    stat_card(r2c2, "Total P&L",     f"${pnl_sign}{net_pnl:,.2f}" if resolved.shape[0] else "—", pnl_color)
    stat_card(r2c3, "Wins / Losses", f"{int(wins)} / {int(losses)}" if resolved.shape[0] else "—", "#ffffff")

    st.markdown("<hr style='margin:6px 0 14px 0;'>", unsafe_allow_html=True)

    # Latest signal
    latest = df.iloc[-1]
    dec    = str(latest.get("decision", "")).upper()

    st.markdown("<div style='font-size:0.72rem;color:#666;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;'>Latest Signal</div>", unsafe_allow_html=True)
    ls1, ls2, ls3, ls4, ls5, ls6 = st.columns(6)
    ls1.metric("Decision",  dec)
    ls2.metric("Side",      str(latest.get("side", "—")).upper())
    ls3.metric("p_model",   _fmt(latest.get("p_yes_model"), "{:.3f}"))
    ls4.metric("p_market",  _fmt(latest.get("p_market"),    "{:.3f}"))
    ls5.metric("Net Edge",  _fmt(latest.get("net_edge"),    "{:+.3f}"))
    ls6.metric("Bet",       _fmt(latest.get("bet_amount"),  "${:,.0f}"))

    # Composite scorer row
    if spot_asset in ("BTC", "ETH", "SOL") and "composite_trend" in df.columns:
        cs1, cs2, cs3, cs4 = st.columns(4)
        comp_trend = latest.get("composite_trend", "—")
        comp_rev   = latest.get("composite_rev",   "—")
        comp_p_up  = latest.get("composite_p_up",  "—")
        cs1.metric("Composite Trend",  _fmt(comp_trend, "{:+.0f}"))
        cs2.metric("Composite Rev",    _fmt(comp_rev,   "{:+.0f}"))
        cs3.metric("Composite p_up",   _fmt(comp_p_up,  "{:.1%}"))
        _p_up_v = _f(comp_p_up)
        _asset_base = {"BTC": 0.504, "ETH": 0.509, "SOL": 0.500}.get(spot_asset, 0.504)
        _edge_vs_base = _p_up_v - _asset_base if _p_up_v == _p_up_v else float("nan")
        cs4.metric("vs Baseline",      f"{_edge_vs_base:+.1%}" if _edge_vs_base == _edge_vs_base else "—")

    # Price / contract context row
    lb1, lb2, lb3, lb4, lb5 = st.columns(5)
    lb1.metric("Spot",              _fmt(latest.get("spot"),   "${:,.2f}"))
    lb2.metric("Strike",            _fmt(latest.get("strike"), "${:,.2f}"))
    lb3.metric("Offset %",          _fmt(latest.get("offset_pct"), "{:+.3f}%"))
    lb4.metric("Vol Eff",           _fmt(latest.get("vol_eff"),    "{:.5f}"))
    _cs = _f(latest.get("contracts_scanned", 0), 0)
    lb5.metric("Contracts Scanned", int(_cs) if _cs == _cs else 0)

    # Sharp move / gate row
    _chg30 = _f(latest.get("chg_30m", float("nan")))
    _sharp_active = str(latest.get("sharp_move_active", "")).strip().lower() in ("true", "1")
    _sharp_label  = "ACTIVE" if _sharp_active else "off"
    _sharp_color  = "#f0a500" if _sharp_active else "#555"
    _gate_blocked = str(latest.get("gate_blocked", "")).strip() or "—"
    _contract_label = latest.get("contract_ticker", "—") or "—"
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("30m Price Chg",  f"{_chg30:+.3f}%" if _chg30 == _chg30 else "—")
    sm2.markdown(f"""
    <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;
                padding:20px 24px;margin-bottom:12px;">
      <div style="color:#666;font-size:0.68rem;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:8px;">Sharp Inversion</div>
      <div style="color:{_sharp_color};font-size:2rem;font-weight:800;line-height:1;">{_sharp_label}</div>
    </div>
    """, unsafe_allow_html=True)
    sm3.metric("Gate Blocked",  _gate_blocked)
    sm4.metric("Contract",      _contract_label)

    st.markdown("<hr style='margin:14px 0 14px 0;'>", unsafe_allow_html=True)

    # Inner tabs
    t1, t2, t3 = st.tabs(["Trades", "Equity Curve", "Indicator Stats"])

    # ── Equity Curve ────────────────────────────────────────────────────────
    with t2:
        if resolved.empty:
            st.markdown("<p style='color:#555;padding:20px 0;'>No resolved trades yet.</p>", unsafe_allow_html=True)
        else:
            def _equity_fig(data: pd.DataFrame, color: str, fillcolor: str, label: str = "", height: int = 260) -> go.Figure:
                d = data.sort_values("logged_at").copy()
                d["cum_pnl"] = d["would_pnl_num"].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=d["logged_at"], y=d["cum_pnl"],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    fill="tozeroy", fillcolor=fillcolor,
                    hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="#444", opacity=0.8)
                annotations = []
                if label:
                    annotations.append(dict(
                        text=label,
                        xref="paper", yref="paper",
                        x=0.01, y=0.97,
                        xanchor="left", yanchor="top",
                        font=dict(size=13, color=color, family="monospace"),
                        bgcolor="rgba(0,0,0,0.45)",
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=4,
                        showarrow=False,
                    ))
                fig.update_layout(
                    xaxis_title=None, yaxis_title="Cumul. P&L ($)", height=height,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#0e0e0e", paper_bgcolor="#0e0e0e",
                    font=dict(color="#888"),
                    xaxis=dict(gridcolor="#1f1f1f", zeroline=False),
                    yaxis=dict(gridcolor="#1f1f1f", zeroline=False),
                    showlegend=False,
                    annotations=annotations,
                )
                return fig

            # Combined
            st.plotly_chart(_equity_fig(resolved, "#00c076", "rgba(0,192,118,0.08)", label="ALL  1h + 15m", height=300), use_container_width=True)

            # Per-timeframe
            ec1, ec2 = st.columns(2)
            for col_ui, tf, color, fillc in [
                (ec1, "1h",  "#4da6ff", "rgba(77,166,255,0.08)"),
                (ec2, "15m", "#f0a500", "rgba(240,165,0,0.08)"),
            ]:
                with col_ui:
                    if "timeframe" not in resolved.columns:
                        st.markdown(f"<p style='color:#555;font-size:0.8rem;'>{tf.upper()} — no timeframe data</p>", unsafe_allow_html=True)
                        continue
                    sub = resolved[resolved["timeframe"] == tf]
                    if sub.empty:
                        st.markdown(f"<p style='color:#555;font-size:0.8rem;'>{tf.upper()} — no resolved trades yet</p>", unsafe_allow_html=True)
                        continue
                    wr   = sub["would_win_bool"].mean()
                    pnl  = sub["would_pnl_num"].sum()
                    wr_c  = "#00c076" if wr  >= 0.5 else "#ff4d4d"
                    pnl_c = "#00c076" if pnl >= 0   else "#ff4d4d"
                    chart_label = (
                        f"{tf.upper()}  "
                        f"{wr:.0%} WR  "
                        f"{'+'if pnl>=0 else ''}${pnl:,.0f}"
                    )
                    st.plotly_chart(_equity_fig(sub, color, fillc, label=chart_label, height=220), use_container_width=True)

    # ── Trade log ───────────────────────────────────────────────────────────
    with t1:
        _composite_only = st.checkbox(
            "Composite drift hybrid only",
            value=True,
            key=f"composite_only_{asset}",
            help="Show only trades using the composite drift hybrid with vol drift model (composite_p_up populated)",
        )

        display_cols = [
            "logged_at", "timeframe", "contract_ticker", "side",
            "offset_pct", "spot", "strike", "tau_minutes",
            "p_yes_model", "p_market", "net_edge",
            "would_pnl", "would_win",
            "composite_trend", "composite_rev", "composite_p_up", "p_up_v2",
            "chg_5m", "chg_10m", "chg_30m", "sharp_move_active",
            "confirmation_score", "no_score",
            "obi_score", "funding_bias", "vol_score", "vwap_score", "ema_stretch_score",
            "vol_eff",
            "kelly_fraction", "bet_amount",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        _trade_source = trades.copy()
        if _composite_only and "composite_p_up" in _trade_source.columns:
            _pup = pd.to_numeric(_trade_source["composite_p_up"], errors="coerce")
            _trade_source = _trade_source[(_pup != 0) | _pup.isna()]
        trade_rows = _trade_source[display_cols].copy().sort_values("logged_at", ascending=False)

        def fmt_result(row):
            v = row.get("would_win", "")
            if pd.isna(v) or str(v).strip() == "": return "pending"
            return "WIN" if _parse_win(v) else "LOSS"

        trade_rows["result"] = trade_rows.apply(fmt_result, axis=1)
        trade_rows = trade_rows.drop(columns=["would_win"], errors="ignore")

        for _nc in ["offset_pct", "spot", "strike", "tau_minutes", "p_yes_model", "p_market",
                    "net_edge", "bet_amount", "would_pnl", "kelly_fraction",
                    "composite_trend", "composite_rev", "composite_p_up", "p_up_v2",
                    "chg_5m", "chg_10m", "chg_30m", "confirmation_score", "no_score",
                    "obi_score", "funding_bias", "vol_score", "vwap_score", "ema_stretch_score", "vol_eff"]:
            if _nc in trade_rows.columns:
                trade_rows[_nc] = pd.to_numeric(trade_rows[_nc], errors="coerce")
        # sharp_move_active: convert bool string to readable label
        if "sharp_move_active" in trade_rows.columns:
            trade_rows["sharp_move_active"] = trade_rows["sharp_move_active"].astype(str).str.lower().map(
                {"true": "INVERTED", "false": "—", "1": "INVERTED", "0": "—"}
            ).fillna("—")

        trade_rows = trade_rows.rename(columns={
            "logged_at":          "Time (PT)",
            "timeframe":          "TF",
            "contract_ticker":    "Contract",
            "side":               "Side",
            "offset_pct":         "Offset%",
            "spot":               "Spot",
            "strike":             "Strike",
            "tau_minutes":        "τ (min)",
            "p_yes_model":        "p_model",
            "p_market":           "p_market",
            "net_edge":           "Net Edge",
            "composite_trend":    "Trend",
            "composite_rev":      "Rev",
            "composite_p_up":     "p_up (old)",
            "p_up_v2":            "p_up v2",
            "chg_5m":             "5m Chg%",
            "chg_10m":            "10m Chg%",
            "chg_30m":            "30m Chg%",
            "sharp_move_active":  "Inversion",
            "confirmation_score": "Conf",
            "no_score":           "NO Score",
            "obi_score":          "OBI",
            "funding_bias":       "Funding",
            "vol_score":          "Vol",
            "vwap_score":         "VWAP",
            "ema_stretch_score":  "EMA Str",
            "vol_eff":            "Vol Eff",
            "kelly_fraction":     "Kelly",
            "bet_amount":         "Bet ($)",
            "would_pnl":         "P&L ($)",
            "result":            "Result",
        })

        # Enforce column order — computed columns (Result) append to end by default
        _ordered = [c for c in [
            "Time (PT)", "TF", "Contract", "Side",
            "Offset%", "Spot", "Strike", "τ (min)",
            "p_model", "p_market", "Net Edge",
            "P&L ($)", "Result", "p_up v2",
            "Trend", "Rev", "p_up (old)",
            "5m Chg%", "10m Chg%", "30m Chg%", "Inversion",
            "Conf", "NO Score", "OBI", "Funding", "Vol", "VWAP", "EMA Str",
            "Vol Eff", "Kelly", "Bet ($)",
        ] if c in trade_rows.columns]
        trade_rows = trade_rows[_ordered]

        def color_result(val):
            if val == "WIN":     return "color: #00c076; font-weight: 700"
            if val == "LOSS":    return "color: #ff4d4d; font-weight: 700"
            if val == "pending": return "color: #f0a500; font-weight: 500"
            return ""

        def color_pnl(val):
            try:
                v = float(val)
                return "color: #00c076" if v > 0 else ("color: #ff4d4d" if v < 0 else "")
            except Exception:
                return ""

        def color_edge(val):
            try:
                v = float(val)
                return "color: #00c076" if v > 0.1 else ("color: #f0a500" if v > 0.05 else "color: #ff4d4d")
            except Exception:
                return ""

        def color_score(val):
            try:
                v = int(float(val))
                if v > 0: return "color: #00c076"
                if v < 0: return "color: #ff4d4d"
            except Exception:
                pass
            return "color: #888"

        score_cols = [c for c in ["Trend", "Rev", "Conf", "NO Score", "OBI", "Funding", "Vol", "VWAP", "EMA Str"] if c in trade_rows.columns]

        styled = (
            trade_rows.style
            .applymap(color_result, subset=["Result"])
            .applymap(color_pnl,    subset=["P&L ($)"])
            .applymap(color_edge,   subset=["Net Edge"])
            .applymap(color_score,  subset=score_cols)
            .format({
                "Spot":     "${:,.2f}",
                "Strike":   "${:,.2f}",
                "Offset%":  "{:+.3f}%",
                "τ (min)":  "{:.0f}",
                "p_model":  "{:.3f}",
                "p_market": "{:.3f}",
                "Net Edge": "{:+.3f}",
                "Trend":    "{:+.0f}",
                "Rev":      "{:+.0f}",
                "p_up":     "{:.1%}",
                "5m Chg%":  "{:+.3f}%",
                "10m Chg%": "{:+.3f}%",
                "30m Chg%": "{:+.3f}%",
                "Vol Eff":  "{:.5f}",
                "Kelly":    "{:.4f}",
                "Bet ($)":  "${:,.0f}",
                "P&L ($)":  "${:,.2f}",
            }, na_rep="—")
            .set_properties(**{"background-color": "#1a1a1a", "color": "#ddd", "border-color": "#2a2a2a"})
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#141414"), ("color", "#777"),
                    ("font-size", "0.68rem"), ("letter-spacing", "0.08em"),
                    ("text-transform", "uppercase"), ("border-bottom", "1px solid #333"),
                ]},
                {"selector": "tr:hover td", "props": [("background-color", "#222")]},
            ])
        )
        _col_cfg = {
            "Time (PT)":  st.column_config.Column(width=140),
            "Contract":   st.column_config.Column(width=210),
            "Side":       st.column_config.Column(width=48),
            "Offset%":    st.column_config.Column(width=75),
            "Spot":       st.column_config.Column(width=90),
            "Strike":     st.column_config.Column(width=90),
            "τ (min)":    st.column_config.Column(width=65),
            "p_model":    st.column_config.Column(width=62),
            "p_market":   st.column_config.Column(width=62),
            "Net Edge":   st.column_config.Column(width=68),
            "P&L ($)":    st.column_config.Column(width=75),
            "Result":     st.column_config.Column(width=70),
            "Trend":      st.column_config.Column(width=55),
            "Rev":        st.column_config.Column(width=50),
            "p_up":       st.column_config.Column(width=60),
            "5m Chg%":    st.column_config.Column(width=75),
            "10m Chg%":   st.column_config.Column(width=80),
            "30m Chg%":   st.column_config.Column(width=80),
            "Inversion":  st.column_config.Column(width=80),
            "Conf":       st.column_config.Column(width=55),
            "NO Score":   st.column_config.Column(width=75),
            "OBI":        st.column_config.Column(width=50),
            "Funding":    st.column_config.Column(width=65),
            "Vol":        st.column_config.Column(width=50),
            "VWAP":       st.column_config.Column(width=55),
            "EMA Str":    st.column_config.Column(width=65),
            "Vol Eff":    st.column_config.Column(width=75),
            "Kelly":      st.column_config.Column(width=65),
            "Bet ($)":    st.column_config.Column(width=65),
        }
        st.dataframe(styled, use_container_width=True, height=450, column_config=_col_cfg)

    # ── Indicator Stats ──────────────────────────────────────────────────────
    with t3:
        st.markdown(
            "<div style='color:#888;font-size:0.78rem;margin-bottom:16px;'>"
            "Win rate by indicator alignment with trade direction. "
            "<b style='color:#aaa;'>Aligned</b> = indicator direction matches trade side. "
            "<b style='color:#aaa;'>Misaligned</b> = indicator opposes trade direction."
            "</div>",
            unsafe_allow_html=True,
        )
        render_indicator_stats(resolved)

    st.markdown("<hr style='margin:18px 0 18px 0;'>", unsafe_allow_html=True)

    # Signal breakdown expander
    with st.expander("Signal Breakdown"):
        fb1, fb2, fb3 = st.columns(3)
        fund_counts = df["funding_bias"].value_counts() if "funding_bias" in df.columns else {}
        fb1.metric("Bullish Funding",  int(fund_counts.get(1,  0)))
        fb2.metric("Neutral Funding",  int(fund_counts.get(0,  0)))
        fb3.metric("Bearish Funding",  int(fund_counts.get(-1, 0)))

        ob1, ob2, ob3 = st.columns(3)
        obi_counts = df["obi_score"].value_counts() if "obi_score" in df.columns else {}
        ob1.metric("Bullish OBI",  int(obi_counts.get(1,  0)))
        ob2.metric("Neutral OBI",  int(obi_counts.get(0,  0)))
        ob3.metric("Bearish OBI",  int(obi_counts.get(-1, 0)))

        sm1_b, sm2_b = st.columns(2)
        _sharp_total = int(df["sharp_move_active"].astype(str).str.lower().isin(["true","1"]).sum()) if "sharp_move_active" in df.columns else 0
        sm1_b.metric("Cycles w/ Sharp Inversion", _sharp_total)
        sm2_b.metric("Normal Cycles",             max(0, len(df) - _sharp_total))


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

now_str = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S PT")

st.markdown(f"""
<div style="margin-bottom:8px;">
  <span style="font-size:1.8rem;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">Kalshi Trader</span>
  &nbsp;
  <span style="background:#c97a00;color:#fff;font-size:0.72rem;font-weight:700;
               padding:3px 10px;border-radius:20px;letter-spacing:0.1em;">BTC+SOL LIVE</span>
</div>
<div style="color:#555;font-size:0.78rem;margin-bottom:4px;">
  BTC · ETH · SOL Event Contracts &nbsp;|&nbsp; Live Signal Engine &nbsp;|&nbsp;
  Updated: {now_str} &nbsp;|&nbsp; live update on data change (no page reload)
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:12px 0 16px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Outer asset tabs
# ---------------------------------------------------------------------------

(tab_btc, tab_eth, tab_sol, tab_sol_shadow, tab_15m_shadow,
 tab_sol_hourly_ab, tab_cmp) = st.tabs(
    ["₿  BTC", "Ξ  ETH", "◎  SOL", "👥  SOL SHADOW A/B", "🧪  BTC/ETH 15M A/B",
     "🥊  HOURLY A/B", "📊  Compare"]
)

with tab_btc:
    render_asset("BTC")

with tab_eth:
    render_asset("ETH")

with tab_sol:
    render_asset("SOL")

with tab_sol_shadow:
    # [2026-07-30] Replaced the BTC (OLD MODEL TEST) tab per user request —
    # that experiment ended 07-28 (runner stopped; verdict in memory; its CSV
    # is preserved untouched). This tab shows the SOL slope-shadow candidate
    # (p_gbdt column since 2026-07-30 01:00 UTC) vs the production model
    # (p_model_15m) as hypothetical flat-$100 books on identical live scans.
    # Promotion decision at the 08-11 review.
    st.markdown(
        "<div style='color:#f0a500;font-size:0.78rem;margin-bottom:8px;'>"
        "SOL 15m model A/B on identical live scans since 2026-07-30 01:00 UTC. "
        "[2026-08-21 PER-MODEL REWORK] Rows are now MODELS, not columns: "
        "blue/p_sol_old = the iso+z-expansion model (★DECIDING since the "
        "08-21 re-swap; also decided pre-08-12), orange/p_sol_slope = the "
        "slope model (decided 08-12→08-21). These streams NEVER change "
        "meaning across seat swaps — totals are one model each. Books "
        "remain hypothetical replays on identical scans."
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Book definitions (3 variants per model)"):
        st.markdown(
            "- **flat $100** (solid): raw model — fee-adjusted edge ≥ 0.04, one "
            "bet per contract, $100 flat, net of fees. No gates, no sizing.\n"
            "- **gated+kelly** (dashed): + v2 band/persistence gates, the live "
            "regime gates (sol_markov + zdrift <0.55 NO-block, 08-03) and the "
            "YES offset gate (08-04 — the one extra YES gate that tested "
            "non-redundant), fee-aware Kelly on $2,500 (frac cap 10%). The "
            "08-11 promotion metric (daily Sharpe, ties by maxDD) is scored "
            "on these books.\n"
            "- **gk zd65** (dotted): same stack with the zdrift NO-block "
            "widened to 0.65 — monitoring only, NOT in the live runner. "
            "Threshold surfaced by the 08-04/05 NO-side dip, so its 08-11 "
            "read scores 08-05+ trades only.\n"
            "- **gk combo+damp** (dotted, thin): vrM+zd65 + the live "
            "drawdown-from-peak Kelly dampener (frozen 10d/z2/×0.5). "
            "Multiplier inert until the window has 15 days (~08-14) — "
            "identical to vrM+zd65 until then; divergence is pure forward "
            "evidence of the layer. (gk vrM+hurst retired 08-08: worst "
            "performance and DD of the field.)\n"
            "- **gk zd65+path** (long-dash-dot): zd65 + block NO when the "
            "live-logged pm-path signal opposes it (pm_path_drift × "
            "pm_path_vr3 > 0 — book momentum pushing up against the NO). "
            "History before 08-11 03:55 uses the candle-archive backfill "
            "(sign rule, nothing fitted); from then on, live-logged "
            "pm_path columns. Backfill evidence: helps the SHADOW-model "
            "books (−$445 A/B window), hurts production's (+$1,201) — "
            "the per-book table is the referee; 08-18 read scores "
            "live-fired trades separately.\n"
            "- **gk vrM+zd65** (long-dash-dot): both modifications combined — "
            "the strongest replay (S 0.57/0.59/0.29, lowest DD) and the third "
            "candidate stack racing forward. Same 08-05+ scoring rule.\n"
            "- *(gk hurst retired 08-11: its differentiating signal's "
            "weekly effect on SOL proved noise — 5 sign flips in 11 weeks, "
            "mean ≈ 0 — and its forward record was last of field. hurst "
            "lives on in the ETH gate, the SOL persist score, and the "
            "pm-path lineage. Revival only via a slow sign-conditioner at "
            "the ~08-25 regime lens.)*")
    _SH_START = pd.Timestamp("2026-07-30 01:00", tz="UTC")
    try:
        _shp = pd.read_csv(ASSET_CSV_15M["SOL"], low_memory=False)
        _shp["dt"] = pd.to_datetime(_shp["logged_at"], errors="coerce", utc=True, format="mixed")
        for _c in ["p_market", "p_model_15m", "p_gbdt", "p_sol_old",
                   "p_sol_slope", "resolved_yes",
                   "sol_persist_score", "slope120_stoch_k_15m",
                   "stoch_cross_1h", "stoch_k_1h", "oi_chg_pct",
                   "offset_pct", "z_drift_6h", "vol_ratio_1h",
                   "hurst_exponent_5m", "pm_path_drift", "pm_path_vr3"]:
            _shp[_c] = pd.to_numeric(_shp[_c], errors="coerce")
        # [2026-08-11] pm-path features: live-logged columns (since 08-11
        # 03:55) take precedence; earlier rows filled from the candle-archive
        # backfill (results/sol15m_pmpath_backfill.csv — sign rule is unfitted
        # so retro rendering is legitimate; the 08-18 read still scores
        # live-fired trades separately from reconstructed ones).
        try:
            _bf = pd.read_csv(RESULTS_DIR / "sol15m_pmpath_backfill.csv")
            _shp = _shp.merge(_bf, on=["logged_at", "contract_ticker"],
                              how="left")
            for _pc, _bc in [("pm_path_drift", "pm_path_drift_bf"),
                             ("pm_path_vr3", "pm_path_vr3_bf")]:
                _shp[_pc] = pd.to_numeric(_shp[_pc], errors="coerce").fillna(
                    pd.to_numeric(_shp[_bc], errors="coerce"))
        except Exception:
            pass
        _sh = _shp[(_shp["dt"] >= _SH_START)
                   & _shp["resolved_yes"].notna()
                   & _shp["p_market"].between(0.03, 0.97)].copy()
        if len(_sh) < 3:
            st.info(f"Collecting… {len(_sh)} resolved scans since shadow go-live "
                    f"(unresolved scans settle within ~15 min of expiry).")
        else:
            # [2026-08-05] tab decluttered: line-family selector (default =
            # the decision-relevant gated books) + summary TABLE below the
            # chart instead of metric cards (whose delta text truncated).
            # [2026-08-03] fixed 50/50 blend tracked as a third book — weights
            # are NEVER fitted (5-day window = noise); pre-registered candidate
            # for the 08-11 review only if it leads or matches with lower DD.
            _sh["p_blend"] = (_sh["p_model_15m"] + _sh["p_gbdt"]) / 2
            # [2026-08-10] chart defaults = TOP-2 gated variants by combined
            # net across the three books (user call — static default showed
            # gated+kelly even when it lagged). Traces are collected first,
            # ranked, then the selector renders with the dynamic default.
            _figsh = go.Figure()
            _rows = []
            _era_rows = []
            _traces = []
            _famdaily = {}

            def _kbook(_qq, _col):
                """Kelly-sized book: pnl series + maxDD ($2500 flat, cap 10%)."""
                _c = np.where(_qq["side"] == "yes", _qq["p_market"],
                              1 - _qq["p_market"])
                _w = np.where(_qq["side"] == "yes", _qq["resolved_yes"] == 1,
                              _qq["resolved_yes"] == 0)
                _f = 0.07 * _qq["p_market"] * (1 - _qq["p_market"])
                _fr = np.where(_qq["side"] == "yes",
                               (_qq[_col] - _qq["p_market"] - _f)
                               / (1 - _qq["p_market"]),
                               (_qq["p_market"] - _qq[_col] - _f)
                               / _qq["p_market"])
                _stk = 2500.0 * np.clip(_fr, 0, 0.10)
                _p = pd.Series(np.where(_w, _stk * (1 - _c) / _c, -_stk)
                               - (_stk / _c) * _f, index=_qq.index)
                _cum = _p.cumsum()
                _dd = float((_cum.cummax() - _cum).max()) if len(_p) else 0.0
                return _p, _dd
            # [2026-08-11] blend book RETIRED: its pre-registered read
            # concluded today — failed every promotion metric (net -$1,721,
            # S -0.20, worst DD; prob-averaging = disagreement-dampening,
            # banked). gated+kelly base variant retired same day: dead
            # weight as a racer; its live-equivalent role is recoverable
            # (zd65 minus the 0.55->0.65 widen).
            # [2026-08-12] labels renamed after the promotion role swap:
            # "deciding" = whatever model drives the paper book at each
            # moment (p_model_15m — old model pre-18:00, slope after);
            # "companion" = the non-deciding comparison stream (p_gbdt).
            # Truthful across the swap boundary, unlike production/shadow.
            # [2026-08-21 RE-SWAP ~04:15 UTC, user call] OLD model decides
            # again (slope's post-promotion record lost to it on every
            # variant — era-split verified). Column-model map now:
            # p_model_15m = old / slope / old across the two boundaries;
            # p_gbdt the mirror. The era_split dump splits only at the
            # 08-12 boundary — reads after 08-21 must also split there.
            # [2026-08-21 PER-MODEL REWORK, user call] rows were COLUMN
            # streams (mixed models across seat swaps — three reads burned
            # on the chimera). Now each row is ONE model, permanently:
            # p_sol_old / p_sol_slope never change meaning. The ★DECIDING
            # tag marks whichever currently drives the paper book.
            for _lbl, _col, _clr in [
                    ("old model ★DECIDING", "p_sol_old", "#4f8bf9"),
                    ("slope model", "p_sol_slope", "#f0a500")]:
                _s = _sh.dropna(subset=[_col]).copy()
                _fee = 0.07 * _s["p_market"] * (1 - _s["p_market"])
                _ey = _s[_col] - _s["p_market"] - _fee
                _en = _s["p_market"] - _s[_col] - _fee
                _s["side"] = np.where(_ey >= _en, "yes", "no")
                _s["edge"] = np.maximum(_ey, _en)
                _q = _s[_s["edge"] >= 0.04].sort_values("dt").drop_duplicates(
                    "contract_ticker", keep="first")
                # [2026-08-20] cheap-ticket block mirror (runner gate in
                # _replica_decide, live ~05:30 UTC): both sides at cost
                # <= 0.20 are blocked-consumed. Applied AFTER keep-first
                # dedup (matches blocked-consumption semantics) and
                # dt-gated so displayed history is not retroactively
                # flattered.
                _tkc = np.where(_q["side"] == "yes", _q["p_market"],
                                1 - _q["p_market"])
                _q = _q[~((_tkc <= 0.20)
                          & (_q["dt"] >= pd.Timestamp("2026-08-20 05:30",
                                                      tz="UTC")))]
                _cost = np.where(_q["side"] == "yes", _q["p_market"], 1 - _q["p_market"])
                _win = np.where(_q["side"] == "yes", _q["resolved_yes"] == 1,
                                _q["resolved_yes"] == 0)
                _feeq = 0.07 * _q["p_market"] * (1 - _q["p_market"])
                _pnl = pd.Series(np.where(_win, 100 * (1 - _cost) / _cost, -100)
                                 - (100 / _cost) * _feeq, index=_q.index)
                _traces.append(("flat $100", _q["dt"], _pnl.cumsum(),
                                f"{_lbl} flat", dict(color=_clr, width=2)))
                # [2026-07-31] gated+kelly variant: v2-package gates (YES needs
                # persist>=3; NO blocked pm>0.8 and pm .5-.65 w/o stoch rescue)
                # + fee-aware Kelly stake on flat $2500 bankroll, frac cap 10%.
                # [2026-08-03] + the live REGIME gates (exact port from the
                # runner): sol_markov_gate (block-unless-rescued, 6h/4h/1h
                # states, stoch/oi/offset rescues) and sol_15m_no_zdrift_gate
                # (block NO when z_drift_6h<0.55) — the protections that kept
                # the real book out of the 07-31/08-01 regime flip.
                _m6 = _q["markov_sol_6h"].astype(str)
                _m4 = _q["markov_sol_4h"].astype(str)
                _m1 = _q["markov_sol_1h"].astype(str)
                _sc1 = pd.to_numeric(_q["stoch_cross_1h"], errors="coerce").fillna(0.0)
                _sk1 = pd.to_numeric(_q["stoch_k_1h"], errors="coerce").fillna(50.0)
                _oiq = pd.to_numeric(_q["oi_chg_pct"], errors="coerce").fillna(0.0)
                _off = pd.to_numeric(_q["offset_pct"], errors="coerce").fillna(0.0)
                _zd6 = pd.to_numeric(_q["z_drift_6h"], errors="coerce")
                _gy = (((_m6 == "Bull") & (_sc1 != 0)) | (_m4 == "Sideways")
                       | ((_m1 == "Sideways") & (_oiq < 0.0535)))
                _ry = (((_m6 == "Bull") & (_sc1 == 0))
                       | ((_m1 == "Sideways") & (_oiq >= 0.0535)))
                _gn = (((_m6 == "Bull") & (_off > -0.006))
                       | ((_m4 == "Sideways") & (_sk1 < 90.0)))
                _rn = (((_m6 == "Bull") & (_off <= -0.006))
                       | ((_m4 == "Sideways") & (_sk1 >= 90.0)))
                _mkv_ok = np.where(_q["side"] == "yes",
                                   ~(_gy & ~_ry), ~(_gn & ~_rn))
                _zd_ok = np.where(_q["side"] == "no",
                                  ~(_zd6 < 0.55).fillna(False), True)
                # [2026-08-05] MONITORING variant: zdrift NO-block widened to
                # 0.65 — surfaced by the 08-04/05 dip (losers sat at zd
                # 0.58-0.63, just above 0.55; m4=Bull block tested as pure
                # dip-fit and was rejected). Threshold was read off the dip
                # losers, so trades BEFORE 08-05 are in-sample for it; the
                # 08-11 read scores 08-05+ only. NOT in the live runner.
                _zd_ok65 = np.where(_q["side"] == "no",
                                    ~(_zd6 < 0.65).fillna(False), True)
                # [2026-08-04] + sol_15m_yes_offset_gate ONLY (of the real
                # chain's 4 extra YES gates): marginal-contribution test
                # showed it alone captures the full benefit (+$2,859 full-
                # history production, = ALL-4 combined; other 3 redundant
                # within this stack, one even net-negative for shadow).
                # Blocks barely-OTM YES (offset in [-10%,0)) unless the
                # validated flip-chain rescue holds (1h=Sideways +
                # oi>=0.0535 + z_drift<0.55). Exact runner port.
                _flip_rescue = ((_m1 == "Sideways") & (_oiq >= 0.0535)
                                & (_zd6 < 0.55).fillna(False))
                _off_ok = np.where(_q["side"] == "yes",
                                   ~((_off >= -10.0) & (_off < 0.0)
                                     & ~_flip_rescue), True)
                _v2_ok = np.where(_q["side"] == "yes",
                                  _q["sol_persist_score"] >= 3,
                                  ~((_q["p_market"] > 0.8) |
                                    (_q["p_market"].between(0.5, 0.65)
                                     & ~(_q["slope120_stoch_k_15m"] >= 40))))
                # [2026-08-20 BUG FIX — user caught via
                # KXSOL15M-26AUG202015-15 (+$507 winner missing from the
                # gated lines)] RESC-2/RESC-5 were wired into the RUNNER
                # 08-18 04:27 UTC (4e3df45) but never ported here — every
                # rescued YES since then was absent from the gated books.
                # Exact port of RESC-2 (zd<0.59 rescues persist<3 YES;
                # rescued trades BYPASS the markov/offset gates),
                # dt-gated at the wire time. DISCLOSED GAP: RESC-5
                # (z_spot_6h_live<-1 & d45_vwap>=0.07) is NOT replayable —
                # z_spot_6h_live was never logged (runner logging fix
                # deployed 08-21; RESC-5 replay can be added once the
                # column accrues).
                # [2026-08-23 RESC-2 DIP TIGHTENING mirror] runner now
                # requires stoch_k_5m<=30 for the rescue (the 08-22 chop
                # run -$1,972 flowed 100% through non-dip rescues).
                # Paper-line replay is dt-gated at the deploy (~04:20 UTC)
                # so displayed history stays what the book traded; the
                # ‹mon› dipRESC variant below shows the RETRO
                # counterfactual (user request).
                _v2_base, _off_base, _mkv_base = _v2_ok, _off_ok, _mkv_ok
                _sk5q2 = pd.to_numeric(_q.get("stoch_k_5m"),
                                       errors="coerce")
                _dipq = (_sk5q2 <= 30).fillna(False)
                _resc_core = np.asarray(
                    (_q["side"] == "yes")
                    & ~(_q["sol_persist_score"] >= 3).fillna(False)
                    & (_zd6 < 0.59).fillna(False)
                    & (_q["dt"] >= pd.Timestamp("2026-08-18 04:27",
                                                tz="UTC")), bool)
                _rescY = _resc_core & np.asarray(
                    (_q["dt"] < pd.Timestamp("2026-08-23 04:20", tz="UTC"))
                    | _dipq, bool)
                _v2_ok = _v2_base | _rescY
                _off_ok = _off_base | _rescY
                _mkv_ok = _mkv_base | _rescY
                _rescY_R = _resc_core & np.asarray(_dipq, bool)
                _v2R = _v2_base | _rescY_R
                _offR = _off_base | _rescY_R
                _mkvR = _mkv_base | _rescY_R
                # [2026-08-05] regime-CONDITIONAL markov (gk vrM): apply the
                # markov gate only when vol_ratio_1h < 1 (compressed/trending
                # — where its blocks tested protective in every book/window);
                # skip it in chop (vr>=1), where its blocks were winners
                # (+$9,258 long-window blocked PnL). vr threshold 1.0 is the
                # natural boundary, NOT swept. Monitoring only; 08-11 read
                # scores 08-05+ trades only (hypothesis came from watching
                # the 08-01 flip). Combo = vrM + the zd65 NO-block widen.
                _vr1 = pd.to_numeric(_q["vol_ratio_1h"], errors="coerce")
                _mkv_vr = np.where((_vr1 >= 1.0).fillna(False), True, _mkv_ok)
                # [2026-08-05 pm] gk hurst: base stack + block NO when
                # hurst_exponent_5m >= 0.6131 (trending micro-dynamics kill
                # the mean-reversion NO book). Survived the full 162-column
                # sweep AND the ex-drawdown test (-$3,568 removed over the 4
                # weeks BEFORE 08-04, all weeks negative; long-window base
                # stack S 0.14->0.48, DD halved). Redundant ON TOP of
                # vrM+zd65 (slightly negative marginal) — so it races AS A
                # COMPETING STACK, not stacked. Threshold from the sweep
                # grid => in-sample through 08-05 15:20; forward read scores
                # trades after that. m4Bull NO-block: TRIPLE-rejected.
                _hu5 = pd.to_numeric(_q["hurst_exponent_5m"], errors="coerce")
                _hu_ok = np.where(_q["side"] == "no",
                                  ~(_hu5 >= 0.6131).fillna(False), True)
                _r = {"book": _lbl,
                      "flat": f"${_pnl.sum():+,.0f} (n={len(_q)})",
                      "WR/BE": (f"{np.mean(_win):.0%}/{np.mean(_cost):.0%}"
                                if len(_q) else "—")}
                for _vn, _vmask, _vdash, _vw in [
                        ("gk zd65", _v2_ok & _mkv_ok & _zd_ok65 & _off_ok,
                         "dot", 1.0),

                        # [2026-08-18 READ #1] vrM+zd65 & combo+damp RETIRED
                        # (vrM collapsed: S 0.03 / DD $5,243 continuous).
                        ("gk zd65+path", _v2_ok & _mkv_ok & _zd_ok65 & _off_ok
                         & np.where(_q["side"] == "no",
                                    ~((_q["pm_path_drift"]
                                       * _q["pm_path_vr3"]) > 0).fillna(False),
                                    True), "longdashdot", 1.0),
                        ("gk zd65+path 1/h", _v2_ok & _mkv_ok & _zd_ok65
                         & _off_ok
                         & np.where(_q["side"] == "no",
                                    ~((_q["pm_path_drift"]
                                       * _q["pm_path_vr3"]) > 0).fillna(False),
                                    True), "dash", 0.8),
                        # [2026-08-11] +SW: two BTC gates ported FROZEN that
                        # survived BOTH the population screen (67% weeks) and
                        # the leader-book marginal: block NO in 1h-Sideways
                        # when pm>=0.70 & stoch_1h>=70, and in double-
                        # Sideways when pm>=0.55. N_obmom REJECTED despite
                        # 100%-week population help — removes +$1,109 of
                        # leader winners (post-selection inversion, third
                        # instance).
                        ("gk zd65+path+SW", _v2_ok & _mkv_ok & _zd_ok65
                         & _off_ok
                         & np.where(_q["side"] == "no",
                                    ~((_q["pm_path_drift"]
                                       * _q["pm_path_vr3"]) > 0).fillna(False),
                                    True)
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.70)
                             & (pd.to_numeric(_q["stoch_k_1h"],
                                              errors="coerce").fillna(50)
                                >= 70))
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["markov_sol_4h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.55)),
                         "dot", 0.8),
                        # [2026-08-13] xHdamp: SAME trades as +SW, stakes
                        # scaled by tape persistence — clip((H-0.4)/0.2,
                        # 0.25, 1.0): full size at H>=0.6, quarter at
                        # H<=0.4. Hurst's first SIZING use (levels/gates
                        # adjudicated dead; d-hurst tested dead 08-13).
                        # Evidence: low-H trades lose on BOTH the SOL stack
                        # book (-$918/15, 3/3 wks, p=0.19) and ETH's young
                        # slice (same sign); modulator +$1,927->+$2,223 with
                        # zero trades dropped, and never fully exits (robust
                        # to SOL hurst's 5-flips/11wks history). MODEST
                        # evidence — forward race decides.
                        # [2026-08-18] ★PAPER: xHdamp sizing promoted into
                        # the SOL paper replica (user override of the 08-25
                        # confirm; revert clause active).
                        ("gk +SW xHdamp ★PAPER", _v2_ok & _mkv_ok & _zd_ok65
                         & _off_ok
                         & np.where(_q["side"] == "no",
                                    ~((_q["pm_path_drift"]
                                       * _q["pm_path_vr3"]) > 0).fillna(False),
                                    True)
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.70)
                             & (pd.to_numeric(_q["stoch_k_1h"],
                                              errors="coerce").fillna(50)
                                >= 70))
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["markov_sol_4h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.55)),
                         "longdash", 0.8),
                        # [2026-08-23] RETRO counterfactual of the paper
                        # stack with the DIP-tightened rescue over FULL
                        # history (user request) — paper line stays honest,
                        # this line shows what the tightening would have
                        # done all along.
                        ("‹mon› PAPER dipRESC (retro)", _v2R & _mkvR
                         & _zd_ok65 & _offR
                         & np.where(_q["side"] == "no",
                                    ~((_q["pm_path_drift"]
                                       * _q["pm_path_vr3"]) > 0).fillna(False),
                                    True)
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.70)
                             & (pd.to_numeric(_q["stoch_k_1h"],
                                              errors="coerce").fillna(50)
                                >= 70))
                         & ~((_q["side"] == "no")
                             & (_q["markov_sol_1h"].astype(str) == "Sideways")
                             & (_q["markov_sol_4h"].astype(str) == "Sideways")
                             & (_q["p_market"] >= 0.55)),
                         "dot", 0.8)]:
                    _vq = _q[_vmask].copy()
                    # [2026-08-11] 1/h refinement: max one trade per
                    # underlying hour (same-hour 15m contracts share the
                    # price path — correlated exposure, the 08-08/08-11
                    # cluster-loss mode). Parameter-free; improves net, DD
                    # and Sharpe on the 12-day record (n small, forward
                    # race decides).
                    if _vn == "gk zd65+path 1/h" and len(_vq):
                        _vq = _vq.sort_values("dt").drop_duplicates(
                            _vq["dt"].dt.floor("h").rename("eh"),
                            keep="first") if False else _vq.assign(
                            _eh=_vq["dt"].dt.floor("h")).drop_duplicates(
                            "_eh", keep="first").drop(columns="_eh")
                    _vp, _vdd = _kbook(_vq, _col)
                    # [2026-08-08] gk combo+damp: vrM+zd65 stack + the LIVE
                    # drawdown-from-peak Kelly dampener (drawdown_risk.py,
                    # frozen params 10d/z2/x0.5/min15d, causal). Long-window
                    # validation: combo +$6,801->+$7,810, DD -$1,009, S
                    # 0.27->0.34; NEVER triggers on gk hurst (DD too small);
                    # realized-edge sibling NOT used — banked negative for
                    # SOL. Needs 15d of window history => multiplier is 1.0
                    # until ~08-14 on this tab: identical to vrM+zd65 till
                    # then, diverges only on live forward data.
                    if _vn in ("gk +SW xHdamp ★PAPER", "‹mon› PAPER dipRESC (retro)") and len(_vq):
                        _hmul = np.clip(
                            (pd.to_numeric(_vq["hurst_exponent_5m"],
                                           errors="coerce") - 0.4) / 0.2,
                            0.25, 1.0).fillna(1.0)
                        _vp = _vp * _hmul
                        _cumh = _vp.cumsum()
                        _vdd = float((_cumh.cummax() - _cumh).max())
                    if _vn == "gk combo+damp" and len(_vq):
                        _dser = _vp.groupby(_vq["dt"].dt.floor("D")).sum()
                        _dmul = {}
                        _ds = _dser.sort_index()
                        for _i, _D in enumerate(_ds.index):
                            _h = _ds.iloc[:_i]
                            if len(_h) < 15:
                                _dmul[_D] = 1.0; continue
                            _cum = _h.cumsum()
                            _hwm = _cum.rolling(10, min_periods=1).max()
                            _ddw = (_hwm - _cum).clip(lower=0)
                            _sd = _ddw.expanding(min_periods=15).std().iloc[-1]
                            if not _sd or pd.isna(_sd):
                                _dmul[_D] = 1.0; continue
                            _dmul[_D] = 0.5 if (_ddw.iloc[-1] / _sd) > 2.0 else 1.0
                        _vp = _vp * _vq["dt"].dt.floor("D").map(_dmul).fillna(1.0)
                        _cumv = _vp.cumsum()
                        _vdd = float((_cumv.cummax() - _cumv).max())
                    if len(_vq):
                        _traces.append((_vn, _vq["dt"], _vp.cumsum(),
                                        f"{_lbl} {_vn}",
                                        dict(color=_clr, width=_vw,
                                             dash=_vdash)))
                    _famdaily.setdefault(_vn, []).append(
                        _vp.groupby(_vq["dt"].dt.floor("D")).sum())
                    # [2026-08-17] companion-row cells self-identify: two
                    # investigations were burned on reading a companion
                    # variant cell as the paper book's (the rows share
                    # identical column names). Any cell read in isolation
                    # now carries its model row.
                    _cell15 = (f"${_vp.sum():+,.0f} (n={len(_vq)}, "
                               f"DD ${_vdd:,.0f})")
                    _r[_vn] = (f"‹slope› {_cell15}"
                               if _lbl.startswith("slope") else _cell15)
                    # [2026-08-20] era-split mirror for the swap-boundary
                    # question (user: which MODEL is better — the rows are
                    # COLUMN streams, mixed models across the 08-12 18:00
                    # role swap; un-flip: pre-swap deciding col = OLD
                    # model, pre-swap companion col = SLOPE; post-swap
                    # reversed). Same series the cell renders.
                    try:
                        _swb = pd.Timestamp("2026-08-12 18:00", tz="UTC")
                        _prem = _vq["dt"] < _swb
                        _dS = _vp.groupby(_vq["dt"].dt.floor("D")).sum()
                        _dS = _dS[_dS != 0]
                        _shp_ = (float(_dS.mean() / _dS.std())
                                 if len(_dS) > 2 and _dS.std() > 0 else None)
                        _era_rows.append({
                            "row": _lbl, "variant": _vn,
                            "pre_net": round(float(_vp[_prem].sum())),
                            "pre_n": int(_prem.sum()),
                            "post_net": round(float(_vp[~_prem].sum())),
                            "post_n": int((~_prem).sum()),
                            "net": round(float(_vp.sum())),
                            "dd": round(float(_vdd)),
                            "daily_sharpe": (round(_shp_, 2)
                                             if _shp_ is not None else None),
                            "days": int(len(_dS))})
                    except Exception:
                        pass
                _rows.append(_r)
            # [2026-08-10] rank by the PRE-REGISTERED promotion metric:
            # pooled daily Sharpe (bucketed to 0.1 — differences inside
            # ~1/sqrt(n_days) are noise) with maxDD as tiebreak. User call:
            # a 0.03 Sharpe gap must not outrank a 2.5x drawdown gap.
            _famstats = {}
            for _k, _ds in _famdaily.items():
                _D = pd.concat(_ds, axis=1).fillna(0).sum(axis=1)
                _nz = _D[_D != 0]
                _S = (_nz.mean() / _nz.std()
                      if len(_nz) > 2 and _nz.std() > 0 else float("-inf"))
                _cumD = _D.cumsum()
                _mdd = float((_cumD.cummax() - _cumD).max())
                _famstats[_k] = (round(_S, 1), _mdd, float(_D.sum()))
            _rank = sorted(_famstats.items(),
                           key=lambda kv: (-kv[1][0], kv[1][1]))
            _top2 = []
            for _k, _v in _rank:
                if (_k == "gk combo+damp" and "gk vrM+zd65" in _top2
                        and abs(_famstats.get("gk vrM+zd65",
                                              (0, 0, 0))[2] - _v[2]) < 1.0):
                    continue
                _top2.append(_k)
                if len(_top2) == 2:
                    break
            # [2026-08-23 BUG FIX, user-caught] options were HARDCODED, so
            # any newly added variant that won the Sharpe ranking became a
            # default missing from the options (dipRESC retro did exactly
            # that). Options now derive from the live variant families —
            # cannot desync again.
            _fam_opts = (["flat $100"]
                         + [k for k in _famdaily.keys() if k != "flat $100"])
            _top2 = [k for k in _top2 if k in _fam_opts]
            _fams = st.multiselect(
                "Lines on chart (default = top-2 by pooled Sharpe, DD tiebreak)",
                _fam_opts, default=_top2, key="solsh_fams")
            for _fam, _x, _y, _nm, _ln in _traces:
                if _fam in _fams:
                    _figsh.add_trace(go.Scatter(x=_x, y=_y, name=_nm,
                                                line=_ln))
            _figsh.update_layout(height=320, margin=dict(l=0, r=0, t=64, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0))
            st.plotly_chart(_figsh, use_container_width=True)
            st.dataframe(pd.DataFrame(_rows), hide_index=True,
                         use_container_width=True)
            # [2026-08-20] debug mirror: dump this table verbatim at render
            # so analysis quotes the TAB's numbers, not a hand replication
            # (hand replicas drifted — user caught it).
            try:
                import json as _json
                (RESULTS_DIR / "sol_shadow_tab_stats.json").write_text(
                    _json.dumps({"rendered_at": str(pd.Timestamp.utcnow()),
                                 "rows": _rows,
                                 "era_split": _era_rows},
                                default=str, indent=1))
            except Exception:
                pass
            _dis = _sh.dropna(subset=["p_sol_old", "p_sol_slope"])
            _dis = _dis[(_dis["p_sol_old"] - _dis["p_sol_slope"]).abs() >= 0.05]
            if len(_dis):
                _shadow_right = np.mean(
                    np.where(_dis["p_sol_slope"] > _dis["p_sol_old"],
                             _dis["resolved_yes"] == 1, _dis["resolved_yes"] == 0))
                st.caption(f"Model disagreements ≥5pp: {len(_dis)} scans — shadow's side "
                           f"settled correct {_shadow_right:.0%} of the time.")
    except Exception as _shex:
        st.warning(f"shadow tab error: {_shex}")

with tab_15m_shadow:
    # [2026-08-05] BTC/ETH 15m shadow A/B — stage 1 of the harness ladder
    # the SOL SHADOW tab graduated through (user-endorsed process): flat
    # $100 books on identical scans first; gates/kelly/variants get added
    # LATER via marginal-contribution testing as the record accrues.
    # BTC challenger: 5-seed refresh ensemble in p_gbdt since 08-02 22:00
    # (staleness-test justified, abba68e). ETH challenger: 5-seed refresh
    # in p_gbdt since 08-05 ~06:00 — DISCLOSED: ETH's staleness test was
    # NOT confirmed; this arm exists to give the harness a real challenger
    # (p_gbdt previously duplicated production on ETH) and is judged on
    # forward paper only.
    _AB15 = {
        "BTC": {"start": pd.Timestamp("2026-08-05 22:00", tz="UTC"),
                "note": "challenger = MARKET-ANCHORED model (08-05: pm + "
                        "pm-trajectory + tau/offset/spread/vol — learns the "
                        "market's residual biases; replaced the refresh "
                        "ensemble, which shared production's 20 features and "
                        "inherited its anti-informative divergence). Walk-"
                        "forward: failed pooled BUT monotone learning curve, "
                        "all 5 seeds positive at the last origin where "
                        "production was sharply negative. 08-11 = descriptive "
                        "peek only; decision ~08-18. Scans without a 3-10min "
                        "prior pm observation get no shadow value (blank "
                        "p_gbdt) by design."},
        "ETH": {"start": pd.Timestamp("2026-08-05 05:15", tz="UTC"),
                "note": "challenger = 5-seed refresh ensemble (08-05) — "
                        "staleness NOT confirmed for ETH; unjustified-retrain "
                        "arm, judged on forward record only; first read "
                        "~08-18."},
    }
    _a15 = st.radio("Asset", ["BTC", "ETH"], horizontal=True, key="ab15_asset")
    _cfg15 = _AB15[_a15]
    st.markdown(
        f"<div style='color:#f0a500;font-size:0.78rem;margin-bottom:8px;'>"
        f"{_a15} 15m model A/B on identical live scans since "
        f"{_cfg15['start']:%Y-%m-%d %H:%M} UTC — production (p_model_15m) vs "
        f"shadow (p_gbdt) vs fixed 50/50 blend, hypothetical flat-$100 books "
        f"(edge ≥ 0.04, one bet per contract, net of fees). Decisions remain "
        f"production-only. {_cfg15['note']}</div>",
        unsafe_allow_html=True,
    )
    try:
        _abp = pd.read_csv(ASSET_CSV_15M[_a15], low_memory=False)
        _abp["dt"] = pd.to_datetime(_abp["logged_at"], errors="coerce",
                                    utc=True, format="mixed")
        # [2026-08-11] pm-path: live-logged cols take precedence; earlier
        # rows filled from the candle-archive backfill (sign rule unfitted
        # => retro render legitimate).
        try:
            _bfp = pd.read_csv(RESULTS_DIR
                               / f"{_a15.lower()}15m_pmpath_backfill.csv")
            _abp = _abp.merge(_bfp, on=["logged_at", "contract_ticker"],
                              how="left")
            for _pc, _bc in [("pm_path_drift", "pm_path_drift_bf"),
                             ("pm_path_vr3", "pm_path_vr3_bf")]:
                if _pc not in _abp.columns:
                    _abp[_pc] = np.nan
                _abp[_pc] = pd.to_numeric(_abp[_pc], errors="coerce").fillna(
                    pd.to_numeric(_abp[_bc], errors="coerce"))
        except Exception:
            pass
        for _c in ["p_market", "p_model_15m", "p_gbdt", "p_sol_old",
                   "p_sol_slope", "resolved_yes",
                   "offset_pct", "body_15m", "dir_15m", "stoch_k_5m",
                   "stoch_k_15m", "stoch_k_1h", "chg_5m", "chg_15m", "chg_1h",
                   "composite_p_up", "liq_score", "vol_ratio",
                   "vwap_hmm_state", "rsi_1h", "consec_dir_15m", "bp_5m",
                   "hurst_exponent_5m"]:
            if _c in _abp.columns:
                _abp[_c] = pd.to_numeric(_abp[_c], errors="coerce")
        _ab = _abp[(_abp["dt"] >= _cfg15["start"])
                   & _abp["resolved_yes"].notna()
                   & _abp["p_market"].between(0.03, 0.97)].copy()
        if len(_ab) < 3:
            st.info(f"Collecting… {len(_ab)} resolved scans since challenger "
                    f"go-live (unresolved scans settle within ~15 min).")
        else:
            # [2026-08-12] BTC p_model_15m semantics fix: the legacy runner
            # logged the BEST-SIDE probability (P(NO) on model-NO-leaning
            # rows, non-coherent convention) while these books read the
            # column as P(YES) — on NO-leaning rows the book bought YES
            # against the model lean. Shown immaterial over the A/B window
            # (honest +$4,589/n43/WR86% vs artifact +$4,627/n50/WR78%; only
            # 53/739 rows flipped). Reconstruct true P(YES) for pre-cutover
            # rows via raw_edge (identifies the leaning); the DUAL-replica
            # runner logs uniform P(YES) from the cutover on.
            if _a15 == "BTC" and "raw_edge" in _ab.columns:
                _cut12 = pd.Timestamp("2026-08-12 22:00", tz="UTC")
                _vv = _ab["p_model_15m"]
                _rev = pd.to_numeric(_ab["raw_edge"], errors="coerce")
                _dyes = (_rev - (_vv - _ab["p_market"])).abs()
                _dno = (_rev - (_vv - (1 - _ab["p_market"]))).abs()
                _flip = (_ab["dt"] < _cut12) & (_dno < _dyes)
                _ab.loc[_flip, "p_model_15m"] = 1 - _vv[_flip]
            _ab["p_blend"] = (_ab["p_model_15m"] + _ab["p_gbdt"]) / 2
            _books15 = [("production", "p_model_15m", "#4f8bf9"),
                        ("shadow", "p_gbdt", "#f0a500"),
                        ("blend 50/50", "p_blend", "#b57edc")]
            # [2026-08-15] ETH refresh challenger RETIRED (5 second-life
            # nulls; p_gbdt blank on new ETH rows). [2026-08-18 user
            # cleanup] Its frozen line AND the blend (which consumes the
            # retired shadow's values — dead since 08-15) are REMOVED
            # from display entirely: each book was dragging a full set of
            # frozen variant sub-lines onto the chart. CSV history
            # intact — the lines rebuild from data if ever re-added.
            # Seat reserved for the ~08-30 retrain candidate.
            if _a15 == "ETH":
                _books15 = [b for b in _books15 if b[0] == "production"]
            # [2026-08-14] BTC: blend book RETIRED from display (user call —
            # seat given to DUAL v3c below; the blend was an early-harness
            # construct never in contention). ETH keeps its blend line.
            if _a15 == "BTC":
                _books15 = [b for b in _books15 if b[0] != "blend 50/50"]
            # [2026-08-05 pm] BTC-only BENCHMARK book: z-expansion applied to
            # the MARKET probability (k=1.8 frozen from SOL, no fitting) —
            # the favorite-longshot bias trade. Backtest 06-01+: +$6,243,
            # WR 74.4% vs BE 72.0%, 8/11 wks, boot p=0.06; smooth k-plateau.
            # Does NOT replicate on ETH (-$1,708) / SOL (-$4,063) => BTC-only,
            # modest confidence; fees cleared at mid — live would fight the
            # spread at extremes (maker territory). Model-FREE: serves as the
            # benchmark the model books must beat at the 08-11/08-18 reads.
            # Forward scoring from 08-05.
            if _a15 == "BTC":
                from scipy.stats import norm as _n15
                _ab["p_mktfav"] = _n15.cdf(
                    1.8 * _n15.ppf(_ab["p_market"].clip(0.01, 0.99)))
                _books15.append(("mkt-fav k1.8", "p_mktfav", "#9aa0a6"))
            _fig15 = go.Figure()
            _rows15 = []
            for _lbl, _col, _clr in _books15:
                _s = _ab.dropna(subset=[_col]).copy()
                _fee = 0.07 * _s["p_market"] * (1 - _s["p_market"])
                _ey = _s[_col] - _s["p_market"] - _fee
                _en = _s["p_market"] - _s[_col] - _fee
                _s["side"] = np.where(_ey >= _en, "yes", "no")
                _s["edge"] = np.maximum(_ey, _en)
                _q = _s[_s["edge"] >= 0.04].sort_values("dt").drop_duplicates(
                    "contract_ticker", keep="first")
                # [2026-08-15] BTC universal regime guard, mirrors the
                # relaxed runner gate: NO trades blocked when z_drift_6h
                # > 2.5 (extreme uptrend; validated sim WR 6.3%). Before
                # the relax these scans were never LOGGED, so no
                # historical row in the A/B window is affected — this only
                # governs newly-visible extreme-uptrend scans, keeping
                # every tab book in lockstep with the paper DUAL.
                if _a15 == "BTC":
                    _zdq = pd.to_numeric(_q.get("z_drift_6h"),
                                         errors="coerce")
                    _q = _q[~((_q["side"] == "no")
                              & (_zdq > 2.5).fillna(False))]
                _cost = np.where(_q["side"] == "yes", _q["p_market"],
                                 1 - _q["p_market"])
                _win = np.where(_q["side"] == "yes", _q["resolved_yes"] == 1,
                                _q["resolved_yes"] == 0)
                _feeq = 0.07 * _q["p_market"] * (1 - _q["p_market"])
                _pnl = pd.Series(np.where(_win, 100 * (1 - _cost) / _cost, -100)
                                 - (100 / _cost) * _feeq, index=_q.index)
                # traces anchor at (window start, $0) so a large first
                # trade reads as a jump, not a starting offset
                _fig15.add_trace(go.Scatter(
                    x=[_cfg15["start"]] + list(_q["dt"]),
                    y=[0.0] + list(_pnl.cumsum()), name=_lbl,
                    line=dict(color=_clr, width=2)))
                _cum15 = _pnl.cumsum()
                _dd15 = float((_cum15.cummax() - _cum15).max()) if len(_q) else 0.0
                _row15 = {
                    "book": _lbl,
                    "net": f"${_pnl.sum():+,.0f}",
                    "n": len(_q),
                    "WR/BE": (f"{np.mean(_win):.0%}/{np.mean(_cost):.0%}"
                              if len(_q) else "—"),
                    "maxDD": f"${_dd15:,.0f}",
                }
                # [2026-08-05] BTC gated+kelly variant: the 12 live-runner
                # gates that SURVIVED marginal-contribution testing (long
                # window 07-10+ production, each gate scored by the kelly
                # PnL of trades it uniquely blocks). Dropped: Y_stochOB
                # (+$130 removed = not helping), Y_lowpm (never fires in
                # this construction). Untestable (no logged column):
                # hmm_state0 NO-gate — disclosed, not silently skipped.
                # FLAGGED for 08-11: N_obmom helps production long-window
                # (−$1,499) but its blocks were shadow-book winners in the
                # first A/B days (+$2,105, n=8) — re-check on forward data.
                # Gates block (live runner FLIPS YES→NO instead — harness
                # convention is block, disclosed). ETH: no gated variant
                # yet — its challenger is hours old; gates get the same
                # marginal treatment once a record exists.
                if _a15 == "BTC" and _lbl == "mkt-fav k1.8":
                    _mfav_book = _q[["contract_ticker", "dt", "side"]].copy()
                    _mfav_book["pnl"] = _pnl.values
                    # [2026-08-23 mkt-fav deep dive — user-approved monitor
                    # lines] The raw book decayed forward (−$510 since
                    # 08-05, wk 08-17 −$2,147). Full-history search
                    # (project_btc15m_mktfav_deepdive_20260823): candidate
                    # package blocks BOTH sides when ou_theta(1h)<2.2243 OR
                    # tau<8.64min. Full-history kept book +$10,822 S 0.29
                    # DD $1,295, 13/13 wks+, blocked bucket boot pT=0.007/
                    # pW=0.015; both plateaus smooth, fire-rate era-
                    # stationary. HONESTY: thresholds are discoveries on
                    # the 05-25..08-23 window — post-08-23 is the first
                    # true forward test (read ~09-01). Frozen constants;
                    # NaN fails OPEN (replay-faithful). Shadow KV package
                    # does NOT transfer to this book (blocks winners) —
                    # tested, not wired.
                    try:
                        _ouq = pd.to_numeric(_q.get("ou_theta"),
                                             errors="coerce")
                        _tauq = pd.to_numeric(_q.get("tau_minutes"),
                                              errors="coerce")
                        _ou_blk = (_ouq < 2.2243).fillna(False)
                        _outau_blk = _ou_blk | (_tauq < 8.64).fillna(False)
                        for _mfnm, _mfblk, _mfdsh in [
                                ("mkt-fav +OU", _ou_blk, "dot"),
                                ("mkt-fav +OUtau", _outau_blk, "dash")]:
                            _qm = _q[~np.asarray(_mfblk, bool)]
                            _pm2 = _pnl[~np.asarray(_mfblk, bool)]
                            if not len(_qm):
                                continue
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_qm["dt"]),
                                y=[0.0] + list(_pm2.cumsum()),
                                name=_mfnm,
                                line=dict(color=_clr, width=1.5,
                                          dash=_mfdsh)))
                            _cm2 = _pm2.cumsum()
                            _rows15.append({
                                "book": _mfnm,
                                "net": f"${_pm2.sum():+,.0f}",
                                "n": len(_qm), "WR/BE": "—",
                                "maxDD": f"${float((_cm2.cummax() - _cm2).max()):,.0f}",
                            })
                    except Exception:
                        pass
                # [2026-08-14] shadow book captured for DUAL v2 (user
                # proposal: prod g+k + SHADOW as the pair — near-disjoint by
                # construction, shadow trades later scans; overlap 8 vs
                # mkt-fav's 56; window S 0.62/DD $784 vs 0.58/$1,652).
                # Paper-runner arm swap waits on the shadow's own
                # pre-registered 08-18 read.
                if _a15 == "BTC" and _lbl == "shadow":
                    _shad_book = _q[["contract_ticker", "dt", "side",
                                     "p_market"]
                                    + (["spread"] if "spread" in _q.columns
                                       else [])].copy()
                    _shad_book["pnl"] = _pnl.values
                    _shad_book["win"] = np.asarray(_win, bool)
                    _shad_book["stake"] = 100.0
                    # [2026-08-18 SHADOW-KV PACKAGE — the leave-no-stone
                    # gate search's product, user-wired same day] Two
                    # gated shadow monitor books. kalman/garch history
                    # BACKFILLED same day (user caught the forward-only
                    # gap): causal completed-bar reconstruction from
                    # Binance 15m klines, sweep-identical windows, blank
                    # cells only, under the CSV lock — the lines show the
                    # full gated history. HONESTY: that history is the kv
                    # gate's own DISCOVERY window (only the volgate half
                    # was pre-registered); the post-08-18 segment is its
                    # first true forward test (08-25 read).
                    #   +KVpkg  = drop kalman_vel_15m<1e-4 | YES&d15_rv<-0.012
                    #             (the PAPER v2 arm's exact gate)
                    #   kv|garch = drop kalman_vel_15m<1e-4 | garch_vol>=0.0769
                    #             (window S 0.82/DD $103 on n=29 — carved,
                    #             monitor-only until it earns n)
                    try:
                        _kvq = pd.to_numeric(
                            _q.get("kalman_velocity_15m"), errors="coerce")
                        _rvq = pd.to_numeric(
                            _q.get("d15_realized_vol_annual"),
                            errors="coerce")
                        _gvq = pd.to_numeric(
                            _q.get("garch_vol_15m"), errors="coerce")
                        _kv_blk = (_kvq < 0.0001).fillna(False)
                        _pkg_blk = _kv_blk | ((_q["side"] == "yes")
                                              & (_rvq < -0.012).fillna(False))
                        _gar_blk = _kv_blk | (_gvq >= 0.0769).fillna(False)
                        for _bknm, _blkm, _dsh in [
                                ("shadow +KVpkg (paper arm)", _pkg_blk, "solid"),
                                ("shadow kv|garch", _gar_blk, "dot")]:
                            _qb = _q[~np.asarray(_blkm, bool)]
                            _pb = _pnl[~np.asarray(_blkm, bool)]
                            if not len(_qb):
                                continue
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_qb["dt"]),
                                y=[0.0] + list(_pb.cumsum()),
                                name=_bknm,
                                line=dict(color=_clr, width=1.5,
                                          dash=_dsh)))
                            _cb = _pb.cumsum()
                            _rows15.append({
                                "book": _bknm,
                                "net": f"${_pb.sum():+,.0f}",
                                "n": len(_qb), "WR/BE": "—",
                                "maxDD": f"${float((_cb.cummax() - _cb).max()):,.0f}",
                            })
                        # [2026-08-18 pre-registered ×EDGEconv — user call
                        # after the deviation-conviction finding: the
                        # anchored arm's margin is MONOTONE in its own
                        # edge (+9/+10/+19/+22pp by edge band); mkt-fav's
                        # tier inversion was a coarse proxy for this.
                        # Constants DECLARED (not fitted): stake ×
                        # clip(edge/0.08, 0.5, 2.0). Monitor only — wires
                        # at 08-25 ONLY IF the KV arm itself confirms AND
                        # this leads flat on Sharpe/DD. Never apply
                        # mkt-fav conviction tiers to this arm (inverted;
                        # see tracker 08-18).]
                        _kvkeep = ~np.asarray(_pkg_blk, bool)
                        _wE2 = np.clip(_q["edge"][_kvkeep] / 0.08, 0.5, 2.0)
                        _pE2 = _pnl[_kvkeep] * _wE2
                        _qE2 = _q[_kvkeep]
                        if len(_qE2):
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_qE2["dt"]),
                                y=[0.0] + list(_pE2.cumsum()),
                                name="‹mon› KVpkg ×EDGEconv",
                                line=dict(color=_clr, width=1.5,
                                          dash="dashdot")))
                            _cE2 = _pE2.cumsum()
                            _rows15.append({
                                "book": "‹mon› KVpkg ×EDGEconv",
                                "net": f"${_pE2.sum():+,.0f}",
                                "n": len(_qE2), "WR/BE": "—",
                                "maxDD": f"${float((_cE2.cummax() - _cE2).max()):,.0f}",
                            })
                        # [2026-08-20 ‹mon› KV ×VOLdamp — pre-declared,
                        # user sizing survey for the shadow arm] The only
                        # monotone-STABLE sizing signals across split
                        # halves of the KV-gated replay (n=114) are
                        # calm-tape ones: realized_vol_annual and side-
                        # aware pm-chase — the arm's OWN edge is NOT
                        # stable-monotone there (temper xEDGEconv
                        # expectations at its 08-25 read). Declared rule,
                        # round constant, one change: stake ×0.5 when
                        # realized_vol_annual >= 0.30 (fires ~33%; replay
                        # per-trade sharpe pre 0.322->0.312 / post
                        # 0.052->0.080 — weak-half improvement for small
                        # strong-half cost). Monitor ONLY; read ~09-03;
                        # never stack with xEDGEconv before both confirm.
                        try:
                            _rvVD = pd.to_numeric(
                                _q.get("realized_vol_annual"),
                                errors="coerce")
                            _wVD = np.where((_rvVD >= 0.30).fillna(False),
                                            0.5, 1.0)
                            _kvk2 = ~np.asarray(_pkg_blk, bool)
                            _pVD = _pnl[_kvk2] * _wVD[_kvk2]
                            _qVD = _q[_kvk2]
                            if len(_qVD):
                                _fig15.add_trace(go.Scatter(
                                    x=[_cfg15["start"]] + list(_qVD["dt"]),
                                    y=[0.0] + list(_pVD.cumsum()),
                                    name="‹mon› KV ×VOLdamp (×0.5 @rv≥.30)",
                                    line=dict(color=_clr, width=1.5,
                                              dash="dot")))
                                _cVD = _pVD.cumsum()
                                _rows15.append({
                                    "book": "‹mon› KV ×VOLdamp",
                                    "net": f"${_pVD.sum():+,.0f}",
                                    "n": len(_qVD), "WR/BE": "—",
                                    "maxDD": f"${float((_cVD.cummax() - _cVD).max()):,.0f}",
                                })
                        except Exception:
                            pass
                        _kvmask = ~np.asarray(_pkg_blk, bool)
                        _shad_kv_book = _q[_kvmask][
                            ["contract_ticker", "dt", "side", "p_market"]
                            + (["spread"] if "spread" in _q.columns
                               else [])].copy()
                        _shad_kv_book["pnl"] = _pnl[_kvmask].values
                        _shad_kv_book["win"] = np.asarray(_win, bool)[_kvmask]
                        _shad_kv_book["stake"] = 100.0
                    except Exception:
                        _shad_kv_book = _shad_book
                    # [2026-08-14] shadow xHdamp(sol) — RETIRED FROM
                    # DISPLAY 08-18 (user cleanup: case inverted with
                    # melt-up data, S 0.12 vs raw 0.13). The sizing
                    # computation stays: DUAL v3c (pre-registered round-1
                    # leader) consumes _shad_book["pnl_h"].
                    try:
                        _shs2 = pd.read_csv(
                            ASSET_CSV_15M["SOL"],
                            usecols=["logged_at", "hurst_exponent_5m"],
                            low_memory=False)
                        _shs2["dt"] = pd.to_datetime(
                            _shs2["logged_at"], errors="coerce", utc=True,
                            format="mixed")
                        _shs2["h"] = pd.to_numeric(
                            _shs2["hurst_exponent_5m"], errors="coerce")
                        _shs2 = _shs2.dropna(
                            subset=["dt", "h"]).sort_values("dt")
                        _s2t = _shs2["dt"].astype("int64").values / 1e9
                        _q2t = _q["dt"].astype("int64").values / 1e9
                        _i2 = np.searchsorted(_s2t, _q2t, side="right") - 1
                        _h2 = np.where(
                            _i2 >= 0,
                            _shs2["h"].values[np.clip(_i2, 0, None)], np.nan)
                        _a2 = _q2t - np.where(
                            _i2 >= 0, _s2t[np.clip(_i2, 0, None)], np.nan)
                        _hm2 = pd.Series(
                            np.clip((np.where(_a2 <= 1800, _h2, np.nan)
                                     - 0.4) / 0.2, 0.25, 1.0),
                            index=_q.index).fillna(1.0)
                        _php = _pnl * _hm2
                        _shad_book["pnl_h"] = _php.values
                    except Exception:
                        pass
                if _a15 == "BTC" and _lbl != "mkt-fav k1.8":
                    _yes15 = _q["side"] == "yes"
                    _no15 = ~_yes15
                    _pm15 = _q["p_market"]
                    _m1r = _q["markov_regime_1h"].astype(str)
                    _m15r = _q["markov_regime_15m"].astype(str)
                    _cpu15 = _q["composite_p_up"]
                    _sk15q = _q["stoch_k_15m"].fillna(50)
                    _sk1hq = _q["stoch_k_1h"].fillna(50)
                    _ok15 = ~(_yes15 & (_q["offset_pct"].fillna(0) < 0.025))
                    _ok15 &= ~(_yes15 & (_m1r == "Bear"))
                    _ok15 &= ~(_yes15 & (_m15r == "Bear")
                               & ~(_cpu15 <= 0.488).fillna(False))
                    _ok15 &= ~(_yes15 & (_q["dir_15m"] == 1)
                               & (_pm15 >= 0.50) & (_pm15 < 0.65)
                               & ~((_m1r == "Bull")
                                   | ((_m1r == "Bear") & (_m15r == "Bear")
                                      & (_sk1hq < 35))))
                    _ok15 &= ~(_yes15 & (_m1r == "Sideways")
                               & (_q["body_15m"].fillna(1) < 0.30)
                               & ~((_cpu15 < 0.40).fillna(False)
                                   | ((_sk15q >= 20) & (_sk15q < 40))))
                    _ok15 &= ~(_yes15 & (_sk1hq >= 95)
                               & (_q["liq_score"] == -1).fillna(False))
                    # [2026-08-20] mid-cost NO block mirror (runner gate
                    # deployed ~04:50 UTC; era-robust bleed cut, see
                    # _replica_decide_btc comment). dt-gated so displayed
                    # history is NOT retroactively flattered — pre-deploy
                    # rows keep the book's real record.
                    _ok15 &= ~(_no15 & (_pm15 >= 0.50) & (_pm15 < 0.80)
                               & (_q["dt"] >= pd.Timestamp(
                                   "2026-08-20 04:50", tz="UTC")))
                    _ok15 &= ~(_no15 & (_q["chg_1h"].fillna(0) > 0)
                               & (_sk1hq >= 30) & (_sk1hq < 70))
                    _ok15 &= ~(_no15 & (_m1r == "Sideways")
                               & (_pm15 >= 0.70) & (_sk1hq >= 70))
                    _ok15 &= ~(_no15 & (_m1r == "Sideways")
                               & (_m15r == "Sideways") & (_pm15 >= 0.55))
                    _ok15 &= ~(_no15 & (_m1r == "Bear") & (_m15r == "Bull"))
                    _ok15 &= ~(_no15 & (_q["stoch_k_5m"] > 76).fillna(False)
                               & (_q["chg_5m"] > 0).fillna(False))
                    _vst15 = _q["vwap_hmm_state"]
                    _ok15 &= ~(_no15 & ((_vst15 == 4)
                               | ((_vst15 == 2)
                                  & (_q["vol_ratio"].fillna(1) < 0.216))
                               | ((_vst15 == 5) & (_sk1hq < 85))
                               | ((_vst15 == 7)
                                  & (_q["chg_15m"].fillna(0) >= -0.112))))
                    _gk15 = _q[np.asarray(_ok15, bool)].copy()
                    _gc = np.where(_gk15["side"] == "yes", _gk15["p_market"],
                                   1 - _gk15["p_market"])
                    _gw = np.where(_gk15["side"] == "yes",
                                   _gk15["resolved_yes"] == 1,
                                   _gk15["resolved_yes"] == 0)
                    _gf = 0.07 * _gk15["p_market"] * (1 - _gk15["p_market"])
                    _gfr = np.where(
                        _gk15["side"] == "yes",
                        (_gk15[_col] - _gk15["p_market"] - _gf)
                        / (1 - _gk15["p_market"]),
                        (_gk15["p_market"] - _gk15[_col] - _gf)
                        / _gk15["p_market"])
                    _gs = 2500.0 * np.clip(_gfr, 0, 0.10)
                    _gp15 = pd.Series(
                        np.where(_gw, _gs * (1 - _gc) / _gc, -_gs)
                        - (_gs / _gc) * _gf, index=_gk15.index)
                    if len(_gk15):
                        _fig15.add_trace(go.Scatter(
                            x=[_cfg15["start"]] + list(_gk15["dt"]),
                            y=[0.0] + list(_gp15.cumsum()),
                            name=f"{_lbl} gated+kelly",
                            line=dict(color=_clr, width=1.5, dash="dash")))
                    _gcum15 = _gp15.cumsum()
                    _gdd15 = (float((_gcum15.cummax() - _gcum15).max())
                              if len(_gk15) else 0.0)
                    _row15["g+k (12 gates)"] = (
                        f"${_gp15.sum():+,.0f} (n={len(_gk15)}, "
                        f"DD ${_gdd15:,.0f})")
                    if _lbl == "production":
                        _prod_gk = _gk15[["contract_ticker", "dt",
                                          "side", "p_market"]
                                         + (["spread"]
                                            if "spread" in _gk15.columns
                                            else [])].copy()
                        _prod_gk["pnl"] = _gp15.values
                        _prod_gk["win"] = np.asarray(_gw, bool)
                        _prod_gk["stake"] = np.asarray(_gs, float)
                        # [2026-08-14] xMFconv (user directive: mkt-fav as
                        # the kelly SIZER — flat-kelly hits the 10% cap on
                        # ~every trade, so sizing carried no information).
                        # PRE-DECLARED tiers on mkt-fav's stance toward the
                        # trade: x1.00 convicted co-sign (WR 94% vs BE 77%),
                        # x0.75 lean-agree, x0.50 lean-object, x0.25
                        # convicted objection (WR 20% at-par — the
                        # fade-the-favorite lottery sleeve, trimmed not
                        # killed). Window: S 0.53->0.99, DD 557->511, net
                        # -$1,600 (almost all = the pm-0.93 13:1 winner cut
                        # to quarter-stake — the declared trade-off).
                        # Multipliers declared 08-14, not fitted; forward
                        # race referees.
                        _pmfq = _gk15["p_mktfav"]
                        _eyq = _pmfq - _gk15["p_market"] - _gf
                        _enq = _gk15["p_market"] - _pmfq - _gf
                        _mfsd = np.where(_eyq >= _enq, "yes", "no")
                        _mfed = np.maximum(_eyq, _enq)
                        _agq = _mfsd == _gk15["side"]
                        _tier = np.where(_agq & (_mfed >= 0.04), 1.00,
                                np.where(_agq, 0.75,
                                np.where(_mfed < 0.04, 0.50, 0.25)))
                        _gsC = _gs * _tier
                        _gpC = pd.Series(
                            np.where(_gw, _gsC * (1 - _gc) / _gc, -_gsC)
                            - (_gsC / _gc) * _gf, index=_gk15.index)
                        if len(_gk15):
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_gk15["dt"]),
                                y=[0.0] + list(_gpC.cumsum()),
                                name="production g+k xMFconv",
                                line=dict(color=_clr, width=1.5,
                                          dash="dashdot")))
                        _cumC = _gpC.cumsum()
                        _ddC = (float((_cumC.cummax() - _cumC).max())
                                if len(_gk15) else 0.0)
                        _row15["g+k xMFconv"] = (
                            f"${_gpC.sum():+,.0f} (n={len(_gk15)}, "
                            f"DD ${_ddC:,.0f})")
                        _prod_gk["pnl_mf"] = _gpC.values
                # [2026-08-05] ETH gated+kelly: the 5 live gates that SURVIVED
                # marginal testing (Y_stoch5m44, Y_stoch1h_mid[rsi<35 rescue],
                # N_daily_sw, N_consec, N_downcandle). REJECTED as inverted on
                # current data: Y_lowvol (+$7,562 of blocked winners long-
                # window, +$6,892 post-swap), N_stoch1h_ob (+$2,890),
                # N_oversold_C/GateC (+$2,051), N_kc (n=7 mixed); Y_lowcpu
                # never fires. Survivor stack turns the harness book POSITIVE:
                # production +$4,338 long / +$6,888 post-swap. Windows straddle
                # the 07-29 production-model swap — gates scored on both.
                if _a15 == "ETH":
                    _yesE = _q["side"] == "yes"
                    _noE = ~_yesE
                    _pmE = _q["p_market"]
                    _mdE = _q["markov_eth_daily"].astype(str)
                    _sk5E = _q["stoch_k_5m"].fillna(50)
                    _sk15E = _q["stoch_k_15m"].fillna(50)
                    _sk1hE = _q["stoch_k_1h"].fillna(50)
                    _okE = ~(_yesE & (_sk5E >= 44))
                    _okE &= ~(_yesE & (_sk1hE >= 30) & (_sk1hE < 70)
                              & ~(_q["rsi_1h"] < 35).fillna(False))
                    _okE &= ~(_noE & (_mdE == "Sideways"))
                    _okE &= ~(_noE & (_q["consec_dir_15m"] <= -1).fillna(False)
                              & (_sk15E <= 40))
                    _okE &= ~(_noE & (_q["dir_15m"] == -1) & (_pmE >= 0.50)
                              & ~((_q["body_15m"].fillna(0) > 0.60)
                                  & (_q["bp_5m"].fillna(0.5) < 0.45)
                                  & ~(_q["liq_score"] == -2).fillna(False)))
                    # [2026-08-07] hurst NO-block runs as its OWN variant
                    # (user call — the 5-gate book keeps its record; one
                    # change at a time per the harness doctrine). Hurst
                    # column logs for ETH from 08-07 02:05 deploy; NaN rows
                    # (pre-deploy) pass — the +hurst book PHASES IN and is
                    # identical to the 5-gate book until then. Evidence:
                    # SOL's frozen 0.6131 on ETH reconstruction -$8,841/251
                    # P=0.03; YES untouched (+$13,130 of winners in trend).
                    _hu_okE = ~(_noE & (_q["hurst_exponent_5m"]
                                        >= 0.6131).fillna(False))
                    # [2026-08-11] +path variant: 5 gates + block NO when
                    # pm_path_drift x pm_path_vr3 > 0. Books-level: targets
                    # -$6,433/65 on this book, 88% of weeks helpful — ETH's
                    # stack does NOT absorb the signal (unlike SOL
                    # production's). History = candle backfill; live cols
                    # (logging from 08-11) take precedence.
                    _pa_okE = np.where(
                        _q["side"] == "no",
                        ~((pd.to_numeric(_q.get("pm_path_drift"),
                                         errors="coerce")
                           * pd.to_numeric(_q.get("pm_path_vr3"),
                                           errors="coerce")) > 0).fillna(False),
                        True)
                    # [2026-08-13] +NOtrio variant: 5 gates + three NO-side
                    # blocks BORROWED from the other assets' validated
                    # stacks (marginal sweep on the post-swap book, n=307):
                    #   SOL pm>0.8 band     — blkWR 14%, +$1,821, n=21
                    #   BTC sk5>76 & chg5>0 — blkWR 20%, +$1,636, n=15
                    #   BTC chg1h>0 & sk1h 30-70 — blkWR 20%, +$1,213, n=10
                    # Union w/ pm-path-family overlap removed: 29-36 blk,
                    # +$2,927, boot p=0.092, 2/3 weeks — MODEST/DIRECTIONAL,
                    # not significant; forward reads 08-18/08-25 decide.
                    # Deliberately NOT the drawdown fix: the 08-12/13 dip is
                    # oversold-YES-in-chop, and every gate that rescues it
                    # blocked winners pre-dip (streak flip, not structure).
                    _n3_okE = ~(_noE & (_pmE > 0.8))
                    _n3_okE &= ~(_noE & (_sk5E > 76)
                                 & (_q["chg_5m"].fillna(0) > 0))
                    _n3_okE &= ~(_noE & (_q["chg_1h"].fillna(0) > 0)
                                 & (_sk1hE >= 30) & (_sk1hE < 70))
                    # [2026-08-13] +YESknife variant: 5 gates + block YES on
                    # falling-knife (chg_1h < -0.45) or dead-tape
                    # (realized_vol_annual < 0.135). Post-swap book: 50 blk
                    # @34% WR, +$5,359, p=0.001, 3/3 wks, catches 3/4 of the
                    # 08-13 consecutive YES losses. DISCLOSED INSTABILITY:
                    # INVERTED on the pre-swap old-model book (blocked 60%
                    # winners, -$3,261, p=0.945) — either the new model
                    # can't price knife-catches (real, model-specific) or
                    # it's an August-regime artifact. Forward reads decide;
                    # do NOT promote on post-swap stats alone.
                    # [2026-08-13 v2 refinement, same day] knife block now
                    # requires OTM (offset_pct<0): ITM knives only need the
                    # price to HOLD and win 72%/56% (Jul/Aug) — released;
                    # OTM knives need a true reversal and lose in BOTH eras
                    # (42%/22%). July cost -$3,261 -> -$1,094 (R1) while
                    # keeping ~90% of the Aug value; with dead-tape,
                    # Aug delta +$4,820 p=0.000 3/3wks vs Jul -$1,278.
                    # DISCLOSED: moneyness was selected with both eras
                    # visible (though consistent within each) — forward
                    # reads referee. Reliability of the July test itself:
                    # triangulated (old-model real book, 6-seed OOS replay,
                    # model-free population edge +3.5pp) — solid.
                    _kn_okE = ~(_yesE
                                & (((_q["chg_1h"] < -0.45).fillna(False)
                                    & (_q["offset_pct"] < 0).fillna(False))
                                   | (_q["realized_vol_annual"]
                                      < 0.135).fillna(False)))
                    # [2026-08-13 rehosted same-day, user call] xHdamp(sol) could
                    # not win as a STANDALONE on the weakest host (base) —
                    # repurposed onto the leading +YESknife stack before any
                    # retirement, per the second-life rule. Composes with
                    # every competitor on fresh data (+NOtrio S .31->.38;
                    # +YESknife S .42->.47 DD 3,339->2,514; combined
                    # .51->.55). Forward race vs its host is a PURE sizing
                    # A/B (identical trades). Original note:
                    # xHdamp(sol): CROSS-ASSET hurst sizing —
                    # stakes scaled by SOL's hurst stream (causal join,
                    # <=30min stale), clip((H-0.4)/0.2, 0.25, 1.0) with
                    # form+params FROZEN from the SOL deploy (transfer
                    # test, not a fit). Improves ALL FOUR ETH books
                    # (base S 0.25->0.33 net +$1,304; +NOtrio 0.32->0.39;
                    # +YESknife 0.43->0.48 DD -$825; combined 0.52->0.56)
                    # where ETH's OWN hurst hurt all of them — the two
                    # streams correlate only 0.19; SOL's is the better
                    # market-wide chop detector and covers the full window
                    # (ETH's own logs only from 08-07).
                    try:
                        _shs = pd.read_csv(
                            ASSET_CSV_15M["SOL"],
                            usecols=["logged_at", "hurst_exponent_5m"],
                            low_memory=False)
                        _shs["dt"] = pd.to_datetime(
                            _shs["logged_at"], errors="coerce", utc=True,
                            format="mixed")
                        _shs["h"] = pd.to_numeric(
                            _shs["hurst_exponent_5m"], errors="coerce")
                        _shs = _shs.dropna(subset=["dt", "h"]).sort_values("dt")
                        _sts = _shs["dt"].astype("int64").values / 1e9
                        _qts = _q["dt"].astype("int64").values / 1e9
                        _si = np.searchsorted(_sts, _qts, side="right") - 1
                        _sh = np.where(_si >= 0,
                                       _shs["h"].values[np.clip(_si, 0, None)],
                                       np.nan)
                        _sage = _qts - np.where(_si >= 0,
                                                _sts[np.clip(_si, 0, None)],
                                                np.nan)
                        _q["h_sol"] = np.where(_sage <= 1800, _sh, np.nan)
                    except Exception:
                        _q["h_sol"] = np.nan
                    # [2026-08-18 READ #1] +hurst RETIRED (pre-registered;
                    # trailed base, blocked winners). COMBO promoted into
                    # the paper replica per its fork — its line is the
                    # paper strategy now; +knife xHdamp keeps racing as
                    # the sizing candidate on the promoted stack's knife
                    # component.
                    # [2026-08-18 DIP PACKAGE — ETH exhaustive gate
                    # search, user option A] Paper replica now = COMBO +
                    # chase-block (YES & chg_15m>=-0.0345) + knife1h
                    # (bp_1h<0.11, either side). Raw COMBO keeps racing.
                    # Monitor books (marginal, one change on the paper
                    # stack): +volhot (garch_sur_1h>=-0.0848 — post-heavy
                    # evidence, NOT wired) and +oidrop (YES &
                    # d15_oi_chg_pct<-0.036 — split-consistent, watch).
                    # garch_sur_1h logs from 08-18 (deploy) and is
                    # BACKFILLED causally from 1h klines; NaN fails open.
                    _dip_okE = ~((_yesE
                                  & (_q["chg_15m"] >= -0.0345).fillna(False))
                                 | (_q["bp_1h"] < 0.11).fillna(False))
                    _vh_okE = ~(pd.to_numeric(
                        _q.get("garch_sur_1h"), errors="coerce")
                        >= -0.0848).fillna(False)
                    _oi_okE = ~(_yesE & (pd.to_numeric(
                        _q.get("d15_oi_chg_pct"), errors="coerce")
                        < -0.036).fillna(False))
                    # [2026-08-23 pm CONSOLIDATION (user call): −YESsdwy
                    # rebuilt as a LAYER, not a competing book.] The full
                    # YES-Sideways block trailed ★PAPER badly (+$4,875 vs
                    # +$13,634) because the bucket's cheap-YES dip
                    # lotteries (cost<=0.15: +$7,229, incl. the +$4,942
                    # 19:1) are exactly what the DIP book exists to catch;
                    # the bleed lives in MID-COST Sideways YES (taken
                    # trades −$2.9k at cost 0.35-0.65; retro flat-to-neg
                    # >=0.30). Layer form: block YES & Sideways & cost >=
                    # 0.35 — keeps the tail, trims the chop. Plateau
                    # smooth 0.25-0.40 (all >= baseline S, DD better
                    # everywhere); 0.35 declared = the pre-existing
                    # cost-tercile boundary, not a fresh fit. Retro: net
                    # +$15,041 S 0.48 DD $2,582 vs ★PAPER +$13,634 / 0.45
                    # / $3,178. Races as monitor; decide at the 08-25 DIP
                    # read or 09-03. Missing regime/cost fails open.
                    _costE = np.where(_yesE, _q["p_market"],
                                      1 - _q["p_market"])
                    _sdwy_okE = ~(_yesE & (_mdE == "Sideways")
                                  & (_costE >= 0.35))
                    _combE = _okE & _n3_okE & _kn_okE
                    # [2026-08-18 user] per-variant colors — with a single
                    # book left every line inherited the book's blue and
                    # was distinguishable only by dash. Paper = green
                    # (matches the BTC tab's paper-line convention).
                    # [2026-08-23 pm FIELD CONSOLIDATION (user call):
                    # lineage lines +path / +NOtrio / +YESknife retired
                    # from display — their logic lives inside the promoted
                    # COMBO and no pending decision reads on them. Kept:
                    # g+k baseline (floor), COMBO (isolates DIP's
                    # contribution), ★PAPER + its monitors, +knife xHdamp
                    # (registered sizing race, pending). CSV history
                    # intact — lines rebuild from masks if re-added.]
                    for _vnE, _mE, _dshE, _clrE in [
                            ("g+k (5 gates)", _okE, "dash", "#4f8bf9"),
                            ("g+k COMBO", _combE, "solid", "#b57edc"),
                            ("COMBO+DIP ★PAPER", _combE & _dip_okE,
                             "solid", "#00c076"),
                            ("‹mon› DIP +volhot",
                             _combE & _dip_okE & _vh_okE, "dot", "#ff7eb6"),
                            ("‹mon› DIP +oidrop",
                             _combE & _dip_okE & _oi_okE, "dot", "#c49c48"),
                            # [2026-08-20 ‹mon› ×INVdamp — pre-declared,
                            # user build] ETH 15m conviction is INVERTED
                            # at the top: hi-edge tercile WR 21%/20% and
                            # −$148/−$118 per trade in BOTH eras since the
                            # 08-12 promotion, while plain kelly saturates
                            # its 10% cap on 91% of trades — max stake on
                            # the model's worst opinions. Constants
                            # DECLARED not fitted: stake ×0.4 when fee-adj
                            # edge >= 0.09, ×1.0 below. Inverse-conviction
                            # findings are post-selection-inversion
                            # cousins (house rule): monitor ONLY, pure
                            # sizing A/B vs ★PAPER (identical trades);
                            # score ~09-03 vs plain paper and xHdamp;
                            # wire only if it leads.
                            ("‹mon› PAPER ×INVdamp", _combE & _dip_okE,
                             "dot", "#7bb662"),
                            ("‹mon› −YESsdwy≥.35", _combE & _dip_okE
                             & _sdwy_okE, "dot", "#e8845b"),
                            ("g+k +knife xHdamp", _okE & _kn_okE,
                             "longdash", "#9aa0a6")]:
                        _gkE = _q[np.asarray(_mE, bool)].copy()
                        _gcE = np.where(_gkE["side"] == "yes",
                                        _gkE["p_market"],
                                        1 - _gkE["p_market"])
                        _gwE = np.where(_gkE["side"] == "yes",
                                        _gkE["resolved_yes"] == 1,
                                        _gkE["resolved_yes"] == 0)
                        _gfE = 0.07 * _gkE["p_market"] * (1 - _gkE["p_market"])
                        _gfrE = np.where(
                            _gkE["side"] == "yes",
                            (_gkE[_col] - _gkE["p_market"] - _gfE)
                            / (1 - _gkE["p_market"]),
                            (_gkE["p_market"] - _gkE[_col] - _gfE)
                            / _gkE["p_market"])
                        _gsE = 2500.0 * np.clip(_gfrE, 0, 0.10)
                        _gpE = pd.Series(
                            np.where(_gwE, _gsE * (1 - _gcE) / _gcE, -_gsE)
                            - (_gsE / _gcE) * _gfE, index=_gkE.index)
                        if _vnE == "g+k +knife xHdamp":
                            _gpE = _gpE * np.clip(
                                (pd.to_numeric(_gkE["h_sol"],
                                               errors="coerce") - 0.4) / 0.2,
                                0.25, 1.0).fillna(1.0)
                        if _vnE == "‹mon› PAPER ×INVdamp":
                            _gpE = _gpE * np.where(
                                pd.to_numeric(_gkE["edge"],
                                              errors="coerce") >= 0.09,
                                0.4, 1.0)
                        if len(_gkE):
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_gkE["dt"]),
                                y=[0.0] + list(_gpE.cumsum()),
                                name=_vnE,
                                line=dict(color=_clrE, width=1.5,
                                          dash=_dshE)))
                        _gcumE = _gpE.cumsum()
                        _gddE = (float((_gcumE.cummax() - _gcumE).max())
                                 if len(_gkE) else 0.0)
                        _row15[_vnE] = (
                            f"${_gpE.sum():+,.0f} (n={len(_gkE)}, "
                            f"DD ${_gddE:,.0f})")
                _rows15.append(_row15)
            # [2026-08-12] DUAL paper book (user architecture idea, first
            # combination to beat both parents): union of production g+k
            # (12 gates, kelly) and mkt-fav (flat) with de-overlap — same
            # contract opposite sides -> skip both (fee-burning
            # cancellation); same side -> model book's position only.
            # 7-day record: +$7,070 S 0.83 vs parents 0.56/0.62; corr of
            # parents' dailies ~0. LIVE preconditions unchanged (tracker).
            if _a15 == "BTC":
                try:
                    _Ad = _prod_gk.set_index("contract_ticker")
                    _Bd = _mfav_book.set_index("contract_ticker")
                    _bothd = set(_Ad.index) & set(_Bd.index)
                    _oppd = {tt for tt in _bothd
                             if _Ad.loc[tt, "side"] != _Bd.loc[tt, "side"]}
                    _rows_d = []
                    # [2026-08-12 corrected per user] FULL INDEPENDENCE:
                    # both books take every bet — no de-overlap, no
                    # arbitration; overlaps/conflicts all stand. DUAL =
                    # literal sum of the two PnL streams.
                    for tt, rr in _Ad.iterrows():
                        _rows_d.append((rr["dt"], rr["pnl"]))
                    for tt, rr in _Bd.iterrows():
                        _rows_d.append((rr["dt"], rr["pnl"]))
                    _Pd = pd.DataFrame(_rows_d, columns=["dt", "pnl"])                         .sort_values("dt")
                    if len(_Pd):
                        _fig15.add_trace(go.Scatter(
                            x=[_cfg15["start"]] + list(_Pd["dt"]),
                            y=[0.0] + list(_Pd["pnl"].cumsum()),
                            name="DUAL v1 (prod+mktfav)",
                            line=dict(color="#00c076", width=2,
                                      dash="dash")))
                        _cumd = _Pd["pnl"].cumsum()
                        _ddd = float((_cumd.cummax() - _cumd).max())
                        _rows15.append({
                            "book": "DUAL v1 (prod+mktfav)",
                            "net": f"${_Pd['pnl'].sum():+,.0f}",
                            "n": len(_Pd),
                            "WR/BE": "—",
                            "maxDD": f"${_ddd:,.0f}",
                        })
                    # [2026-08-14] DUAL v2: prod g+k + SHADOW. Held the
                    # seat 08-14..08-17 ungated (arm bled -$911/144),
                    # reverted to v1 for hours, then [2026-08-18 RESTORED
                    # + KV PACKAGE, user call after the exhaustive gate
                    # search]: the flat arm is now the KV-gated shadow
                    # (kalman_vel_15m<1e-4 | YES&d15_rv<-0.012 blocks).
                    # This line uses the gated arm — matches the paper
                    # book going forward (history: gate partially active,
                    # see the monitor-book comment above). Raw-v2 line
                    # removed in the same cleanup (dominated by
                    # construction).
                    _sb2 = (_shad_kv_book if "_shad_kv_book" in dict(locals())
                            else _shad_book)
                    _rows_d2 = ([(rr["dt"], rr["pnl"])
                                 for _, rr in _Ad.iterrows()]
                                + [(rr["dt"], rr["pnl"])
                                   for _, rr in _sb2.iterrows()])
                    _Pd2 = pd.DataFrame(_rows_d2, columns=["dt", "pnl"]
                                        ).sort_values("dt")
                    if len(_Pd2):
                        _fig15.add_trace(go.Scatter(
                            x=[_cfg15["start"]] + list(_Pd2["dt"]),
                            y=[0.0] + list(_Pd2["pnl"].cumsum()),
                            name="DUAL v2+KV ★PAPER (restored 08-18)",
                            line=dict(color="#00c076", width=2,
                                      dash="dot")))
                        _cumd2 = _Pd2["pnl"].cumsum()
                        _ddd2 = float((_cumd2.cummax() - _cumd2).max())
                        _rows15.append({
                            "book": "DUAL v2+KV ★PAPER (restored 08-18)",
                            "net": f"${_Pd2['pnl'].sum():+,.0f}",
                            "n": len(_Pd2),
                            "WR/BE": "—",
                            "maxDD": f"${_ddd2:,.0f}",
                        })
                    # [2026-08-20] GO-LIVE INFRASTRUCTURE MONITORS (user
                    # build):
                    # (a) ‹mon› DUALv2 barbell — drop mid-cost trades, keep
                    #     cost<=0.20 | >=0.70 on both arms. The MIDDLE CUT
                    #     is the only threshold-robust finding of the 08-20
                    #     cost sweep (removed PnL negative in both eras at
                    #     every boundary combo); band edges are plateau
                    #     centers, not fitted peaks. Scored as a convexity
                    #     book (ticket EV + carry) at 08-22/08-29.
                    # (b) DUAL v2+KV (fill-true) — same trades repriced at
                    #     the prices the LIVE order path actually pays
                    #     (YES: ask+1c = pm+spread/2+0.01; NO: 1-bid =
                    #     1-pm+spread/2; fee on fill; cost>=0.99 = no
                    #     fill, $0). THIS is the go-live number the 08-22
                    #     read should quote.
                    try:
                        _bb_rows, _ft_rows = [], []
                        for _armf in (_prod_gk, _sb2):
                            if not {"p_market", "win", "stake",
                                    "pnl"}.issubset(
                                        getattr(_armf, "columns", [])):
                                continue
                            _apm = pd.to_numeric(_armf["p_market"],
                                                 errors="coerce").values
                            _asp = pd.to_numeric(
                                _armf["spread"], errors="coerce").fillna(
                                    0.01).values if "spread" in \
                                _armf.columns else np.full(len(_armf), 0.01)
                            _ays = (_armf["side"] == "yes").values
                            _amc = np.where(_ays, _apm, 1 - _apm)
                            for _dtv, _pv in zip(
                                    _armf["dt"][(_amc <= 0.20)
                                                | (_amc >= 0.70)],
                                    _armf["pnl"][(_amc <= 0.20)
                                                 | (_amc >= 0.70)]):
                                _bb_rows.append((_dtv, _pv))
                            _cf = np.where(_ays, _apm + _asp / 2 + 0.01,
                                           1 - _apm + _asp / 2)
                            _stk = pd.to_numeric(_armf["stake"],
                                                 errors="coerce").values
                            _aw = _armf["win"].astype(bool).values
                            _feef = (_stk / np.clip(_cf, 0.01, None)) \
                                * 0.07 * _apm * (1 - _apm)
                            _pf = np.where(
                                _cf >= 0.99, 0.0,
                                np.where(_aw, _stk * (1 - _cf) / _cf,
                                         -_stk) - _feef)
                            for _dtv, _pv in zip(_armf["dt"], _pf):
                                _ft_rows.append((_dtv, _pv))
                        for _nm, _rw, _clr2, _dsh2 in (
                                ("‹mon› DUALv2 barbell (≤.20|≥.70)",
                                 _bb_rows, "#e2586e", "dash"),
                                ("DUAL v2+KV (fill-true live px)",
                                 _ft_rows, "#00c076", "solid")):
                            _Px = pd.DataFrame(
                                _rw, columns=["dt", "pnl"]).sort_values("dt")
                            if not len(_Px):
                                continue
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_Px["dt"]),
                                y=[0.0] + list(_Px["pnl"].cumsum()),
                                name=_nm,
                                line=dict(color=_clr2, width=1.5,
                                          dash=_dsh2)))
                            _cx = _Px["pnl"].cumsum()
                            _rows15.append({
                                "book": _nm,
                                "net": f"${_Px['pnl'].sum():+,.0f}",
                                "n": len(_Px), "WR/BE": "—",
                                "maxDD": f"${float((_cx.cummax() - _cx).max()):,.0f}",
                            })
                    except Exception:
                        pass
                    # [2026-08-14] DUAL v3c (user call, takes the blend's
                    # seat): 12-gate g+k arm sized by xMFconv (mkt-fav
                    # conviction tiers) + shadow arm sized by xHdamp
                    # (SOL-hurst). Window: +$5,494 / S 0.99 / DD $765 vs
                    # v2's +$8,749 / 0.77 / $948 — the risk-first
                    # composition and the registered GO-LIVE shape (halves
                    # net for ~double Sharpe). Races v2 on the tab; paper
                    # adoption decisions at the 08-18 read.
                    if ("pnl_mf" in getattr(_prod_gk, "columns", [])
                            and "pnl_h" in getattr(_shad_book, "columns", [])):
                        _rows_d3 = ([(rr["dt"], rr["pnl_mf"])
                                     for _, rr in _prod_gk.iterrows()]
                                    + [(rr["dt"], rr["pnl_h"])
                                       for _, rr in _shad_book.iterrows()])
                        _Pd3 = pd.DataFrame(_rows_d3, columns=["dt", "pnl"]
                                            ).sort_values("dt")
                        if len(_Pd3):
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_Pd3["dt"]),
                                y=[0.0] + list(_Pd3["pnl"].cumsum()),
                                name="DUAL v3c (MFconv + shadow xHdamp)",
                                line=dict(color="#b57edc", width=2,
                                          dash="dash")))
                            _cumd3 = _Pd3["pnl"].cumsum()
                            _rows15.append({
                                "book": "DUAL v3c (MFconv + shadow xHdamp)",
                                "net": f"${_Pd3['pnl'].sum():+,.0f}",
                                "n": len(_Pd3),
                                "WR/BE": "—",
                                "maxDD": f"${float((_cumd3.cummax() - _cumd3).max()):,.0f}",
                            })
                except Exception:
                    pass
            # [2026-08-15] legend separated from the plot: with 9+ books the
            # horizontal legend wraps to 3-4 rows and was spilling onto the
            # traces (t=48 fit one row). Rows grow upward from y=1.02 into
            # a 150px top margin; height raised so the plot area is
            # unchanged.
            _fig15.update_layout(height=430, margin=dict(l=0, r=0, t=150, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0,
                                             font=dict(size=10)))
            st.plotly_chart(_fig15, use_container_width=True)
            st.dataframe(pd.DataFrame(_rows15), hide_index=True,
                         use_container_width=True)
            _dis15 = _ab.dropna(subset=["p_gbdt", "p_model_15m"])
            _dis15 = _dis15[(_dis15["p_gbdt"] - _dis15["p_model_15m"]).abs() >= 0.05]
            if len(_dis15):
                _sr15 = np.mean(
                    np.where(_dis15["p_gbdt"] > _dis15["p_model_15m"],
                             _dis15["resolved_yes"] == 1,
                             _dis15["resolved_yes"] == 0))
                st.caption(f"Model disagreements ≥5pp: {len(_dis15)} scans — "
                           f"shadow's side settled correct {_sr15:.0%} of the "
                           f"time.")
    except Exception as _abex:
        st.warning(f"15m A/B tab error: {_abex}")

with tab_sol_hourly_ab:
    # [2026-07-31] HOURLY model-challenger A/B, all three assets. Challenger
    # books are real standalone runner CSVs (flat $100 net of fees,
    # pre-registered YES pm[.20,.80] edge>=.05 primary; NO secondary;
    # ctx_gates tagged at booking). Production hourly trades flat-$100-
    # normalized for comparability. Dashed = challenger behind transferable
    # production context gates. First reads ~08-13/14; decisions late-Aug.
    st.markdown(
        "<div style='color:#f0a500;font-size:0.78rem;margin-bottom:16px;'>"
        "Hourly model challengers vs production, per asset — flat $100/contract, "
        "net of fees, challenger PRIMARY = YES side (pre-registered). Dashed "
        "lines = challenger trades passing the transferable production context "
        "gates. SOL clock starts 07-30; BTC/ETH 07-31. Decisions late-Aug."
        "</div>", unsafe_allow_html=True)
    # [2026-08-04] BTC/ETH challengers switched bookdyn -> vol-tail books
    # (user decision; bookdyn rides to its 08-13/14 read off-dashboard, CSVs
    # keep accruing). 4th tuple field = book kind: "dir" (YES-primary
    # directional w/ ctx gates) or "tail" (two-sided vol-tail legs, ask
    # fills, no gates — cumulative over ALL legs, events counted).
    _AB_CFG = [
        # [2026-08-06] v7/v8 removed from DISPLAY (user: nowhere near
        # deployable). Runners + CSVs keep accruing; their 08-13/14
        # retirement read happens off-dashboard.
        # [2026-08-19 per user: SOL fav + rescues books added as CHALLENGER
        # LINES only — seat and cutoff unchanged (production isn't failing);
        # promotion question deferred to the ~09-01 read. Books are
        # live-fill accounting from birth (no era mixing).]
        ("SOL", "2026-07-30", RESULTS_DIR / "paper_trades_sol.csv",
         [("fav (YES .80-.97, live-fill)",
           RESULTS_DIR / "paper_trades_sol_hourly_fav.csv", "#f0a500", "dir"),
          ("fav-rescues B/C/M (live-fill)",
           RESULTS_DIR / "paper_trades_sol_hourly_fav_rescues.csv",
           "#00c076", "mixed")]),
        # [2026-08-14] niche v1/v2 books ADDED TO DISPLAY (user believed the
        # hourly seat was empty — the live niche challengers were never on
        # this tab; both forward-positive since 07-28: v1 +$1,506/81 all
        # weeks green, v2 +$490/133 with wk33 +$1,761 its best). Their
        # 08-11 v1-vs-v2 review rides forward with visible books now.
        # [2026-08-14] BTC/ETH windows moved back 08-04 -> 07-28 (the niche
        # books' true forward start): the 08-04 start (set for voltail
        # go-live) was amputating niche v2's -$1,291 first week — tab showed
        # +$1,802 vs the real +$490 (user caught it). Voltail unaffected
        # (no rows before 08-04).
        ("BTC", "2026-07-28", RESULTS_DIR / "paper_trades.csv",
         # [2026-08-20 per user] vol-tail line REPLACED by the niche
         # REFRESH challenger (frozen niche silent since 08-13; refresh
         # 6/6 seeds holdout-positive, wf-July caveat, live-fill book,
         # read ~09-03). Voltail RUNNER untouched — CSV accrues, its
         # ~08-22 pre-registered read happens off-dashboard (v7/v8
         # precedent).
         [("niche REFRESH (live-fill)",
           RESULTS_DIR / "paper_trades_btc_hourly_niche_refresh.csv",
           "#b57edc", "dir"),
          ("niche v1 ★PAPER (promoted 08-14)", RESULTS_DIR / "paper_trades_btc_hourly_niche.csv",
           "#f0a500", "dir"),
          ("niche v2", RESULTS_DIR / "paper_trades_btc_hourly_niche_v2.csv",
           "#00c076", "dir")]),
        ("ETH", "2026-07-28", RESULTS_DIR / "paper_trades_eth.csv",
         [("vol-tail", RESULTS_DIR / "paper_trades_eth_hourly_voltail.csv",
           "#b57edc", "tail"),
          ("niche v2", RESULTS_DIR / "paper_trades_eth_hourly_niche_v2.csv",
           "#00c076", "dir"),
          # [2026-08-19] fav book promoted to the ETH hourly paper seat
          # (see load_trades); shown here fee-net from its 08-18 launch.
          ("fav ★PAPER (promoted 08-19)",
           RESULTS_DIR / "paper_trades_eth_hourly_fav.csv",
           "#f0a500", "dir")]),
    ]
    def _maxdd(_pnls):
        # peak-to-trough drawdown of the cumulative flat-$100 curve
        _c = _pnls.cumsum()
        return float((_c.cummax() - _c).max()) if len(_c) else 0.0

    for _aname, _astart, _prod_csv, _chals in _AB_CFG:
        st.markdown(f"<div style='font-size:1.0rem;font-weight:700;color:#fff;"
                    f"margin:14px 0 6px 0;'>{_aname} hourly — since {_astart}"
                    "</div>", unsafe_allow_html=True)
        try:
            _abst = pd.Timestamp(_astart, tz="UTC")
            _figab = go.Figure()
            _hrows = []   # [2026-08-23] full stats table (user: no hover/cram)
            _pr = pd.read_csv(_prod_csv, low_memory=False)
            _pr["dt"] = pd.to_datetime(_pr["logged_at"], errors="coerce", utc=True,
                                       format="mixed")
            _pr = _pr[(_pr["decision"] == "trade") & (_pr["dt"] >= _abst)].copy()
            _pr["p_market"] = pd.to_numeric(_pr["p_market"], errors="coerce")
            _pr = _pr.dropna(subset=["p_market", "would_win"])
            if len(_pr):
                _prc = np.where(_pr["side"] == "yes", _pr["p_market"],
                                1 - _pr["p_market"])
                _prw = _pr["would_win"].astype(str).str.lower().isin(
                    ["true", "1", "1.0"])
                _prf = 0.07 * _pr["p_market"] * (1 - _pr["p_market"])
                _pr["pnl"] = np.where(_prw, 100 * (1 - _prc) / _prc, -100.0) \
                    - (100 / _prc) * _prf
                _pq = _pr.sort_values("dt")
                _figab.add_trace(go.Scatter(x=_pq["dt"], y=_pq["pnl"].cumsum(),
                                            name="production",
                                            line=dict(color="#4f8bf9", width=2)))
                # [2026-08-06] SOL: minimal frozen gate — block trades when
                # daily Markov = Sideways (hourly directional edge needs a
                # trending daily regime). Only survivor of 6 pre-declared
                # frozen candidates on the 78-trade history: removes -$1,041
                # over 48 trades, 4/5 weeks helpful, P(>=0)=0.07; flips the
                # full book -$562 -> +$479. Kelly-logged sizing tested WORSE
                # than flat (-$795 vs -$562) — shown as a number, not a line.
                if _aname == "SOL":
                    _mkvd = _pq["markov_regime_daily"].astype(str)
                    _gq = _pq[_mkvd != "Sideways"]
                    if len(_gq):
                        _figab.add_trace(go.Scatter(
                            x=_gq["dt"], y=_gq["pnl"].cumsum(),
                            name="production mkv-gated",
                            line=dict(color="#4f8bf9", width=1.5, dash="dash")))
                        _hrows.append({"book": "production mkv-gated",
                            "net": f"${_gq['pnl'].sum():+,.0f}",
                            "n": len(_gq), "WR/BE": "—",
                            "maxDD": f"${_maxdd(_gq['pnl']):,.0f}",
                            "pending": ""})
                # [2026-08-06] BTC: same 6 frozen candidates replayed on the
                # 645-trade book (07-01+). Sole survivor = cpu-disagree block
                # (side contradicts composite_p_up): removes -$1,240, P>=0
                # 0.05 — the correlated-agreement doctrine as a gate. MIRROR
                # of SOL, where this same gate removed winners. markov-daily
                # candidates never fired: BTC daily regime = Bull for 609/645
                # trades (monoculture, not a data gap). hurst adds ~$123 on
                # top — not worth 151 fewer trades. No gate rescues the book
                # to positive (-$6,342 -> -$5,102 full): the July-era model's
                # bleed is broad, per the fee-audit inversion finding.
                # [2026-08-06] ETH: frozen-6 replay — FOUR candidates clear
                # (richest of the 3 assets). Minimal displayed set = the two
                # independently strong ones: NO-block at oi_chg_pct>=0.0535
                # (-$1,114/67, P=0.02, negative EVERY week; threshold is the
                # frozen SOL 15m constant replicating cross-asset AND cross-
                # timeframe) + markov_daily==Sideways block (-$807/42, P=0.04
                # — the SOL hourly finding replicating on ETH). cpu/hurst
                # also clear individually but stack poorly (3-gate book's
                # "positive" total leans on one week; 4-gate worse). Kelly
                # note: ETH kelly-logged BEATS flat (-$1,175 vs -$1,770) —
                # opposite of SOL hourly; dampener port likely helping.
                if _aname == "ETH":
                    _oie = pd.to_numeric(_pq["oi_chg_pct"], errors="coerce")
                    _bad = ((_pq["side"] == "no")
                            & (_oie >= 0.0535).fillna(False))                         | (_pq["markov_regime_daily"].astype(str)
                           == "Sideways")
                    _gq = _pq[~np.asarray(_bad, bool)]
                    if len(_gq):
                        _figab.add_trace(go.Scatter(
                            x=_gq["dt"], y=_gq["pnl"].cumsum(),
                            name="production oi+mkv-gated",
                            line=dict(color="#4f8bf9", width=1.5, dash="dash")))
                        _hrows.append({"book": "production oi+mkv-gated",
                            "net": f"${_gq['pnl'].sum():+,.0f}",
                            "n": len(_gq), "WR/BE": "—",
                            "maxDD": f"${_maxdd(_gq['pnl']):,.0f}",
                            "pending": ""})
                if _aname == "BTC":
                    _cpub = pd.to_numeric(_pq["composite_p_up"],
                                          errors="coerce")
                    _agree = np.where(_pq["side"] == "yes",
                                      ~(_cpub < 0.5).fillna(False),
                                      ~(_cpub > 0.5).fillna(False))
                    _gq = _pq[np.asarray(_agree, bool)]
                    if len(_gq):
                        _figab.add_trace(go.Scatter(
                            x=_gq["dt"], y=_gq["pnl"].cumsum(),
                            name="production cpu-gated",
                            line=dict(color="#4f8bf9", width=1.5, dash="dash")))
                        _hrows.append({"book": "production cpu-gated",
                            "net": f"${_gq['pnl'].sum():+,.0f}",
                            "n": len(_gq), "WR/BE": "—",
                            "maxDD": f"${_maxdd(_gq['pnl']):,.0f}",
                            "pending": ""})
                    # [2026-08-22] btc_mkv_sideways_gate RETRO line (user
                    # call): gate wired to the production paper runner +
                    # both niche runners same day (cross-asset frozen
                    # rule, BTC-confirmed by the 08-22 niche run). Shown
                    # retroactively over the full window per user request.
                    _mkvb = _pq["markov_regime_daily"].astype(str)
                    _gq2 = _pq[_mkvb != "Sideways"]
                    if len(_gq2):
                        _figab.add_trace(go.Scatter(
                            x=_gq2["dt"], y=_gq2["pnl"].cumsum(),
                            name="production mkv-gated ★LIVE-RULE",
                            line=dict(color="#4f8bf9", width=1.5,
                                      dash="dot")))
                        _hrows.append({"book": "production mkv-gated ★LIVE-RULE",
                            "net": f"${_gq2['pnl'].sum():+,.0f}",
                            "n": len(_gq2), "WR/BE": "—",
                            "maxDD": f"${_maxdd(_gq2['pnl']):,.0f}",
                            "pending": ""})
                _hrows.insert(0, {"book": "production",
                    "net": f"${_pq['pnl'].sum():+,.0f}", "n": len(_pq),
                    "WR/BE": f"{_prw.mean():.0%}/{np.mean(_prc):.0%}",
                    "maxDD": f"${_maxdd(_pq['pnl']):,.0f}", "pending": ""})
            else:
                _hrows.insert(0, {"book": "production", "net": "—", "n": 0,
                                  "WR/BE": "collecting…", "maxDD": "",
                                  "pending": ""})
            for _ci, (_lbl, _path, _clr, _kind) in enumerate(_chals, start=1):
                _b = pd.read_csv(_path, low_memory=False)
                # [2026-08-19] live-fill books: filled=0 rows are recorded
                # no-fill attempts (zero PnL), not trades — exclude.
                if "filled" in _b.columns:
                    _b = _b[pd.to_numeric(_b["filled"], errors="coerce") == 1]
                _b["dt"] = pd.to_datetime(_b["logged_at"], errors="coerce", utc=True, format="mixed")
                for _c in ["p_market", "would_pnl_net"]:
                    if _c in _b.columns:
                        _b[_c] = pd.to_numeric(_b[_c], errors="coerce")
                    else:
                        _b[_c] = np.nan
                _b = _b[_b["dt"] >= _abst]
                if _kind == "tail":
                    _res = _b[_b["would_pnl_net"].notna()]
                elif _kind == "mixed":
                    # [2026-08-19] fav-rescue books log YES and NO bands —
                    # both sides count (the plain side filter below is the
                    # v7/v8-era YES-primary convention, wrong here).
                    _res = _b[_b["would_pnl_net"].notna()]
                elif "side" in _b.columns:
                    _res = _b[(_b["side"] == "yes") & _b["would_pnl_net"].notna()]
                else:
                    # niche books are YES-only by design (no side column)
                    _res = _b[_b["would_pnl_net"].notna()]
                _pend = int(_b["would_pnl_net"].isna().sum())
                if len(_res):
                    _rq = _res.sort_values("dt")
                    _figab.add_trace(go.Scatter(
                        x=_rq["dt"], y=_rq["would_pnl_net"].cumsum(), name=_lbl,
                        line=dict(color=_clr, width=2)))
                    # [2026-08-14] niche v1 xHdamp(sol): the FROZEN
                    # cross-asset hurst sizing (same params as the SOL/ETH
                    # 15m deploys) as a racing overlay. Chop trades
                    # (solH<0.5) went 1/7 (-$498, p=0.011); modulator
                    # +$1,506 -> +$1,892 with zero trades dropped. The
                    # niche needs a trending tape — coheres with its
                    # buy-the-fear thesis. Display-only variant; the
                    # runner is untouched.
                    if _aname == "BTC" and _lbl.startswith("niche v1"):
                        try:
                            _shn = pd.read_csv(
                                ASSET_CSV_15M["SOL"],
                                usecols=["logged_at", "hurst_exponent_5m"],
                                low_memory=False)
                            _shn["dt"] = pd.to_datetime(
                                _shn["logged_at"], errors="coerce",
                                utc=True, format="mixed")
                            _shn["h"] = pd.to_numeric(
                                _shn["hurst_exponent_5m"], errors="coerce")
                            _shn = _shn.dropna(
                                subset=["dt", "h"]).sort_values("dt")
                            _nts = _shn["dt"].astype("int64").values / 1e9
                            _qts = _rq["dt"].astype("int64").values / 1e9
                            _ni = np.searchsorted(_nts, _qts,
                                                  side="right") - 1
                            _nh = np.where(
                                _ni >= 0,
                                _shn["h"].values[np.clip(_ni, 0, None)],
                                np.nan)
                            _nage = _qts - np.where(
                                _ni >= 0, _nts[np.clip(_ni, 0, None)],
                                np.nan)
                            _nh = np.where(_nage <= 1800, _nh, np.nan)
                            _nmul = pd.Series(
                                np.clip((_nh - 0.4) / 0.2, 0.25, 1.0),
                                index=_rq.index).fillna(1.0)
                            _hp = _rq["would_pnl_net"] * _nmul
                            _figab.add_trace(go.Scatter(
                                x=_rq["dt"], y=_hp.cumsum(),
                                name="niche v1 xHdamp(sol)",
                                line=dict(color=_clr, width=1.5,
                                          dash="dash")))
                        except Exception:
                            pass
                    _wr = float(pd.to_numeric(_rq["would_win"],
                                              errors="coerce").mean())
                    if _kind == "mixed" and "side" in _rq.columns:
                        # side-aware breakeven: NO trades cost 1 - p_market
                        _be = float(np.where(_rq["side"] == "no",
                                             1 - _rq["p_market"],
                                             _rq["p_market"]).mean())
                    else:
                        _be = (float(_rq["p_market"].mean())
                               if _rq["p_market"].notna().any()
                               else float(pd.to_numeric(_rq.get("cost_ask"),
                                                        errors="coerce").mean()))
                    _gtxt = ""
                    if _kind == "tail":
                        # [2026-08-06] mean-cost "BE" is misleading when leg
                        # costs vary (3-15c): wins at 14c pay 6:1, wins at 3c
                        # pay 32:1, so avg WR vs avg cost is not a valid
                        # comparison. Show the realized-payout breakeven
                        # instead: avg_loss/(avg_win+avg_loss).
                        _wl = _rq[pd.to_numeric(_rq["would_win"],
                                                errors="coerce") == 1]
                        _ll = _rq[pd.to_numeric(_rq["would_win"],
                                                errors="coerce") == 0]
                        if len(_wl) and len(_ll):
                            _aw = float(_wl["would_pnl_net"].mean())
                            _al = float(-_ll["would_pnl_net"].mean())
                            _be = _al / (_aw + _al)
                        _gtxt = f"{_rq['event'].nunique()} events"
                    elif "ctx_gates" in _rq.columns:
                        _gt = _rq[_rq["ctx_gates"].fillna("") == ""]
                        if len(_gt):
                            _figab.add_trace(go.Scatter(
                                x=_gt["dt"], y=_gt["would_pnl_net"].cumsum(),
                                name=f"{_lbl} gated",
                                line=dict(color=_clr, width=1.5, dash="dash")))
                        _hrows.append({"book": f"{_lbl} gated",
                            "net": f"${_gt['would_pnl_net'].sum():+,.0f}",
                            "n": len(_gt), "WR/BE": "—",
                            "maxDD": f"${_maxdd(_gt['would_pnl_net']):,.0f}",
                            "pending": ""})
                        _gtxt = ""
                    # [2026-08-22] books carrying markov_daily_regime (niche
                    # v1 + refresh, backfilled) get the RETRO mkv-gated line
                    # — the rule now live in their runners, shown over full
                    # history per user request.
                    if "markov_daily_regime" in _rq.columns:
                        _gm = _rq[_rq["markov_daily_regime"].astype(str)
                                  != "Sideways"]
                        if len(_gm):
                            _figab.add_trace(go.Scatter(
                                x=_gm["dt"],
                                y=_gm["would_pnl_net"].cumsum(),
                                name=f"{_lbl} mkv-gated ★LIVE-RULE",
                                line=dict(color=_clr, width=1.5,
                                          dash="dot")))
                            _hrows.append({
                                "book": f"{_lbl} mkv-gated ★LIVE-RULE",
                                "net": f"${_gm['would_pnl_net'].sum():+,.0f}",
                                "n": len(_gm), "WR/BE": "—",
                                "maxDD": f"${_maxdd(_gm['would_pnl_net']):,.0f}",
                                "pending": ""})
                    _hrows.append({"book": _lbl,
                        "net": f"${_rq['would_pnl_net'].sum():+,.0f}",
                        "n": len(_rq),
                        "WR/BE": f"{_wr:.0%}/{_be:.0%}" + (f" · {_gtxt}"
                                                           if _gtxt else ""),
                        "maxDD": f"${_maxdd(_rq['would_pnl_net']):,.0f}",
                        "pending": _pend})
                else:
                    _hrows.append({"book": _lbl, "net": "—", "n": 0,
                                   "WR/BE": "collecting…", "maxDD": "",
                                   "pending": _pend})
            _figab.update_layout(height=260, margin=dict(l=0, r=0, t=8, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0))
            st.plotly_chart(_figab, use_container_width=True)
            st.dataframe(pd.DataFrame(_hrows), hide_index=True,
                         use_container_width=True,
                         height=38 * (len(_hrows) + 1) + 8)
        except Exception as _abex:
            st.warning(f"{_aname} hourly A/B error: {_abex}")

with tab_cmp:
    st.markdown(
        "<div style='color:#888;font-size:0.78rem;margin-bottom:16px;'>"
        "Indicator alignment win rates — side by side across all three assets."
        "</div>",
        unsafe_allow_html=True,
    )
    col_btc, col_eth, col_sol = st.columns(3)

    for col_ui, asset in [(col_btc, "BTC"), (col_eth, "ETH"), (col_sol, "SOL")]:
        with col_ui:
            symbol = {"BTC": "₿", "ETH": "Ξ", "SOL": "◎"}[asset]
            st.markdown(f"<div style='font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:12px;'>{symbol} {asset}</div>", unsafe_allow_html=True)
            df_a = load_trades(asset)
            if df_a.empty:
                st.markdown("<p style='color:#555;font-size:0.8rem;'>No data yet.</p>", unsafe_allow_html=True)
                continue
            cutoff_1h_a  = pd.Timestamp(ASSET_DISPLAY_FROM.get(asset, DISPLAY_FROM), tz="UTC").tz_convert("America/Los_Angeles")
            cutoff_15m_a = pd.Timestamp(ASSET_DISPLAY_FROM_15M.get(asset, ASSET_DISPLAY_FROM.get(asset, DISPLAY_FROM)), tz="UTC").tz_convert("America/Los_Angeles")
            df_a = df_a[
                ((df_a["timeframe"] == "15m") & (df_a["logged_at"] >= cutoff_15m_a)) |
                ((df_a["timeframe"] != "15m") & (df_a["logged_at"] >= cutoff_1h_a))
            ]
            trades_a   = df_a[(df_a["decision"] == "trade") & (df_a["contract_ticker"].fillna("").str.strip() != "")]
            resolved_a = trades_a.dropna(subset=["would_win"]).copy()
            if not resolved_a.empty:
                resolved_a["would_win_bool"] = resolved_a["would_win"].apply(_parse_win)
                resolved_a["would_pnl_num"]  = pd.to_numeric(resolved_a["would_pnl"], errors="coerce")
                wr  = resolved_a["would_win_bool"].mean()
                pnl = resolved_a["would_pnl_num"].sum()
                wr_color  = "#00c076" if wr  >= 0.5 else "#ff4d4d"
                pnl_color = "#00c076" if pnl >= 0   else "#ff4d4d"
                st.markdown(
                    f"<div style='font-size:0.75rem;color:#666;margin-bottom:8px;'>"
                    f"<span style='color:{wr_color};font-weight:700;'>{wr:.1%} WR</span>"
                    f" &nbsp;|&nbsp; "
                    f"<span style='color:{pnl_color};font-weight:700;'>${pnl:+,.2f}</span>"
                    f" &nbsp;|&nbsp; {len(resolved_a)} resolved"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            render_indicator_stats(resolved_a)
