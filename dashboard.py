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
    "BTC": "2026-07-28 22:15:13",
    "ETH": "2026-07-28 22:15:13",
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
    "BTC": "2026-07-28 06:53:02",  # cleared 2026-07-28 per user request — fee-audit gate
                                    # package live (no_midhigh/no_deep/yes_thrust gates +
                                    # fee-aware Kelly). Prior: 2026-06-29 23:31:53.
    "ETH": "2026-07-29 18:25:54",  # cleared 2026-07-29 per user request — slope-feature
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
    "SOL": "2026-07-28 06:18:03",  # cleared 2026-07-28 per user request — persistence YES
                                    # gate + NO band gates + fee-aware Kelly live (restart
                                    # 07-28 06:18 UTC). Prior: 2026-06-18 06:18:28.
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
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
""", unsafe_allow_html=True)


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
            df_15m = df_15m.drop(columns=["_ts", "_live", "_session"])

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
  Updated: {now_str} &nbsp;|&nbsp; Auto-refresh: {REFRESH_SECONDS}s
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
        "SOL 15m model A/B on identical live scans since 2026-07-30 01:00 UTC — "
        "production (iso+z-expansion, blue) vs slope-shadow (p_gbdt, orange) vs "
        "fixed 50/50 blend (purple). Hypothetical books; decisions remain "
        "production-only. Promotion decision at the 08-11 review."
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
            "- **gk vrM+hurst** (dash-dot): conditional markov (applies only "
            "when vol_ratio_1h < 1) PLUS the hurst NO-block. Replaced plain "
            "gk vrM 08-06: the stack dominates it on every window/book "
            "(long S 0.37 vs 0.30, DD $2,882 vs $4,794) though hurst alone "
            "still leads Sharpe. Inherits both fitted thresholds — scored on "
            "trades after 08-05 15:20 UTC only.\n"
            "- **gk vrM+zd65** (long-dash-dot): both modifications combined — "
            "the strongest replay (S 0.57/0.59/0.29, lowest DD) and the third "
            "candidate stack racing forward. Same 08-05+ scoring rule.\n"
            "- **gk hurst** (long-dash): base stack + NO blocked when "
            "hurst_exponent_5m ≥ 0.61 (trending tape kills mean-reversion "
            "NO). Survived the full-column sweep AND the ex-drawdown test "
            "(all 4 pre-drawdown weeks helped; long-window S 0.14→0.48, DD "
            "halved). Races AS A COMPETING stack — it is redundant stacked "
            "on vrM+zd65. Scored on trades after 08-05 15:20 UTC only.")
    _SH_START = pd.Timestamp("2026-07-30 01:00", tz="UTC")
    try:
        _shp = pd.read_csv(ASSET_CSV_15M["SOL"], low_memory=False)
        _shp["dt"] = pd.to_datetime(_shp["logged_at"], errors="coerce", utc=True, format="mixed")
        for _c in ["p_market", "p_model_15m", "p_gbdt", "resolved_yes",
                   "sol_persist_score", "slope120_stoch_k_15m",
                   "stoch_cross_1h", "stoch_k_1h", "oi_chg_pct",
                   "offset_pct", "z_drift_6h", "vol_ratio_1h",
                   "hurst_exponent_5m"]:
            _shp[_c] = pd.to_numeric(_shp[_c], errors="coerce")
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
            _fams = st.multiselect(
                "Lines on chart",
                ["flat $100", "gated+kelly", "gk zd65", "gk vrM+hurst",
                 "gk vrM+zd65", "gk hurst"],
                default=["gated+kelly", "gk vrM+zd65", "gk hurst"],
                key="solsh_fams")
            _figsh = go.Figure()
            _rows = []

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
            for _lbl, _col, _clr in [("production", "p_model_15m", "#4f8bf9"),
                                     ("shadow", "p_gbdt", "#f0a500"),
                                     ("blend 50/50", "p_blend", "#b57edc")]:
                _s = _sh.dropna(subset=[_col]).copy()
                _fee = 0.07 * _s["p_market"] * (1 - _s["p_market"])
                _ey = _s[_col] - _s["p_market"] - _fee
                _en = _s["p_market"] - _s[_col] - _fee
                _s["side"] = np.where(_ey >= _en, "yes", "no")
                _s["edge"] = np.maximum(_ey, _en)
                _q = _s[_s["edge"] >= 0.04].sort_values("dt").drop_duplicates(
                    "contract_ticker", keep="first")
                _cost = np.where(_q["side"] == "yes", _q["p_market"], 1 - _q["p_market"])
                _win = np.where(_q["side"] == "yes", _q["resolved_yes"] == 1,
                                _q["resolved_yes"] == 0)
                _feeq = 0.07 * _q["p_market"] * (1 - _q["p_market"])
                _pnl = pd.Series(np.where(_win, 100 * (1 - _cost) / _cost, -100)
                                 - (100 / _cost) * _feeq, index=_q.index)
                if "flat $100" in _fams:
                    _figsh.add_trace(go.Scatter(
                        x=_q["dt"], y=_pnl.cumsum(), name=f"{_lbl} flat",
                        line=dict(color=_clr, width=2)))
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
                        ("gated+kelly", _v2_ok & _mkv_ok & _zd_ok & _off_ok,
                         "dash", 1.5),
                        ("gk zd65", _v2_ok & _mkv_ok & _zd_ok65 & _off_ok,
                         "dot", 1.0),
                        ("gk vrM+hurst", _v2_ok & _mkv_vr & _zd_ok & _off_ok
                         & _hu_ok, "dashdot", 1.0),
                        ("gk vrM+zd65", _v2_ok & _mkv_vr & _zd_ok65 & _off_ok,
                         "longdashdot", 1.0),
                        ("gk hurst", _v2_ok & _mkv_ok & _zd_ok & _off_ok & _hu_ok,
                         "longdash", 1.0),
                        ("gk combo+damp", _v2_ok & _mkv_vr & _zd_ok65 & _off_ok,
                         "dot", 1.0)]:
                    _vq = _q[_vmask].copy()
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
                    if len(_vq) and _vn in _fams:
                        _figsh.add_trace(go.Scatter(
                            x=_vq["dt"], y=_vp.cumsum(), name=f"{_lbl} {_vn}",
                            line=dict(color=_clr, width=_vw, dash=_vdash)))
                    _r[_vn] = (f"${_vp.sum():+,.0f} (n={len(_vq)}, "
                               f"DD ${_vdd:,.0f})")
                _rows.append(_r)
            _figsh.update_layout(height=320, margin=dict(l=0, r=0, t=64, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0))
            st.plotly_chart(_figsh, use_container_width=True)
            st.dataframe(pd.DataFrame(_rows), hide_index=True,
                         use_container_width=True)
            _dis = _sh.dropna(subset=["p_gbdt", "p_model_15m"])
            _dis = _dis[(_dis["p_gbdt"] - _dis["p_model_15m"]).abs() >= 0.05]
            if len(_dis):
                _shadow_right = np.mean(
                    np.where(_dis["p_gbdt"] > _dis["p_model_15m"],
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
        for _c in ["p_market", "p_model_15m", "p_gbdt", "resolved_yes",
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
            _ab["p_blend"] = (_ab["p_model_15m"] + _ab["p_gbdt"]) / 2
            _books15 = [("production", "p_model_15m", "#4f8bf9"),
                        ("shadow", "p_gbdt", "#f0a500"),
                        ("blend 50/50", "p_blend", "#b57edc")]
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
                    for _vnE, _mE, _dshE in [
                            ("g+k (5 gates)", _okE, "dash"),
                            ("g+k +hurst", _okE & _hu_okE, "dot")]:
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
                        if len(_gkE):
                            _fig15.add_trace(go.Scatter(
                                x=[_cfg15["start"]] + list(_gkE["dt"]),
                                y=[0.0] + list(_gpE.cumsum()),
                                name=f"{_lbl} {_vnE}",
                                line=dict(color=_clr, width=1.5,
                                          dash=_dshE)))
                        _gcumE = _gpE.cumsum()
                        _gddE = (float((_gcumE.cummax() - _gcumE).max())
                                 if len(_gkE) else 0.0)
                        _row15[_vnE] = (
                            f"${_gpE.sum():+,.0f} (n={len(_gkE)}, "
                            f"DD ${_gddE:,.0f})")
                _rows15.append(_row15)
            _fig15.update_layout(height=320, margin=dict(l=0, r=0, t=48, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0))
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
        ("SOL", "2026-07-30", RESULTS_DIR / "paper_trades_sol.csv", []),
        ("BTC", "2026-08-04", RESULTS_DIR / "paper_trades.csv",
         [("vol-tail", RESULTS_DIR / "paper_trades_btc_hourly_voltail.csv",
           "#b57edc", "tail")]),
        ("ETH", "2026-08-04", RESULTS_DIR / "paper_trades_eth.csv",
         [("vol-tail", RESULTS_DIR / "paper_trades_eth_hourly_voltail.csv",
           "#b57edc", "tail")]),
    ]
    for _aname, _astart, _prod_csv, _chals in _AB_CFG:
        st.markdown(f"<div style='font-size:1.0rem;font-weight:700;color:#fff;"
                    f"margin:14px 0 6px 0;'>{_aname} hourly — since {_astart}"
                    "</div>", unsafe_allow_html=True)
        try:
            _abst = pd.Timestamp(_astart, tz="UTC")
            _figab = go.Figure()
            _cols = st.columns(1 + len(_chals))
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
                _gk_txt = ""
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
                        _gk_txt = (f" · mkv-gated ${_gq['pnl'].sum():+,.0f} "
                                   f"(n={len(_gq)})")
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
                        _gk_txt = (f" · oi+mkv-gated ${_gq['pnl'].sum():+,.0f} "
                                   f"(n={len(_gq)})")
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
                        _gk_txt = (f" · cpu-gated ${_gq['pnl'].sum():+,.0f} "
                                   f"(n={len(_gq)})")
                _cols[0].metric("production: net (flat $100)",
                                f"${_pq['pnl'].sum():+,.0f}",
                                f"{len(_pq)} resolved · WR {_prw.mean():.0%} "
                                f"vs BE {np.mean(_prc):.0%}{_gk_txt}")
            else:
                _cols[0].metric("production: net (flat $100)", "—", "collecting…")
            for _ci, (_lbl, _path, _clr, _kind) in enumerate(_chals, start=1):
                _b = pd.read_csv(_path, low_memory=False)
                _b["dt"] = pd.to_datetime(_b["logged_at"], errors="coerce", utc=True, format="mixed")
                for _c in ["p_market", "would_pnl_net"]:
                    if _c in _b.columns:
                        _b[_c] = pd.to_numeric(_b[_c], errors="coerce")
                    else:
                        _b[_c] = np.nan
                _b = _b[_b["dt"] >= _abst]
                if _kind == "tail":
                    _res = _b[_b["would_pnl_net"].notna()]
                else:
                    _res = _b[(_b["side"] == "yes") & _b["would_pnl_net"].notna()]
                _pend = int(_b["would_pnl_net"].isna().sum())
                if len(_res):
                    _rq = _res.sort_values("dt")
                    _figab.add_trace(go.Scatter(
                        x=_rq["dt"], y=_rq["would_pnl_net"].cumsum(), name=_lbl,
                        line=dict(color=_clr, width=2)))
                    _wr = float(pd.to_numeric(_rq["would_win"],
                                              errors="coerce").mean())
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
                            _gtxt = (f" (realized-payout BE; avg win "
                                     f"${_aw:,.0f} / loss ${_al:,.0f})"
                                     f" · {_rq['event'].nunique()} events")
                        else:
                            _gtxt = (f" (avg cost — no wins yet) · "
                                     f"{_rq['event'].nunique()} events")
                    elif "ctx_gates" in _rq.columns:
                        _gt = _rq[_rq["ctx_gates"].fillna("") == ""]
                        if len(_gt):
                            _figab.add_trace(go.Scatter(
                                x=_gt["dt"], y=_gt["would_pnl_net"].cumsum(),
                                name=f"{_lbl} gated",
                                line=dict(color=_clr, width=1.5, dash="dash")))
                        _gtxt = (f" · gated ${_gt['would_pnl_net'].sum():+,.0f} "
                                 f"(n={len(_gt)})")
                    _cols[_ci].metric(f"{_lbl}: net (flat $100)",
                                      f"${_rq['would_pnl_net'].sum():+,.0f}",
                                      f"{len(_rq)} resolved · WR {_wr:.0%} vs "
                                      f"BE {_be:.0%}{_gtxt} · {_pend} pending")
                else:
                    _cols[_ci].metric(f"{_lbl}: net (flat $100)", "—",
                                      f"collecting… ({_pend} pending)")
            _figab.update_layout(height=260, margin=dict(l=0, r=0, t=8, b=0),
                                 legend=dict(orientation="h", yanchor="bottom",
                                             y=1.02, xanchor="left", x=0))
            st.plotly_chart(_figab, use_container_width=True)
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
