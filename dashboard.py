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
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR     = Path(__file__).parent / "results"
REFRESH_SECONDS = 60
DISPLAY_FROM    = "2026-04-21 03:40:00"   # hide trades before this UTC time (dashboard cleared Apr 20 8:40 PM PDT — BTC drift multiplier reverted to k=1.0, 0.65×/0.90× restored)

ASSET_CSV = {
    "BTC": RESULTS_DIR / "paper_trades.csv",
    "ETH": RESULTS_DIR / "paper_trades_eth.csv",
    "SOL": RESULTS_DIR / "paper_trades_sol.csv",
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
def load_trades(asset: str) -> pd.DataFrame:
    csv_path = ASSET_CSV.get(asset)
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")
    df = df[df["logged_at"].notna()]  # drop rows whose logged_at didn't parse (split CSV lines)
    for col in ["resolved_yes", "would_win"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)
    return df


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

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


def render_asset(asset: str):
    spot_data = fetch_spot(asset)
    spot      = spot_data["avg"]
    prices    = spot_data["prices"]

    # Spot bar
    spot_cols = st.columns(len(prices) + 1)
    with spot_cols[0]:
        st.metric(f"{asset} (avg)", f"${spot:,.2f}" if spot else "—")
    for i, (name, price) in enumerate(prices.items()):
        spot_cols[i + 1].metric(name.capitalize(), f"${price:,.2f}")

    st.markdown("<hr style='margin:14px 0 12px 0;'>", unsafe_allow_html=True)

    # Load data
    df_all = load_trades(asset)

    if df_all.empty:
        st.markdown("<p style='color:#666;text-align:center;padding:30px 0;'>No trades logged yet.</p>", unsafe_allow_html=True)
        return

    # Apply display cutoff for all assets
    cutoff = pd.Timestamp(DISPLAY_FROM, tz="UTC").tz_convert("America/Los_Angeles")
    df = df_all[df_all["logged_at"] >= cutoff].copy()

    if df.empty:
        st.markdown("<p style='color:#666;text-align:center;padding:30px 0;'>No trades in display window.</p>", unsafe_allow_html=True)
        return

    trades   = df[(df["decision"] == "trade") & (df["contract_ticker"].fillna("").str.strip() != "")].copy()
    resolved = trades.dropna(subset=["would_win"]).copy()
    pending  = trades[trades["would_win"].isna()].copy()

    if not resolved.empty:
        resolved["would_win_bool"] = resolved["would_win"].astype(str).str.lower().isin(["true", "1", "yes"])
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
    if asset in ("BTC", "ETH", "SOL") and "composite_trend" in df.columns:
        cs1, cs2, cs3, cs4 = st.columns(4)
        comp_trend = latest.get("composite_trend", "—")
        comp_rev   = latest.get("composite_rev",   "—")
        comp_p_up  = latest.get("composite_p_up",  "—")
        cs1.metric("Composite Trend",  _fmt(comp_trend, "{:+.0f}"))
        cs2.metric("Composite Rev",    _fmt(comp_rev,   "{:+.0f}"))
        cs3.metric("Composite p_up",   _fmt(comp_p_up,  "{:.1%}"))
        _p_up_v = _f(comp_p_up)
        _asset_base = {"BTC": 0.504, "ETH": 0.509, "SOL": 0.500}.get(asset, 0.504)
        _edge_vs_base = _p_up_v - _asset_base if _p_up_v == _p_up_v else float("nan")
        cs4.metric("vs Baseline",      f"{_edge_vs_base:+.1%}" if _edge_vs_base == _edge_vs_base else "—")

    # Price / contract context row
    lb1, lb2, lb3, lb4, lb5 = st.columns(5)
    lb1.metric("Spot",              _fmt(latest.get("spot"),   "${:,.2f}"))
    lb2.metric("Strike",            _fmt(latest.get("strike"), "${:,.2f}"))
    lb3.metric("Offset %",          _fmt(latest.get("offset_pct"), "{:+.3f}%"))
    lb4.metric("Vol Eff",           _fmt(latest.get("vol_eff"),    "{:.5f}"))
    lb5.metric("Contracts Scanned", int(_f(latest.get("contracts_scanned", 0), 0)))

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
        if not resolved.empty:
            res_sorted = resolved.sort_values("logged_at")
            res_sorted["cumulative_pnl"] = res_sorted["would_pnl_num"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res_sorted["logged_at"],
                y=res_sorted["cumulative_pnl"],
                mode="lines+markers",
                line=dict(color="#00c076", width=2),
                marker=dict(size=5, color="#00c076"),
                fill="tozeroy",
                fillcolor="rgba(0,192,118,0.08)",
                name="Cumulative P&L",
                hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#444", opacity=0.8)
            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Cumulative P&L ($)",
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e0e0e",
                paper_bgcolor="#0e0e0e",
                font=dict(color="#888"),
                xaxis=dict(gridcolor="#1f1f1f", zeroline=False),
                yaxis=dict(gridcolor="#1f1f1f", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<p style='color:#555;padding:20px 0;'>No resolved trades yet.</p>", unsafe_allow_html=True)

    # ── Trade log ───────────────────────────────────────────────────────────
    with t1:
        _composite_only = st.checkbox(
            "Composite drift hybrid only",
            value=True,
            key=f"composite_only_{asset}",
            help="Show only trades using the composite drift hybrid with vol drift model (composite_p_up populated)",
        )

        display_cols = [
            "logged_at", "contract_ticker", "side",
            "offset_pct", "spot", "strike", "tau_minutes",
            "p_yes_model", "p_market", "net_edge",
            "would_pnl", "would_win",
            "composite_trend", "composite_rev", "composite_p_up",
            "chg_5m", "chg_10m", "chg_30m", "sharp_move_active",
            "confirmation_score", "no_score",
            "obi_score", "funding_bias", "vol_score", "vwap_score", "ema_stretch_score",
            "vol_eff",
            "kelly_fraction", "bet_amount",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        _trade_source = trades.copy()
        if _composite_only and "composite_p_up" in _trade_source.columns:
            _trade_source = _trade_source[
                pd.to_numeric(_trade_source["composite_p_up"], errors="coerce").fillna(0) != 0
            ]
        trade_rows = _trade_source[display_cols].copy().sort_values("logged_at", ascending=False)

        def fmt_result(row):
            w = str(row.get("would_win", "")).lower()
            if w == "true":  return "WIN"
            if w == "false": return "LOSS"
            return "pending"

        trade_rows["result"] = trade_rows.apply(fmt_result, axis=1)
        trade_rows = trade_rows.drop(columns=["would_win"], errors="ignore")

        for _nc in ["offset_pct", "spot", "strike", "tau_minutes", "p_yes_model", "p_market",
                    "net_edge", "bet_amount", "would_pnl", "kelly_fraction",
                    "composite_trend", "composite_rev", "composite_p_up",
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
            "composite_p_up":     "p_up",
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
            "Time (PT)", "Contract", "Side",
            "Offset%", "Spot", "Strike", "τ (min)",
            "p_model", "p_market", "Net Edge",
            "P&L ($)", "Result",
            "Trend", "Rev", "p_up",
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
               padding:3px 10px;border-radius:20px;letter-spacing:0.1em;">PAPER</span>
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

tab_btc, tab_eth, tab_sol, tab_cmp = st.tabs(["₿  BTC", "Ξ  ETH", "◎  SOL", "📊  Compare"])

with tab_btc:
    render_asset("BTC")

with tab_eth:
    render_asset("ETH")

with tab_sol:
    render_asset("SOL")

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
            cutoff = pd.Timestamp(DISPLAY_FROM, tz="UTC").tz_convert("America/Los_Angeles")
            df_a = df_a[df_a["logged_at"] >= cutoff]
            trades_a   = df_a[(df_a["decision"] == "trade") & (df_a["contract_ticker"].fillna("").str.strip() != "")]
            resolved_a = trades_a.dropna(subset=["would_win"]).copy()
            if not resolved_a.empty:
                resolved_a["would_win_bool"] = resolved_a["would_win"].astype(str).str.lower().isin(["true", "1", "yes"])
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
