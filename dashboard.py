"""
Live paper-trading dashboard — dark mode.

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

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"
REFRESH_SECONDS  = 30
DISPLAY_FROM     = "2026-03-28 02:00:00"  # hide trades before 7:00 PM PDT (02:00 UTC 2026-03-28)

st.set_page_config(
    page_title="Kalshi BTC",
    page_icon="₿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Dark mode CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
/* Base dark background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background-color: #0e0e0e !important;
    color: #e0e0e0 !important;
}}
[data-testid="stSidebar"] {{ background-color: #141414 !important; }}
[data-testid="stHeader"]  {{ background-color: #0e0e0e !important; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{ color: #888 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700; }}

/* Dividers */
hr {{ border-color: #2a2a2a !important; }}

/* Dataframe */
[data-testid="stDataFrame"] {{ background-color: #1a1a1a !important; border-radius: 10px; }}
thead tr th {{ background-color: #1f1f1f !important; color: #aaa !important; font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #333 !important; }}

/* Plotly chart background */
.js-plotly-plot {{ background-color: transparent !important; }}

/* Expander */
[data-testid="stExpander"] {{ background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; }}

/* Caption / info */
.stCaption {{ color: #666 !important; }}

/* Tabs */
[data-testid="stTabs"] button {{ color: #888 !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: #4da6ff !important; border-bottom-color: #4da6ff !important; }}
</style>
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=25)
def fetch_brti() -> dict:
    sources = {
        "coinbase": "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        "kraken":   "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        "bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/",
        "gemini":   "https://api.gemini.com/v1/pubticker/btcusd",
    }
    prices = {}
    for name, url in sources.items():
        try:
            r = requests.get(url, timeout=6)
            if name == "coinbase":   prices[name] = float(r.json()["data"]["amount"])
            elif name == "kraken":   prices[name] = float(r.json()["result"]["XXBTZUSD"]["c"][0])
            elif name == "bitstamp": prices[name] = float(r.json()["last"])
            elif name == "gemini":   prices[name] = float(r.json()["last"])
        except Exception:
            pass
    avg = sum(prices.values()) / len(prices) if prices else None
    return {"prices": prices, "avg": avg}


@st.cache_data(ttl=25)
def load_trades() -> pd.DataFrame:
    if not PAPER_TRADES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PAPER_TRADES_CSV)
    if df.empty:
        return df
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True).dt.tz_convert("America/Los_Angeles")
    for col in ["resolved_yes", "would_win"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)
    return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

now_str = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S PT")

st.markdown(f"""
<div style="margin-bottom:8px;">
  <span style="font-size:1.8rem;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">₿ Kalshi BTC</span>
  &nbsp;
  <span style="background:#c97a00;color:#fff;font-size:0.72rem;font-weight:700;
               padding:3px 10px;border-radius:20px;letter-spacing:0.1em;">PAPER</span>
</div>
<div style="color:#555;font-size:0.78rem;margin-bottom:4px;">
  BTC Event Contracts &nbsp;|&nbsp; Live Signal Engine &nbsp;|&nbsp;
  Updated: {now_str} &nbsp;|&nbsp; Auto-refresh: {REFRESH_SECONDS}s
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BRTI spot bar
# ---------------------------------------------------------------------------

brti  = fetch_brti()
spot  = brti["avg"]
prices = brti["prices"]

spot_cols = st.columns(5)
with spot_cols[0]:
    st.metric("BRTI (avg)", f"${spot:,.2f}" if spot else "—")
for i, (name, price) in enumerate(prices.items()):
    spot_cols[i + 1].metric(name.capitalize(), f"${price:,.2f}")

st.markdown("<hr style='margin:18px 0 14px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load & filter trades
# ---------------------------------------------------------------------------

df_all = load_trades()

if df_all.empty:
    st.markdown("<p style='color:#666;text-align:center;padding:40px 0;'>No trades logged yet.</p>", unsafe_allow_html=True)
    st.stop()

df = df_all[df_all["logged_at"] >= pd.Timestamp(DISPLAY_FROM, tz="UTC").tz_convert("America/Los_Angeles")].copy()

if df.empty:
    st.markdown(f"<p style='color:#666;text-align:center;padding:40px 0;'>No trades since {DISPLAY_FROM}.</p>", unsafe_allow_html=True)
    st.stop()

trades   = df[(df["decision"] == "trade") & (df["contract_ticker"].fillna("").str.strip() != "")].copy()
resolved = trades.dropna(subset=["would_win"]).copy()
pending  = trades[trades["would_win"].isna()].copy()

if not resolved.empty:
    resolved["would_win_bool"] = resolved["would_win"].astype(str).str.lower().isin(["true", "1", "yes"])
    resolved["would_pnl_num"]  = pd.to_numeric(resolved["would_pnl"], errors="coerce")
    win_rate  = resolved["would_win_bool"].mean()
    net_pnl   = resolved["would_pnl_num"].sum()
    wins      = resolved["would_win_bool"].sum()
    losses    = (~resolved["would_win_bool"]).sum()
else:
    win_rate = net_pnl = wins = losses = 0

# ---------------------------------------------------------------------------
# Stat cards (2 rows x 3 cols)
# ---------------------------------------------------------------------------

pnl_color  = "#00c076" if net_pnl >= 0 else "#ff4d4d"
wr_color   = "#00c076" if win_rate >= 0.5 else "#ff4d4d"
pnl_sign   = "+" if net_pnl >= 0 else ""

r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

def stat_card(col, label, value, color="#ffffff"):
    col.markdown(f"""
    <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;
                padding:20px 24px;margin-bottom:12px;">
      <div style="color:#666;font-size:0.68rem;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div style="color:{color};font-size:2rem;font-weight:800;line-height:1;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

stat_card(r1c1, "Total Trades",  len(trades),              "#4da6ff")
stat_card(r1c2, "Pending",       len(pending),              "#f0a500")
stat_card(r1c3, "Resolved",      len(resolved),             "#ffffff")
stat_card(r2c1, "Win Rate",      f"{win_rate:.1%}" if resolved.shape[0] else "—", wr_color)
stat_card(r2c2, "Total P&L",     f"${pnl_sign}{net_pnl:,.2f}" if resolved.shape[0] else "—", pnl_color)
stat_card(r2c3, "Wins / Losses", f"{int(wins)} / {int(losses)}" if resolved.shape[0] else "—", "#ffffff")

st.markdown("<hr style='margin:6px 0 18px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Latest signal
# ---------------------------------------------------------------------------

latest = df.iloc[-1]
dec    = str(latest.get("decision", "")).upper()
dec_color = "#00c076" if dec == "TRADE" else "#ff4d4d"

def _f(val, default=float("nan")):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _fmt(val, fmt, fallback="—"):
    v = _f(val)
    return fmt.format(v) if v == v else fallback  # v != v means NaN

st.markdown(f"""
<div style="font-size:0.72rem;color:#666;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">Latest Signal</div>
""", unsafe_allow_html=True)

ls1, ls2, ls3, ls4, ls5, ls6 = st.columns(6)
ls1.metric("Decision",  dec)
ls2.metric("Side",      str(latest.get("side", "—")).upper())
ls3.metric("p_model",   _fmt(latest.get("p_yes_model"), "{:.3f}"))
ls4.metric("p_market",  _fmt(latest.get("p_market"),    "{:.3f}"))
ls5.metric("Net edge",  _fmt(latest.get("net_edge"),    "{:+.3f}"))
ls6.metric("Bet",       _fmt(latest.get("bet_amount"),  "${:,.0f}"))

lb1, lb2, lb3, lb4 = st.columns(4)
lb1.metric("Spot",               _fmt(latest.get("spot"),   "${:,.2f}"))
lb2.metric("Strike",             _fmt(latest.get("strike"), "${:,.2f}"))
lb3.metric("Contracts Scanned",  int(_f(latest.get("contracts_scanned", 0), 0)))
lb4.metric("Contract",           latest.get("contract_ticker", "—") or "—")

st.markdown("<hr style='margin:18px 0 18px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Trades", "Equity Curve", "Indicator Stats"])

with tab2:
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

# ---------------------------------------------------------------------------
# Trade log table
# ---------------------------------------------------------------------------

with tab1:
    display_cols = [
        "logged_at", "contract_ticker", "side", "spot", "strike",
        "tau_minutes", "p_yes_model", "p_market", "net_edge",
        "structure_bias", "confirmation_score", "no_score", "obi_score",
        "bet_amount", "would_pnl", "would_win",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    trade_rows   = trades[display_cols].copy().sort_values("logged_at", ascending=False)

    # Format result column
    def fmt_result(row):
        w = str(row.get("would_win", "")).lower()
        if w == "true":  return "WIN"
        if w == "false": return "LOSS"
        return "pending"

    trade_rows["result"] = trade_rows.apply(fmt_result, axis=1)

    # Coerce numeric columns so styler format specs don't fail on corrupt string values
    for _nc in ["spot", "strike", "tau_minutes", "p_yes_model", "p_market",
                "net_edge", "bet_amount", "would_pnl",
                "structure_bias", "confirmation_score", "no_score", "obi_score"]:
        if _nc in trade_rows.columns:
            trade_rows[_nc] = pd.to_numeric(trade_rows[_nc], errors="coerce")

    # Drop raw would_win, rename columns
    trade_rows = trade_rows.drop(columns=["would_win"], errors="ignore")
    trade_rows = trade_rows.rename(columns={
        "logged_at":       "Time (PT)",
        "contract_ticker": "Contract",
        "side":            "Side",
        "spot":            "Spot",
        "strike":          "Strike",
        "tau_minutes":     "τ (min)",
        "p_yes_model":     "p_model",
        "p_market":        "p_market",
        "net_edge":          "Net Edge",
        "structure_bias":    "Struct",
        "confirmation_score":"Conf",
        "no_score":          "NO Score",
        "obi_score":         "OBI",
        "bet_amount":        "Bet ($)",
        "would_pnl":       "P&L ($)",
        "result":          "Result",
    })

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

    score_cols = [c for c in ["Struct", "Conf", "NO Score", "OBI"] if c in trade_rows.columns]

    styled = (
        trade_rows.style
        .applymap(color_result, subset=["Result"])
        .applymap(color_pnl,    subset=["P&L ($)"])
        .applymap(color_edge,   subset=["Net Edge"])
        .applymap(color_score,  subset=score_cols)
        .format({
            "Spot":     "${:,.2f}",
            "Strike":   "${:,.2f}",
            "p_model":  "{:.3f}",
            "p_market": "{:.3f}",
            "Net Edge": "{:+.3f}",
            "Bet ($)":  "${:,.0f}",
            "τ (min)":  "{:.0f}",
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

    st.dataframe(styled, use_container_width=True, height=450)

# ---------------------------------------------------------------------------
# Indicator Stats tab
# ---------------------------------------------------------------------------

with tab3:
    if resolved.empty:
        st.markdown("<p style='color:#555;padding:20px 0;'>No resolved trades yet — stats will appear once contracts settle.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#888;font-size:0.78rem;margin-bottom:16px;'>Win rate by indicator value across all resolved trades in this session.</div>", unsafe_allow_html=True)

        INDICATORS = [
            ("structure_bias",    "Structure Bias",    {-1: "Bearish (-1)", 0: "Neutral (0)", 1: "Bullish (+1)"}),
            ("confirmation_score","Confirmation Score", None),
            ("no_score",          "NO Score",           None),
            ("obi_score",         "OBI Score",          {-1: "Bearish (-1)", 0: "Neutral (0)", 1: "Bullish (+1)"}),
            ("vol_score",         "Vol Score",          {-1: "Down vol (-1)", 0: "Low vol (0)", 1: "Up vol (+1)"}),
            ("vwap_score",        "VWAP Score",         {-1: "Below VWAP (-1)", 1: "Above VWAP (+1)"}),
            ("ema_stretch_score", "EMA Stretch Score",  {-1: "Overbought (-1)", 0: "Neutral (0)", 1: "Oversold (+1)"}),
        ]

        for col, label, val_labels in INDICATORS:
            if col not in resolved.columns:
                st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:12px 0 2px 0;'>{label}</div>", unsafe_allow_html=True)
                st.markdown("<div style='color:#555;font-size:0.75rem;padding:4px 0 8px 0;'>No data yet — populates as new trades resolve.</div>", unsafe_allow_html=True)
                continue
            col_data = pd.to_numeric(resolved[col], errors="coerce").dropna()
            if col_data.empty:
                st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:12px 0 2px 0;'>{label}</div>", unsafe_allow_html=True)
                st.markdown("<div style='color:#555;font-size:0.75rem;padding:4px 0 8px 0;'>No data yet — populates as new trades resolve.</div>", unsafe_allow_html=True)
                continue

            merged = resolved.copy()
            merged["_ind"] = pd.to_numeric(merged[col], errors="coerce")
            merged = merged.dropna(subset=["_ind"])
            merged["_ind"] = merged["_ind"].astype(int)
            merged["_win"] = merged["would_win_bool"].astype(int)

            grouped = merged.groupby("_ind").agg(
                Trades=("_win", "count"),
                Wins=("_win", "sum"),
            ).reset_index()
            grouped["Trades"]   = grouped["Trades"].astype(int)
            grouped["Wins"]     = grouped["Wins"].astype(int)
            grouped["Losses"]   = grouped["Trades"] - grouped["Wins"]
            grouped["Win Rate"] = grouped["Wins"].astype(float) / grouped["Trades"].astype(float)

            if val_labels:
                grouped["Value"] = grouped["_ind"].map(lambda v: val_labels.get(v, str(v)))
            else:
                grouped["Value"] = grouped["_ind"].astype(str)

            grouped = grouped[["Value", "Trades", "Wins", "Losses", "Win Rate"]].sort_values("Value")

            def color_wr(val):
                try:
                    v = float(str(val).rstrip("%")) / 100 if "%" in str(val) else float(val)
                    if v >= 0.6:   return "color: #00c076; font-weight: 700"
                    if v >= 0.5:   return "color: #7ec8a0"
                    if v >= 0.4:   return "color: #f0a500"
                    return "color: #ff4d4d"
                except Exception:
                    return ""

            styled_ind = (
                grouped.style
                .applymap(color_wr, subset=["Win Rate"])
                .format({"Win Rate": "{:.1%}"})
                .set_properties(**{"background-color": "#1a1a1a", "color": "#ddd", "border-color": "#2a2a2a"})
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#141414"), ("color", "#777"),
                        ("font-size", "0.68rem"), ("letter-spacing", "0.08em"),
                        ("text-transform", "uppercase"), ("border-bottom", "1px solid #333"),
                    ]},
                ])
                .hide(axis="index")
            )

            st.markdown(f"<div style='font-size:0.8rem;color:#aaa;font-weight:600;margin:12px 0 4px 0;'>{label}</div>", unsafe_allow_html=True)
            st.dataframe(styled_ind, use_container_width=True, height=min(38 * len(grouped) + 48, 200))

st.markdown("<hr style='margin:18px 0 18px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Signal breakdown expander
# ---------------------------------------------------------------------------

with st.expander("Signal Breakdown"):
    sb1, sb2, sb3 = st.columns(3)
    bias_counts = df["structure_bias"].value_counts()
    sb1.metric("Bullish Structure",  int(bias_counts.get(1,  0)))
    sb2.metric("Neutral Structure",  int(bias_counts.get(0,  0)))
    sb3.metric("Bearish Structure",  int(bias_counts.get(-1, 0)))

    cb1, cb2, cb3 = st.columns(3)
    conf_counts = df["confirmation_bias"].value_counts()
    cb1.metric("Bullish Confirmation", int(conf_counts.get(1,  0)))
    cb2.metric("Neutral Confirmation", int(conf_counts.get(0,  0)))
    cb3.metric("Bearish Confirmation", int(conf_counts.get(-1, 0)))
