"""
Live paper-trading dashboard.

Run with:
    streamlit run dashboard.py

Auto-refreshes every 30 seconds. Displays:
  - Live BRTI spot price
  - Current signal (latest row in paper_trades.csv)
  - Full trade log with P&L
  - Running metrics (trades, win rate, net P&L)
  - Equity curve
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

st.set_page_config(
    page_title="Kalshi BTC Dashboard",
    page_icon="₿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
    """,
    unsafe_allow_html=True,
)

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
            if name == "coinbase":
                prices[name] = float(r.json()["data"]["amount"])
            elif name == "kraken":
                prices[name] = float(r.json()["result"]["XXBTZUSD"]["c"][0])
            elif name == "bitstamp":
                prices[name] = float(r.json()["last"])
            elif name == "gemini":
                prices[name] = float(r.json()["last"])
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

st.title("₿ Kalshi BTC Live Dashboard")
now = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S PST")
st.caption(f"Last updated: {now}  ·  auto-refreshes every {REFRESH_SECONDS}s")

# ---------------------------------------------------------------------------
# BRTI spot
# ---------------------------------------------------------------------------

brti = fetch_brti()
spot = brti["avg"]
prices = brti["prices"]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("BRTI (avg)", f"${spot:,.2f}" if spot else "—")
for i, (name, price) in enumerate(prices.items()):
    [col2, col3, col4, col5][i].metric(name.capitalize(), f"${price:,.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

df = load_trades()

if df.empty:
    st.info("No trades logged yet. Run `live_monitor.py` to start.")
    st.stop()

# ---------------------------------------------------------------------------
# Latest signal
# ---------------------------------------------------------------------------

latest = df.iloc[-1]
decision_color = "🟢" if latest["decision"] == "trade" else "🔴"

st.subheader("Latest Signal")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Decision", f"{decision_color} {latest['decision'].upper()}")
c2.metric("Side", latest["side"].upper())
c3.metric("p_model", f"{float(latest['p_yes_model']):.3f}")
c4.metric("p_market", f"{float(latest['p_market']):.3f}")
c5.metric("Net edge", f"{float(latest['net_edge']):+.3f}")
c6.metric("Bet", f"${float(latest['bet_amount']):,.0f}")

c1b, c2b, c3b, c4b = st.columns(4)
c1b.metric("Spot", f"${float(latest['spot']):,.2f}")
c2b.metric("Strike", f"${float(latest['strike']):,.2f}")
c3b.metric("Contracts scanned", int(latest.get("contracts_scanned", 0)))
c4b.metric("Contract", latest.get("contract_ticker", "—") or "—")

st.divider()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

trades = df[df["decision"] == "trade"].copy()
resolved = trades.dropna(subset=["would_win"])

st.subheader("Performance")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total signals", len(df))
m2.metric("Trades taken", len(trades))

if not resolved.empty:
    resolved["would_win"] = resolved["would_win"].astype(str).str.lower().isin(["true", "1", "yes"])
    win_rate = resolved["would_win"].mean()
    net_pnl  = pd.to_numeric(resolved["would_pnl"], errors="coerce").sum()
    m3.metric("Win rate", f"{win_rate:.1%}")
    m4.metric("Net P&L", f"${net_pnl:+,.2f}")
    m5.metric("Resolved", len(resolved))
else:
    m3.metric("Win rate", "—")
    m4.metric("Net P&L", "—")
    m5.metric("Resolved", 0)

st.divider()

# ---------------------------------------------------------------------------
# Gate breakdown
# ---------------------------------------------------------------------------

if "neutral_gate" in df.columns and "pure_edge_gate" in df.columns:
    st.subheader("Gate Breakdown")
    g1, g2, g3, g4 = st.columns(4)
    normal_trades = trades[
        ~trades["neutral_gate"].astype(str).isin(["True", "true", "1"]) &
        ~trades["pure_edge_gate"].astype(str).isin(["True", "true", "1"])
    ]
    neutral = trades[trades["neutral_gate"].astype(str).isin(["True", "true", "1"])]
    pure_edge = trades[trades["pure_edge_gate"].astype(str).isin(["True", "true", "1"])]
    g1.metric("Normal gate trades", len(normal_trades))
    g2.metric("Neutral gate trades", len(neutral))
    g3.metric("Gate P trades", len(pure_edge))
    g4.metric("No-trades", len(df[df["decision"] == "no_trade"]))
    st.divider()

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

if not resolved.empty:
    st.subheader("Equity Curve")
    resolved = resolved.sort_values("logged_at")
    resolved["cumulative_pnl"] = pd.to_numeric(resolved["would_pnl"], errors="coerce").cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=resolved["logged_at"],
        y=resolved["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#00cc96", width=2),
        marker=dict(size=6),
        name="Cumulative P&L",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Cumulative P&L ($)",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# ---------------------------------------------------------------------------
# Trade log table
# ---------------------------------------------------------------------------

st.subheader("Trade Log")

display_cols = [
    "logged_at", "contract_ticker", "side", "spot", "strike",
    "tau_minutes", "p_yes_model", "p_market", "net_edge", "decision",
    "gate_blocked", "neutral_gate", "pure_edge_gate", "bet_amount",
    "resolved_yes", "would_win", "would_pnl",
]
display_cols = [c for c in display_cols if c in df.columns]
display_df = df[display_cols].copy().sort_values("logged_at", ascending=False)

# Colour rows: green = win, red = loss, yellow = unresolved trade, white = no_trade
def row_style(row):
    if row.get("decision") == "no_trade":
        return [""] * len(row)
    won = str(row.get("would_win", "")).lower()
    if won == "true":
        return ["background-color: #d4edda; color: #000000"] * len(row)
    elif won == "false":
        return ["background-color: #f8d7da; color: #000000"] * len(row)
    else:
        return ["background-color: #fff3cd; color: #000000"] * len(row)

st.dataframe(
    display_df.style.apply(row_style, axis=1),
    use_container_width=True,
    height=400,
)

# ---------------------------------------------------------------------------
# Structure / confirmation breakdown
# ---------------------------------------------------------------------------

with st.expander("Signal breakdown"):
    sb1, sb2, sb3 = st.columns(3)
    bias_counts = df["structure_bias"].value_counts()
    sb1.metric("Bullish structure", int(bias_counts.get(1, 0)))
    sb2.metric("Neutral structure", int(bias_counts.get(0, 0)))
    sb3.metric("Bearish structure", int(bias_counts.get(-1, 0)))

    cb1, cb2, cb3 = st.columns(3)
    conf_counts = df["confirmation_bias"].value_counts()
    cb1.metric("Bullish confirmation", int(conf_counts.get(1, 0)))
    cb2.metric("Neutral confirmation", int(conf_counts.get(0, 0)))
    cb3.metric("Bearish confirmation", int(conf_counts.get(-1, 0)))
