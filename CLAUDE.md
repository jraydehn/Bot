# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the System

```bash
# Continuous live monitor (primary trading loop, 2-min polling):
python3 live_monitor.py --bankroll 10000

# Single-run paper trade (one decision + CSV log):
python3 paper_trade_runner.py --bankroll 10000

# One-off live signal report (no CSV logging):
python3 live_signal.py

# Streamlit dashboard:
streamlit run dashboard.py

# Historical single-point evaluation:
python3 evaluate_point.py --time "2026-03-01 14:00"
python3 evaluate_point.py --time "2026-03-01 14:00" --offset 0.005 --p-market 0.48

# Walk-forward backtest:
python3 backtest.py --start "2026-02-01" --end "2026-03-01" --bankroll 10000

# Refresh OHLCV data from Binance US:
python3 fetch_data.py

# Fill in P&L for expired contracts:
python3 outcome_checker.py

# Automated hourly runner (fetch → trade → resolve):
bash run_paper_trade.sh
```

All scripts accept `--sim` to use simulated Kalshi prices (no auth required).

## Authentication

Kalshi credentials are loaded from environment variables (set in `~/.zshrc` and Claude Code settings):
```
KALSHI_KEY_ID=25b3db1f-83bd-436f-b97a-bb74dabcfdfe
KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
```

## Architecture

The system is a **sequential gate pipeline** for trading 1-hour BTC binary event contracts on Kalshi.

### Signal Pipeline (executed each poll cycle)

```
OHLCV parquet (1m/1h/4h) + live Binance candles
    └─> market_data.py          realized volatility (60m window)
    └─> probability_engine.py   log-normal p_yes (blends realized + implied vol, 60/40)
    └─> market_structure.py     swing high/low detection on 4h → bias (+1/-1/0)
    └─> confirmation_indicators.py  EMA(20/50) + RSI(21) + Volume → score (-3 to +3)
    └─> pricing_comparison.py   net_edge = raw_edge − fee − slippage − spread
    └─> decision.py             gate evaluation → trade/no_trade + side
    └─> kelly_sizing.py         bet sizing (¼ Kelly YES, ½ Kelly NO, 5% cap)
```

### Decision Gates (decision.py)

Gates are evaluated in order; failure at any gate triggers Gate P before returning no_trade:

- **Gate 1 – Structure**: `structure_bias` must support the side. Neutral (0) is allowed for NO trades with a +1% edge premium at Gate 3.
- **Gate 2 – Confirmation**: `confirmation_bias` must align with side.
- **Gate 3 – Net edge**: Must exceed `MIN_NET_EDGE` (1%) + neutral premium if applicable.
- **Gate P – Pure-edge override**: Fires when mispricing is large enough regardless of signals. Uses 1/8 Kelly. Tiered by `confirmation_score`:

| confirmation_score | YES min edge | no_score | NO min edge |
|--------------------|-------------|----------|-------------|
| ≥ 3 (all bullish)  | 6%          | ≤ −2     | 6%          |
| ≥ 1 (net positive) | 8%          | ≤ 0      | 8%          |
| ≤ 0                | blocked     | ≤ 1      | 10%         |
|                    |             | ≥ 2      | blocked     |

### Trade Side Logic

Side is determined by signals, not by comparing `p_model` vs `p_market`:
- `structure_bias = +1` → YES
- `structure_bias = -1` → NO
- `structure_bias = 0` → defer to `confirmation_bias` (+1 → YES, else → NO)

### Kelly Sizing

- YES bets: ¼ Kelly (low ~26% win rate, high variance)
- NO bets: ½ Kelly (high ~96% win rate, low variance)
- Hard cap: never risk more than 5% of bankroll per trade

### Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MIN_NET_EDGE` | 1% | pricing_comparison.py |
| `NEUTRAL_STRUCTURE_EDGE_PREMIUM` | 1% | decision.py |
| `PURE_EDGE_MIN_NET_EDGE` | 8% | decision.py |
| `KALSHI_RAKE` | 7% | pricing_comparison.py |
| `DEFAULT_SLIPPAGE` | 0.5% | pricing_comparison.py |
| `DEFAULT_SPREAD` | 1.0% | pricing_comparison.py |
| `REALIZED_VOL_WEIGHT` | 0.6 | probability_engine.py |
| `MAX_BET_FRACTION` | 5% | kelly_sizing.py |
| `TAU` | 60 min | live_signal.py |

## Data

- **OHLCV parquets**: `data/binanceus_BTCUSDT_{1m,1h,4h}_*.parquet` (loaded by `evaluate_point.load_data()`)
- **Paper trades**: `results/paper_trades.csv` — 38-column log; `resolved_yes`, `would_win`, `would_pnl` filled in later by `outcome_checker.py`
- **Live candle extension**: `live_signal.extend_with_live_candles(df, interval, lookback_bars)` appends fresh Binance candles to the parquet tail at startup and on hourly refresh

## Git Branches

- **`main`** — 7-indicator model (EMA + RSI + Vol + MACD + VWAP + Mom15m + Mom60m, score −9 to +9)
- **`baseline-3indicator-tiered`** — 3-indicator model (EMA + RSI + Vol only, score −3 to +3) — the model under which most paper trade data was collected

All edits auto-commit and push to `https://github.com/jraydehn/Bot.git` via a PostToolUse hook.

## Version Control

Beyond the auto-save hook, **commit meaningful checkpoints manually** with descriptive messages whenever:
- A model change is complete (new indicator, revised threshold, new gate logic)
- A bug is fixed
- Before and after any experimental change that could be reverted
- A branch is ready to compare against another

Use clean commit messages that describe *what changed and why*, not just "auto-save". Example:
```bash
git add decision.py confirmation_indicators.py
git commit -m "raise Gate P YES floor to 25% for score 2-3 (max losing edge was 24.2%)"
git push
```

This ensures any model state can be recovered exactly, and branches can be compared or cherry-picked without guesswork.

## Live Monitor Safeguards

`live_monitor.py` includes session-level drawdown limits:
- **Soft halt** (−20% of bankroll): only trades with `net_edge >= 8%` are allowed
- **Hard halt** (−35% of bankroll): all trading stopped
- Contradictory positions within the same expiry window are blocked (e.g. YES on strike A + NO on strike B where B ≤ A)
- Each contract+side pair is only logged once per session
