# Regime Data & Deep Gate Analysis — System Breakdown

This document covers two things:
1. Every Markov and GARCH data point collected and how each is used
2. The deep gate analysis process — the exact methodology used to improve decision logic

---

## Part 1: Markov Regime Data

### What "Markov Regime" means here

This system does not use a full Hidden Markov Model with Viterbi decoding. The regime labels (Bull / Bear / Sideways) are computed using a **rolling return threshold** on price history. The name "Markov" refers to the regimes being treated as discrete states that condition all downstream decisions — not to the fitting algorithm itself.

The computation pattern for every asset and timeframe:
```
rolling_return = close.pct_change(N_bars).iloc[-1]
regime = "Bull" if rolling_return > +threshold else
         "Bear" if rolling_return < -threshold else
         "Sideways"
```

---

### BTC Markov Regimes

#### BTC Daily Regime (`markov_regime_daily`)
- **Source**: yfinance BTC-USD daily closes, last 65 days
- **Lookback**: 20 bars (20 calendar days)
- **Threshold**: ±2.0%
  - Bull  → 20d return > +2%
  - Bear  → 20d return < −2%
  - Sideways → within ±2%
- **Cache**: Once per UTC calendar day
- **Where used**: `paper_trade_runner.py` (1h runner)
  - `markov_sideways_gate`: blocks BTC YES + most BTC NO when macro is flat
  - `garch_markov_vol_adjust`: deflates sigma when Sideways + GARCH ratio suppressed
- **Logged column**: `markov_regime_daily`

#### BTC 1h Regime (`markov_regime_1h`)
- **Source**: yfinance BTC-USD 1h closes, last 4 days (~96 bars)
- **Lookback**: 20 bars (20 hours)
- **Threshold**: ±0.8%
  - Bull  → 20h return > +0.8%
  - Bear  → 20h return < −0.8%
  - Sideways → within ±0.8%
- **Cache**: Once per UTC hour
- **Where used**: `paper_trade_runner_15m.py` (15m runner)
  - Logged for LGBM feature accumulation; potential future gates
- **Logged column**: `markov_regime_1h`

#### BTC 15m Regime (`markov_regime_15m`)
- **Source**: Derived from the same live 1m Binance feed used for signals
  - Resample 1m → 15m, compute 20-bar rolling return
- **Threshold**: ±0.4%
- **Where used**: `paper_trade_runner_15m.py`
  - Logged for LGBM feature accumulation
- **Logged column**: `markov_regime_15m`

---

### ETH Markov Regimes

#### ETH Daily Regime (`markov_eth_daily`)
- **Source**: yfinance ETH-USD 1h closes resampled to daily, last 120 days
- **Lookback**: 20 bars (20 calendar days)
- **Threshold**: ±3.0% (higher than BTC to account for ETH's larger base volatility)
- **Cache**: Once per UTC hour
- **Where used**: `paper_trade_runner_15m.py`
  - Logged; potential future gates after sufficient data
- **Logged column**: `markov_eth_daily`

---

### SOL Markov Regimes (Multi-Timeframe)

SOL uses three simultaneous regime reads — one per timeframe — because SOL moves faster and the broader macro context matters less at the contract level.

| Column | Timeframe | Lookback | Threshold |
|--------|-----------|----------|-----------|
| `markov_sol_6h` | 6h resampled from 1h | 20 bars (5 days) | ±3.0% |
| `markov_sol_4h` | 4h resampled from 1h | 20 bars (~3.3 days) | ±2.5% |
| `markov_sol_1h` | 1h direct | 20 bars (20 hours) | ±1.5% |

- **Source**: yfinance SOL-USD 1h closes, last 120 days
- **Cache**: Once per UTC hour, all three timeframes in a single fetch
- **Where used**: `paper_trade_runner_15m.py`
  - Logged; multi-timeframe stack alignment is a planned future gate trigger
- **Logged columns**: `markov_sol_6h`, `markov_sol_4h`, `markov_sol_1h`

---

### Regime Summary Table

| Column | Asset | Timeframe | Lookback | Threshold | Cache |
|--------|-------|-----------|----------|-----------|-------|
| `markov_regime_daily` | BTC | Daily | 20d | ±2.0% | Per day |
| `markov_regime_1h` | BTC | 1h | 20h | ±0.8% | Per hour |
| `markov_regime_15m` | BTC | 15m | 20×15m | ±0.4% | Per scan |
| `markov_eth_daily` | ETH | Daily | 20d | ±3.0% | Per hour |
| `markov_sol_6h` | SOL | 6h | 20×6h | ±3.0% | Per hour |
| `markov_sol_4h` | SOL | 4h | 20×4h | ±2.5% | Per hour |
| `markov_sol_1h` | SOL | 1h | 20h | ±1.5% | Per hour |

---

## Part 2: GARCH Data Points

### What is being fit

GARCH(1,1) with normal innovations is fit on the last **500 bars of 1h BTC log returns** (scaled ×100 for numerical stability). BTC only — ETH and SOL use the simpler vol_layer instead.

The GARCH variance equation:
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Two quantities are extracted from each fit:

#### 1. Conditional Vol Ratio (`garch_ratio`)
```
cond_vol     = σ_t (conditional vol from latest bar, in percentage units)
long_run_vol = sqrt(ω / (1 - α - β))   [unconditional vol; fallback to rolling std if α+β≥1]
ratio        = cond_vol / long_run_vol
```

- **ratio > 1.0** → current vol elevated vs long-run average (spike regime)
- **ratio > 1.5** → high-vol trigger (GARCH gate fires)
- **ratio < 0.67** → suppressed vol trigger (sigma deflation fires)

**Persistence** `α + β` is also computed and logged. On validated data, BTC shows `α+β ≈ 0.935`, meaning vol shocks take many hours to decay.

#### 2. Conditional Vol Effective (`garch_ve`)
```
garch_ve = cond_vol (%) / 100 / sqrt(60)    [per sqrt-minute, same units as vol_eff]
```

Used to override the blended sigma (`vol_eff`) in the BTC branched drift model when a reliable GARCH estimate is available.

- **Cache**: Once per UTC hour (one `arch` model fit per hour maximum)
- **BTC only**: `_get_garch_ratio()` returns `None` for ETH/SOL immediately

---

### How GARCH feeds into the system

#### Gate: `btc_garch_highvol_yes_gate`
- **Trigger**: GARCH ratio > 1.5 AND BTC YES bet
- **Mechanism**: High conditional vol = elevated realized uncertainty that the market has NOT fully priced into contracts. In practice, this regime corresponds to bull-trap bounces inside a broader bear macro. Win rate on YES drops to ~25% in this zone.
- **Rescue**: `pm ≥ 0.80 AND tau < 45 min` → deep-ITM contract near expiry; GARCH vol cannot bridge the distance to strike
- **Validated on**: 308 blocked trades (WR=25%), 231 losses saved

#### Vol Adjustment: `garch_markov_vol_adjust`
- **Trigger**: GARCH ratio < 0.67 AND daily Markov = Sideways
- **Action**: Deflate sigma by one vote step (`−0.08` from vol_layer factor)
- **Why**: Suppressed GARCH + flat macro = the model overestimates sigma, accepting YES bets below breakeven (validated: n=146 YES in this regime, WR=43.8% vs BE=48.9%, −$7.34 per $)
- **Scope**: BTC only (ETH and SOL are profitable in LOW+Sideways)

#### Sigma Override in Branched Drift Model
- When `garch_ve > 0` is available, BTC NO bets use it directly as `vol_eff` instead of the blended Deribit/realized estimate
- Provides a real-time vol anchor that is more responsive than hour-lagged realized vol

---

### Vol Layer (Complements GARCH)

The `vol_layer.py` runs alongside GARCH with five independent vote signals. Each signal casts ±1 vote; total score maps to a factor in [0.60, 1.40] that multiplies `vol_eff`.

| Signal | High-vol trigger | Low-vol trigger |
|--------|-----------------|-----------------|
| ATR ratio | current ATR > 1.5× 24h mean | < 0.75× |
| Abs z-score | \|z\| > 2.0 | < 0.5 |
| Volume ratio | current vol > 3.0× mean | < 0.30× |
| VWAP deviation | > 1.0% from VWAP | < 0.2% |
| Realized vol 6h | RV > 0.30 ann | < 0.10 ann |

Thresholds are asset-specific (ETH/SOL are more volatile, so their HIGH thresholds are lower).

---

### Rolling Drift Measures (Logged alongside GARCH)

These are logged in both the 1h and 15m runners and used in the branched YES/NO drift model for BTC:

| Column | Formula | Purpose |
|--------|---------|---------|
| `mu6h` | rolling mean of last 6 1h log returns | Short-term directional drift |
| `mu12h` | rolling mean of last 12 1h log returns | Medium-term drift |
| `mu24h` | rolling mean of last 24 1h log returns | Daily drift anchor |
| `regime_z` | `clip(ewm(span=12).mean / ewm(span=24).std, −3, 3)` of 1h log returns | Trend-to-noise ratio |
| `z_drift_6h` | empirical z-drift from last 6h of resolved contracts | Realized price-to-strike drift |

These feed into `z_drift` computation:
```
z_drift_yes = mu6h×w1 + mu12h×w2 + mu24h×w3 + regime_z×scale
```
The model shifts the log-normal distribution's mean before computing P(settle > strike).

---

## Part 3: BTC Drift Model Sweep — How We Found the Current Formula

### Background

The drift model answers: *given where price is trending, how much does that shift the log-normal probability?* Over the system's history there were four model eras. The sweep described here is what produced the current Era 4 formula (live May 17, 2026+).

---

### The Base Formula

Every version of the drift model works by shifting the z-score of the log-normal distribution:

```
p_model = 1 - Φ(log(K/S) / σ·√τ  −  z_drift)
```

`z_drift > 0` tilts the distribution toward YES (bullish). `z_drift < 0` tilts toward NO (bearish). Without drift the model is a pure log-normal: `z_drift = 0`.

---

### Era History

| Era | Period | Formula | BTC Performance |
|-----|--------|---------|----------------|
| 1 — Pure log-normal | pre-Apr 10 | `z_drift = 0` | Edge = −8.9% to −17.2% |
| 2 — Composite table | Apr 10–19 | `z_drift = Φ⁻¹(p_up)`, k=1.0 | Marginally positive |
| 3 — Drift multiplier | Apr 20–May 16 | `z_drift = Φ⁻¹(p_up) × k`, BTC k=1.40 | Best period; degraded late |
| **4 — Rolling mu drift** | **May 17+** | **`z_drift = f(mu6h, mu24h, regime_z)`** | **WR=52.8%, +$9,832** |

---

### The Sweep: Why Rolling Mu Beat the Lookup Table

The lookup table (Era 3) encoded p_up from a static calibration file. As the market shifted, stale calibration caused systematic bias. The hypothesis was: *recent actual price drift is a better signal than a month-old probability table.*

**Three models were head-to-head simulated on 751 decision slots (9,684 candidate contracts):**

| Model | Trades | WR | YES n / WR | NO n / WR | Total PnL |
|-------|--------|----|-----------|----------|----------|
| No Drift (pure log-normal) | 464 | 47.8% | 115 / 60.0% | 349 / 43.8% | −$1,019 |
| **Current (6h rolling mu)** | **544** | **52.8%** | **207 / 61.4%** | **337 / 47.5%** | **+$9,832** |
| Model A (p_up_v2, k=1.14) | 441 | 48.1% | 425 / 49.2% | 16 / 18.8% | +$1,645 |

Rolling mu dominated by **+$8,187 vs p_up_v2** and **+$10,851 vs no drift**.

---

### The YES Drift Formula (Current)

```
_sq          = sqrt(tau / 60)                     # time scale: tau in minutes, normalize to hours
sigma_tau    = vol_eff × sqrt(tau)                # total vol over contract life

z_drift_yes  = (mu6h + mu24h) × (tau/60) / sigma_tau   # drift contribution from recent returns
             + regime_z × _sq                            # trend-strength contribution
             + (composite_trend / 5.0) × 0.15 × _sq     # composite bias (small anchor)
```

- `mu6h`: rolling mean of last 6 hourly log returns (short-term momentum)
- `mu24h`: rolling mean of last 24 hourly log returns (daily anchor)
- `regime_z`: `clip(ewm(span=12).mean / ewm(span=24).std, −3, 3)` — trend-to-noise ratio
- `composite_trend`: discrete score [−5, +5] from technical indicators, normalized and dampened to 15% contribution

---

### The NO Drift Formula (Current)

NO uses an **independent** formula — it is NOT simply `1 − p_yes`. This prevents the YES model's bullish bias from mechanically creating incorrect NO edges.

```
vol_eff_no   = garch_ve  (GARCH conditional vol, if available)
               else vol_eff  (blended Deribit/realized fallback)

sigma_tau_no = vol_eff_no × sqrt(tau)
_sq_no       = sqrt(tau / 60)

z_drift_no   = (mu6h + mu12h + mu24h) × (tau/60) / sigma_tau_no   # three-window drift
             + regime_z × _sq_no
             + Φ⁻¹(p_up_v2) × 1.14 × _sq_no                       # p_up_v2 ML model contribution
```

Key differences from YES:
- Adds `mu12h` (medium-term window, not in YES formula)
- Replaces `composite_trend` anchor with the ML model (`p_up_v2`, k=1.14)
- Uses GARCH conditional vol as sigma override when available (more responsive than realized)

---

### The p_up_v2 Calibration Sub-Sweep

`p_up_v2` is an LGBM model that predicts 1h directional probability. Its contribution to the NO drift formula was calibrated separately by testing whether a vol factor should scale its z_drift contribution.

**Nine configurations tested (log-loss on 549,331 calibration records):**

| Config | k | Log-loss | Δ vs baseline |
|--------|---|----------|--------------|
| **A — no vol factor** | **1.1431** | **0.53191** | **baseline (winner)** |
| D+ vol_trend × | 1.048 | 0.53650 | +0.00460 |
| D− vol_trend / | 0.996 | 0.53822 | +0.00632 |
| E+ term_spread × | 0.978 | 0.54431 | +0.01241 |
| G+ bb_expansion × | 0.852 | 0.54617 | +0.01426 |
| F− momentum_conf / | 0.519 | 0.54761 | +0.01570 |

Vol factors tested:
- **D vol_trend**: `sigma_24h / sigma_72h` — vol acceleration (short vs medium realized vol)
- **E term_spread**: `sigma_6h / sigma_168h` — short vs long realized vol (term structure proxy)
- **F momentum_conf**: `|net_6h_return| / sigma_6h` — momentum clarity (z-score of recent trend)
- **G bb_expansion**: `bb_width_24h / bb_width_72h_mean` — Bollinger Band expansion ratio

All factors tested in both forward (×) and inverse (/) directions. Every variant was worse than the plain `k=1.1431` with no vol scaling. Conclusion: p_up_v2's z_drift contribution should not be modulated by the vol regime — it already accounts for regime implicitly through its LGBM features.

---

### What Was Validated Before Going Live

1. **Simulation on paper_trades.csv** (9,684 rows, 751 decision slots): head-to-head vs no-drift and p_up_v2 baselines
2. **Causal check**: rolling mu drift fires when there is actual recent price directional momentum, not just model conviction — avoids stale calibration table bias
3. **Breakeven verification**: both YES and NO sides showed WR above their respective breakeven win rates in the winning model
4. **Versioned backup created**: `paper_trade_runner_pre_branched_drift.py` before going live

---

## Part 4: Deep Gate Analysis — Process for an AI

This is the exact methodology used to improve trading logic. Follow all six phases in order. Do not collapse or skip phases.

---

### Phase 1 — Baseline Audit

Run on both runners (15m and 1h) for the target asset. Report:

- Total resolved n, win rate (WR), breakeven WR, total P&L
- YES side: n, WR, breakeven WR, P&L
- NO side: n, WR, breakeven WR, P&L
- Weekly P&L trend (last 4–6 weeks) to spot regime shifts

**Key definition**: Breakeven WR = `p_market` for YES bets, `1 - p_market` for NO bets. Win rate alone is meaningless without the breakeven comparator.

If weekly deterioration is severe (e.g., −$200/week trend), diagnose root cause before proposing gates. A regime shift needs different treatment than a bad feature.

---

### Phase 2 — Gate Candidate Identification

Segment resolved trades by signal features to find losing populations:

- **Target**: WR significantly below breakeven WR, `n ≥ 20`
- **Report per candidate**: feature name, condition, n, WR, breakeven WR, P&L impact
- **Sort by**: total P&L lost (largest dollar losses first, not just worst WR)
- **Required**: a causal explanation — why does this condition mechanically predict the bet losing?

Example of valid causal logic:
> "GARCH ratio > 1.5 → elevated realized vol → model underestimates probability of reaching strike → YES bets systematically lose"

Example of invalid correlation logic:
> "Hour 14 UTC has low WR" — no mechanism; likely noise unless n is very large

---

### Phase 3 — Exhaustive Rescue Search

For **every** gate candidate, search the blocked population for rescue conditions before declaring a hard block.

- **Rescue threshold**: WR > 65% AND n ≥ 8 within the blocked population
- **Minimum search breadth**: 8–12 feature combinations, including cross-features (e.g., `pm < 0.35 AND ema_stack = −1`)
- **Critical check**: Verify the rescue condition can exist simultaneously with the gate trigger

**The impossible rescue trap**: If the gate fires on condition A and the rescue fires on condition NOT-A, the rescue can never trigger. Example:
- Gate: `rvol < 0.80` (low volume)
- Broken rescue: `rvol > 1.0` — this condition is logically excluded by the gate trigger

If no rescue meets threshold → hard block; note this explicitly.
If rescue has n < 8 → note as "promising, revisit at n=30+".

---

### Phase 4 — Logic Validation

Before presenting, validate each gate and rescue for causal soundness:

**Gate checks:**
- Does the condition mechanically explain why the bet loses?
- Is the signal a coincidence or a structural cause?

**Rescue checks:**
- Does the rescue condition mechanically explain why the bet recovers within the blocked population?
- Does the rescue direction make sense? (A bearish EMA rescue on a bearish YES gate often means "very deep ITM" — valid. Same rescue on a bearish NO gate is internally contradictory.)

**Red flags:**
- Rescue is the logical complement of the gate condition (impossible)
- Rescue has n < 8 (too thin)
- Rescue direction contradicts gate direction without a clear structural reason

---

### Phase 5 — Presentation for Approval

Present in a table per runner with:

| Gate | Condition | Rescue | Blocked n | WR | BE WR | Est. P&L saved |
|------|-----------|--------|-----------|-----|-------|---------------|

Then provide exact code blocks (not pseudocode), including print statements for logging. Wait for explicit approval before editing any file.

---

### Phase 6 — Implementation

After approval:
1. Create a versioned backup: `<runner>_pre_<asset>_gates_<YYYYMMDD>.py`
2. Insert gates in logical order:
   - YES gates before NO gates
   - Harder blocks before softer Kelly dampeners
3. Run `python3 -m py_compile <runner>.py` to verify syntax
4. Log all new signals to both the accepted-trades CSV and the blocked-trades CSV in the same change — never add a gate without logging it

---

### Key Rules

- **Wins blocked + losses blocked + net P&L delta**: Always report all three. A gate that saves $500 in losses but blocks $600 in wins is a net loser.
- **Never block on market price alone**: The model exists to find edge where the market misprices. `pm = 0.70` is not a reason to skip a bet — it's the model's starting point.
- **Calibration error is a diagnostic, not an optimization target**: Tune model parameters against backtest P&L with the full gate stack and Kelly sizing active.
- **Correlated signals carry conviction**: Don't drop features because they correlate with others. Test ensemble agreement-count as a meta-feature.
- **Never implement without approval**: Present first, wait for explicit sign-off, then edit.

---

## Appendix: Data Fields Reference

### 1h Runner (`paper_trades_btc.csv` and equivalents)

| Field | Type | Description |
|-------|------|-------------|
| `markov_regime_daily` | Bull/Bear/Sideways | BTC 20-day macro regime |
| `mu6h` | float | 6-bar rolling mean of 1h log returns |
| `mu12h` | float | 12-bar rolling mean of 1h log returns |
| `mu24h` | float | 24-bar rolling mean of 1h log returns |
| `regime_z` | float (−3 to 3) | EWM trend-to-noise ratio from 1h log returns |
| `z_drift_6h` | float | Empirical realized z-drift over last 6h of contracts |
| `rvol_1h` | float | Current 1h volume / 30-bar same-hour average |
| `adx_1h` | float | 14-period ADX on 1h bars (trend strength) |

### 15m Runner (`paper_trades_btc15m.csv`)

| Field | Type | Description |
|-------|------|-------------|
| `markov_regime_1h` | Bull/Bear/Sideways | BTC 20-hour regime (1h timeframe) |
| `markov_regime_15m` | Bull/Bear/Sideways | BTC 20-bar regime (15m timeframe) |
| `markov_eth_daily` | Bull/Bear/Sideways | ETH 20-day macro regime |
| `markov_sol_6h` | Bull/Bear/Sideways | SOL 20-bar 6h regime |
| `markov_sol_4h` | Bull/Bear/Sideways | SOL 20-bar 4h regime |
| `markov_sol_1h` | Bull/Bear/Sideways | SOL 20-bar 1h regime |
| `mu6h` | float | 6-bar rolling mean of 1h log returns |
| `mu12h` | float | 12-bar rolling mean of 1h log returns |
| `mu24h` | float | 24-bar rolling mean of 1h log returns |
| `regime_z` | float (−3 to 3) | EWM trend-to-noise ratio from 1h log returns |
| `z_score` | float | Per-contract moneyness: log(K/S) / σ·√τ |
| `bb_pct_1h` | float (0–1) | Bollinger Band %B on 1h close (0=lower band, 1=upper band) |
| `ema20_dist_1h` | float % | Distance from 1h close to 20-period EMA (%) |
| `ema50_dist_1h` | float % | Distance from 1h close to 50-period EMA (%) |
| `stoch_k_4h` | float (0–100) | Stochastic %K on 4h bars |
| `rsi_4h` | float (0–100) | RSI on 4h bars |
| `chg_4h` | float % | 4h bar close-to-open return (%) |
| `bp_4h` | float (0–1) | 4h bar buy pressure: (close−low)/(high−low) |

---

*Generated 2026-05-25 from live system at `/Users/justindehn/Documents/ClaudeCode/kalshi_btc/`*
