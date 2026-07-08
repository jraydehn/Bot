---
name: comprehensive_rescue
description: Exhaustive rescue-search methodology for a losing gate/bucket in the kalshi_btc trading bot — every signal category, causal reconstruction of anything missing from the CSV, coverage verification, and formal significance testing. Use whenever asked to find a rescue for a blocked/losing population, or to validate that "no rescue exists."
---

# comprehensive_rescue

This project has repeatedly found that a "thorough" rescue search which stops
at 15-20 hand-picked signals misses the real rescue, or wrongly concludes none
exists. This skill exists because the user has had to demand this rigor
multiple times (2026-07-06, BTC p_up_v3 gates and again for ETH p_up_v1) and
does not want to keep re-explaining it. Follow every step below without being
asked. Do not summarize this checklist back to the user before running it —
run it, then report results.

## When to use

Any time you are:
- Deciding whether a blocked/losing population (a gate's blocked bucket, an
  "agree" bucket, any population with negative edge) has a rescuable subset.
- Asked "does a rescue exist" or "check for rescues."
- Tempted to report "no rescue found" after testing fewer than ~100 signals.

## The 5 mandatory phases

### Phase 1 — Enumerate EVERY column, not a memorized shortlist
Load every column from the relevant paper_trades*.csv (or scan archive), MINUS
a small exclude-list of pure bookkeeping/outcome fields (logged_at, ticker,
kelly_fraction, bet_amount, bankroll, decision, side, would_win, would_pnl,
resolved_yes, spot_at_expiry, price_move_pct, miss_pct, loss_category — these
are metadata or the target itself, not candidate signals). Everything else is
a candidate: composite_trend/rev/p_up, ema_stack/stretch, vwap variants,
stoch_k/stoch_d/stoch_bias/stoch_crossover_active/stoch_flipped/stoch_k_4h,
rsi variants, bp_1h/bp_5m, chg_* (all timeframes), dir_*, offset_pct,
vol_60m/vol_ratio/vol_eff/vol_implied_kalshi/rvol_1h, structure_bias,
confirmation_bias/score, no_score, vpin_score/raw, obi_score/raw, funding_bias,
liq_score/bias, cg_* (CoinGlass), ou_theta/hurst_exponent/autocorr1_15/
autocorr1_30/kalman_velocity/kalman_residual, ADX, MACD, all hmm_* columns,
markov_regime_*, smc/supply/demand fields, hs_* (head-and-shoulders), flag_*,
pip_*, v_hawk/hawk_vol_regime, arima_forecast_1h, kc_pct_1h/kc_bo_1h (Keltner),
p_gbdt, opt_pc_ratio/max_pain_*. If in doubt, include it — the cost of testing
a useless column is cheap; the cost of skipping a real one is what this skill
exists to prevent.

For each candidate: if boolean or ≤6 unique values, test each category
directly (n per category, WR, edge, PnL). Otherwise sweep a full DECILE grid
(quantiles 0.1 through 0.9), both directions (`>=` and `<`), not just
median/quartiles.

### Phase 2 — Verify coverage BEFORE trusting "tested"
A column with <20 non-null rows in the population under test was NOT
meaningfully tested even if your sweep loop technically iterated over it.
Explicitly print/log a coverage table (column, non-null count, tested Y/N)
and treat any column below the coverage floor as **silently skipped** unless
you go reconstruct it (Phase 3). This exact bug — a sweep silently skipping
thin columns with the aggregate summary implying "everything was checked" —
is what triggered the creation of this skill. Never let "N total columns
swept" imply "N total columns meaningfully tested."

### Phase 3 — Reconstruct anything missing or thin, don't skip it

**ZERO-LOOKAHEAD RULE (added 2026-07-08 after it invalidated three deployed
gates in one day):** any reconstructed signal or regime-state join MUST use
only bars whose CLOSE time is <= the trade's decision timestamp. The lazy
pattern `idx = index.searchsorted(ts, "right") - 1` selects the bar
CONTAINING ts — complete in a historical parquet (includes up to a full bar
of future data) but partial in live reality. For a 15m contract, a signal
computed on the completed containing bar essentially IS the outcome (a
"rescue" found this way showed 100% WR / P=0.000 and collapsed to -26pp
when computed honestly). Correct pattern:
`cutoff = ts - bar_duration; idx = index.searchsorted(cutoff, "right") - 1`.
Same rule for regime-state series indexed by bar OPEN time: the state is
only knowable at open + bar_duration; join on that effective time, never on
the open. A 100%-WR or P=0.000 rescue on a small n is a lookahead alarm,
not a discovery — re-derive it zero-lookahead before believing it.
Two known root causes for thin/missing coverage in this project, both fixable:
1. **The signal was never logged as a CSV column at all** (GARCH, macro
   regime Bull/Sideways/Bear, MACD, Donchian have all hit this). Fix: recompute
   causally from the asset's own historical 1h price parquet
   (`reform_results/pup_v2_rebuild_20260704/hist_<SYM>USDT_1h.parquet` or
   equivalent), using the EXACT formula already live in `paper_trade_runner.py`
   (grep for the variable name to find it — do not invent a different
   formula). GARCH: refit `arch_model(..., vol="Garch", p=1, q=1)` on trailing
   500-bar log returns. ARIMA: `ARIMA(log_returns, order=(2,0,1)).forecast(1)`
   on a trailing window. Keltner: EMA10 ± 1.5×ATR14. Donchian: 20-bar
   high/low position. Macro regime: if no asset-specific HMM exists, the
   BTC-trained one can be applied to the target asset's own return features
   as a *disclosed cross-asset proxy* — label it as such, don't imply it's
   asset-native.
2. **The signal exists but only reliably logs from a recent date forward**
   (this happened to ms/vd/of HMM states across BTC, ETH, and SOL alike —
   real coverage only starts ~2026-07-03 because of the degenerate-decode fix
   that same day; anything backfilled from before that date sees ~0-2%
   coverage). Fix: retroactively decode the state for every historical
   timestamp using the asset's own trained model file (`models/hmm_<kind>_
   <asset>.pkl`) and the validated trailing-SEQUENCE decode logic (NOT a
   single-observation `.predict()` — see the ms/vd/of fix, 2026-07-06, for
   why that's degenerate). Test the reconstructed states as INDIVIDUAL
   CATEGORIES (state==0, state==1, ...), never as a continuous threshold —
   a quantile split on a categorical HMM state blends multiple regimes
   together and will hide a state-specific effect.

If a model genuinely does not exist for the target asset (check `models/*_
<asset>.pkl` directly — do not assume symmetry across BTC/ETH/SOL) and
building one from scratch would be a materially larger undertaking than a
rescue search (this has applied to hmm_vol_state/r1_prob, hmm_pnl_state,
hmm_ps_state, hmm_gd_state, hmm_zdrift_state for ETH/SOL — BTC-only
experimental builds), do not silently skip it and do not fabricate a
substitute. State explicitly in the report: "X was not reconstructed because
no <asset>-specific model exists; building one is out of scope for this
search" — disclosure, not omission.

### Phase 4 — Formal significance testing, not "edge>0 and n>=15"
An ad-hoc bar of "positive edge and a double-digit sample size" is not
sufficient and has produced false positives in this project before. For every
candidate that clears a raw positive-edge screen, run BOTH:
- **Trade-level bootstrap**: resample the candidate's per-trade edges
  (`would_win - breakeven`) with replacement (~3000-5000 draws), report
  `P(mean edge <= 0)` and the 95% CI. A CI that includes zero, or a p above
  ~0.10, means the finding is not distinguishable from noise — say so.
- **Week-level bootstrap**: resample the candidate's per-*distinct-week* PnL
  totals (not individual trades — weeks are the natural cluster unit here),
  report `P(mean weekly PnL <= 0)` and the count of weeks with positive PnL
  out of total weeks tested. A candidate winning in fewer than ~2/3 of its
  weeks is fragile even with a positive average.

Report both numbers for every surviving candidate. Do not report only the
aggregate n/WR/PnL and let that imply statistical robustness — that exact
gap (reporting aggregate stats without a significance test) is what this
skill was created to close.

### Phase 5 — Report format
1. State the total test count and how it decomposes (N CSV-column splits + M
   reconstructed-signal splits + K individual HMM-state tests), so "how many
   did you check" never needs to be asked again.
2. List every category from the standard checklist (ARIMA, GARCH, Donchian,
   Keltner, all stochastic variants, offset, price change, direction, buying
   pressure, Kalman, every HMM family, Markov regime) with an explicit
   present/reconstructed/genuinely-unavailable status — not just the ones
   that turned something up.
3. For any surviving candidate(s): n, WR, breakeven edge, PnL, distinct weeks,
   worst-single-week PnL share (overfitting/concentration check), trade-level
   bootstrap p, week-level bootstrap p and win-week fraction.
4. If multiple candidates survive, check pairwise overlap (are they the same
   underlying condition wearing different measurement units?) before treating
   them as independent evidence, and report the UNION's own week-by-week
   breakdown and bootstrap significance — not just each condition's own stats
   in isolation. The union is what a real gate would actually rescue.
5. Give an honest verdict: "robust" (passes both bootstraps cleanly, wins
   most weeks), "modest/directional" (right sign, positive union, but doesn't
   clear conventional significance — say this plainly, don't round it up to
   "found a rescue"), or "none found" (only after Phases 1-4 are actually
   complete, with the disclosure list from Phase 3 attached).

## What NOT to do

- Do not test a "representative" subset of signals and extrapolate.
- Do not let a sweep loop's `if len(sub) < 20: continue` silently determine
  what counts as "tested" without reporting it.
- Do not report an aggregate PnL/edge number as a finding without the
  bootstrap p-value next to it.
- Do not skip reconstructing a missing signal because the CSV doesn't have
  it — that absence is usually fixable from raw price history plus an
  existing formula already live in the runner.
- Do not present a marginal (p>0.10) result with the same confidence as a
  robust one (p<0.01). The user wants the true confidence level stated, not
  smoothed over.
