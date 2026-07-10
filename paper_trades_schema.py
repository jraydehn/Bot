"""
paper_trades_schema.py

Single source of truth for the paper_trades.csv (and paper_trades_{eth,sol}.csv)
column schema. Both paper_trade_runner.py (writer) and outcome_checker.py
(settlement rewriter) must use this SAME list.

2026-07-02: outcome_checker.py previously kept its own separate, stale
CSV_COLUMNS (108 entries vs the runner's 171). Its settlement rewrite uses
csv.DictWriter(..., extrasaction="ignore"), so every settlement cycle
silently stripped the 63 columns missing from its list (all CoinGlass/HL
flow data, every HMM shadow state, OU/Kalman/Hurst signals, flag/H&S
patterns) from EVERY row in the file, including rows written correctly
moments earlier. Root cause of the persistent "shadow column is always
empty" mystery for paper_trades.csv. Single shared list prevents recurrence.
"""

CSV_COLUMNS = [
    "logged_at",
    "decision_time",
    "contract_ticker",
    "close_ts",
    "spot",
    "strike",
    "offset_pct",
    "p_market",
    "p_market_source",
    "p_yes_model",
    "z_score",
    "vol_60m",
    "vol_60m_model",
    "vol_implied_kalshi",
    "vol_ratio",
    "spread",
    "vol_eff",
    "structure_bias",
    "confirmation_bias",
    "confirmation_score",
    "no_score",
    "obi_score",
    "obi_raw",
    "obi_exchanges",
    "vpin_score",
    "vpin_raw",
    "funding_bias",
    "avg_funding_rate",
    "vol_score",
    "cmf_raw",
    "cmf_score",
    "vwap_score",
    "vwap_signal",
    "vwap_total",
    "vwap_stretch_score",
    "vwap_distance_pct",
    "bearish_rejection",
    "bullish_rejection",
    "ema_stretch_score",
    "stoch_bias",
    "stoch_k",
    "stoch_k_4h",
    "stoch_d",
    "stoch_crossover_active",
    "ema_stack_bias",
    "ema_alignment",
    "z_shift",
    "direction_strength",
    "raw_edge",
    "net_edge",
    "decision",
    "side",
    "neutral_gate",    # True if trade passed via neutral structure path (+0.02 edge premium)
    "pure_edge_gate",  # True if trade passed via pure-edge override (Gate P, 1/8 Kelly)
    "contracts_scanned",  # number of contracts with real bid/ask evaluated at this decision point
    "tau_minutes",        # minutes to expiry at decision time (used in probability engine)
    "gate_blocked",       # which gate blocked a no_trade (Gate 1/2/3); empty for trades
    "kelly_fraction",
    "bet_fraction",
    "bet_amount",
    "bankroll",
    "composite_trend",    # trend score from composite_scorer (-6 to +6)
    "composite_rev",      # reversion score from composite_scorer (-15 to +15)
    "composite_p_up",     # calibrated directional probability from composite scorer (lookup table)
    "p_up_v2",            # BTC p_up v2 LightGBM model value (overrides composite_p_up for BTC)
    "chg_30m",            # 30-minute price change fraction at decision time
    "chg_10m",            # 10-minute price change fraction at decision time
    "chg_5m",             # 5-minute price change fraction at decision time
    "bp_5m",              # buying pressure on last completed 5m bar: (close-low)/(high-low)
    "bp_1h",              # buying pressure on last completed 1h bar: (close-low)/(high-low)
    "chg_1h",             # 1-hour close pct-change (%) at decision time
    "chg_2h",             # 2-hour cumulative pct-change (%) — sustained rally/selloff detection
    "chg_3h",             # 3-hour cumulative pct-change (%) — regime momentum
    "body_15m",           # body ratio on last completed 15m bar: |close-open|/(high-low)
    "dir_15m",            # direction of last completed 15m bar: +1=bullish, -1=bearish
    "p_gbdt",             # BTC/ETH/SOL LGBM shadow model probability [SHADOW — gate eval after 2,000+ scan archive rows + retrain]
    "sharp_move_active",  # True if sharp move inversion was applied this cycle
    "smc_4h",             # SMC 4h structure: bullish / bearish / neutral
    "smc_1h",             # SMC 1h structure: bullish / bearish / neutral
    "choch_1h",           # True if 1h ChoCH fired in the last 5 bars (regime flip)
    "choch_4h",           # True if 4h ChoCH fired in the last 3 bars (regime flip)
    "supply_pct",         # % above nearest supply zone (None if no zone)
    "demand_pct",         # % below nearest demand zone (None if no zone)
    "in_supply_zone",     # True if price is currently inside a supply zone
    "in_demand_zone",     # True if price is currently inside a demand zone
    "stoch_flipped",      # retained for backward compatibility
    "squeeze_1h",         # True if BB width < KC width (volatility compression before breakout)
    "kc_pct_1h",         # Keltner Channel position: (close - mid) / (upper - mid); <0 = below midline [LIVE gate: ETH 15m NO block]
    "kc_bo_1h",          # Keltner Channel breakout: +1 = above upper band, -1 = below lower band, 0 = inside
    "adx_1h",            # 14-period ADX on 1h bars (trend strength; >25=trending, <20=ranging)
    "rvol_1h",           # relative volume: current 1h vol / 30-bar avg for this UTC hour
    "pm_drift_5m",       # p_market change over last 5 minutes for this contract
    "hour_utc",          # UTC hour of decision (0-23) — calendar seasonality analysis
    "liq_score",         # Coinalyze: composite liquidation+positioning score (-2 to +2)
    "liq_bias",          # Coinalyze: (short_liqs - long_liqs) / total_liqs; +1=squeeze, -1=cascade
    "ls_long_pct",       # Coinalyze: % of open perp positions that are long (crowding signal)
    "oi_chg_pct",        # Coinalyze: open interest % change over last completed 15m bar
    "cvd_4h",                # Binance.us spot: 4h cumulative volume delta (taker buy - sell USDT); +ve = net buying [SHADOW]
    "cg_futures_delta_4h",   # CoinGlass futures (Binance+OKX+Bybit perps): buy-sell USD, last completed 4h bar [SHADOW]
    "cg_futures_ratio_4h",   # CoinGlass futures: buy/sell ratio last 4h bar; >1 = net buying [SHADOW]
    "cg_futures_cvd_12h",    # CoinGlass futures: rolling 12h cumulative delta (3×4h bars) [SHADOW]
    "cg_spot_cb_ratio_4h",   # CoinGlass spot CVD (Binance+Coinbase+OKX): buy/sell ratio last 4h bar [LIVE gate]
    "cg_liq_ratio_4h",       # CoinGlass agg liquidation: long_usd/short_usd last 4h bar; <0.7 blocks, >5 boosts [LIVE gate]
    "cg_liq_total_4h",       # CoinGlass agg liquidation: total USD (long+short) last 4h bar; >$60M boosts [LIVE gate]
    "hl_ls_ratio",           # HL whale long_value / short_value; >1=net long [SHADOW — gate eval after 6wks]
    "hl_squeeze_idx",        # HL (short_liq_near - long_liq_near) / total; +ve=squeeze risk [SHADOW]
    "hl_liq_ratio_4h",       # HL-only rolling 4h long_liq / short_liq [SHADOW]
    "all_liq_ratio_1h",      # All-exchange rolling 1h long_liq / short_liq [SHADOW]
    "all_liq_ratio_4h",      # All-exchange rolling 4h long_liq / short_liq (broader than gate) [SHADOW]
    "max_pain_nearest",      # Deribit max pain price of nearest expiry [SHADOW — gate eval after 6wks]
    "max_pain_dist_pct",     # (max_pain_nearest - spot) / spot * 100; +ve = max pain above spot [SHADOW]
    "opt_pc_ratio",          # put_notional / call_notional of nearest Deribit expiry; >1 = bearish [SHADOW]
    "arima_forecast_1h",    # ARIMA(2,0,1) 1-step-ahead forecast of next 1h log return [SHADOW — gate eval after 500+ obs; fixed 2026-06-02 (disp arg)]
    "markov_regime_daily",   # BTC 3-state HMM: Bull / Bear / Sideways
    "markov_regime_7state",  # BTC 7-state HMM: Correction / Consolidation / Bull / etc.
    "ob_imbalance",      # Coinbase spot OB: (bid-ask)/(bid+ask) in 0.5% window around strike [SHADOW — gate eval after 200+ obs]
    "ob_path_ask_usd",   # USD ask notional between spot and strike (OTM YES resistance to clear) [SHADOW — same batch as ob_imbalance]
    "ob_path_bid_usd",   # USD bid notional between strike and spot (ITM YES / OTM NO floor support) [SHADOW — same batch as ob_imbalance]
    "ob_ask_frac",       # ask_mass at strike / total book ask (normalized resistance) [SHADOW — same batch as ob_imbalance]
    "ob_bid_wall_pct",   # distance to nearest $500k+ bid wall below spot (fraction of spot, negative) [SHADOW — same batch as ob_imbalance]
    "ob_ask_wall_pct",   # distance to nearest $500k+ ask wall above spot (fraction of spot, positive) [SHADOW — same batch as ob_imbalance]
    "hmm_vol_state",     # hard Viterbi rank: 0=R0 low-vol, 1=R1 high-vol
    "hmm_r1_prob",       # soft posterior P(R1|data) 0-1; catches transitions ~5 bars early [SHADOW]
    "hmm_vol_k10",       # 10-step forward P(R1): posterior @ P^10 [SHADOW]
    "hmm_time_in_state", # sojourn depth in bars: early=1-3 (spike), mid=4-15, deep=16+ (committed) [SHADOW]
    "ou_z_score",        # OU AR(1) fit: (spot - ou_mean) / ou_sigma; +ve=extended up, -ve=extended down [SHADOW]
    "ou_halflife_min",   # OU expected reversion half-life in minutes [SHADOW]
    "ou_tau_drift",      # OU expected log-return over contract tau: mu+(spot-mu)*exp(-theta*tau_h); tau-aware [SHADOW]
    "hs_pattern_type",   # most recent H&S pattern on 1h: 'hs' (bearish) or 'ihs' (bullish) [SHADOW — gate eval after 200+ obs]
    "hs_bars_since_break", # 1h bars elapsed since that pattern broke [SHADOW]
    "hs_r2",             # H&S pattern R² fit quality [SHADOW]
    "hs_neck_slope",     # H&S neckline slope (log price / bar) [SHADOW]
    "hs_head_height",    # H&S head height in log price units [SHADOW]
    "hs_head_width",     # H&S head width in bars [SHADOW]
    "flag_signal",        # +1=recent bull flag/pennant, -1=bear, 0=none [SHADOW — gate eval after 200+ obs]
    "flag_bull_bars_ago", # 1h bars since last confirmed bull flag/pennant (-1=none in lookback)
    "flag_bear_bars_ago", # 1h bars since last confirmed bear flag/pennant (-1=none)
    "flag_bull_tip_y",    # real price at top of bull pole
    "flag_bear_tip_y",    # real price at bottom of bear pole
    "flag_bull_pole_pct", # bull pole height as % of base price
    "flag_bear_pole_pct", # bear pole depth as % of base price
    "pip_last_slope",     # slope of last PIP segment on 1h (log-price/bar); +ve=up, -ve=down [SHADOW]
    "pip_up_frac",        # fraction of total PIP amplitude from upward legs [0,1] [SHADOW]
    "pip_n_turns",        # direction changes in 5-PIP skeleton [0,4] [SHADOW]
    "v_hawk",             # Hawkes vol intensity on 1h norm_range (kappa=0.01, lb=336) [SHADOW — gate eval after 200+ elevated-regime trades]
    "hawk_vol_regime",    # rolling vol regime: quiet/low/mid/elevated/spike (q25/50/75/90 thresholds) [SHADOW]
    "pc1_rsi",            # PC1 RSI divergence score: fast-vs-slow RSI (2-24 periods); gate fires NO block when <= -34.93
    # Shadow stochastic signals (log-return based, complement to log-price OU above)
    "ou_theta",          # log-return AR(1) OU mean-reversion speed per hour (48-bar window) [SHADOW]
    "hurst_exponent",    # R/S Hurst exponent (64 1h bars); H>0.5=trending, H<0.5=mean-reverting [SHADOW]
    "autocorr1_15",      # lag-1 autocorr of 1h log-returns (30-bar window) [SHADOW]
    "autocorr1_30",      # lag-1 autocorr of 1h log-returns (60-bar window) [SHADOW]
    "kalman_velocity",   # Kalman-filtered 1h return trend (constant-velocity model) [SHADOW]
    "kalman_residual",   # actual minus Kalman-filtered value (mean-reversion signal) [SHADOW]
    "hmm_ms_state",  # microstructure HMM state (0-7); gates: BTC NO St6, ETH YES St0, SOL NO St4
    "hmm_ms_prob",   # posterior P(current state | observations) [SHADOW]
    "hmm_of_state",  # order-flow HMM state (0-N); gate: BTC NO St2 (stale crowded-long)
    "hmm_of_prob",   # posterior P(current of-state | observations) [SHADOW]
    "hmm_vd_state",  # vol+direction HMM state (0-N); gate: SOL NO St4 conviction, SOL YES St7 block
    "hmm_vd_prob",   # posterior P(current vd-state | observations) [SHADOW]
    "hmm_pnl_state", # P&L regime HMM state [SHADOW — 0=active 1=degraded; gate after 60+ days]
    "hmm_ps_state",  # Phase-space trajectory HMM state [SHADOW — 6 states; gate after 50+ per state]
    "hmm_gd_state",  # Gate-density HMM state [SHADOW — 5 states; St1=hostile YES-block, St3=NO-dominant]
    "hmm_zdrift_state",  # Z-Drift HMM state (0-5); gate: BTC YES St2+drift<-0.001+not_R1 rescue:bp1h>=0.60
    "resolved_yes",   # filled by outcome_checker.py
    "would_win",      # filled by outcome_checker.py
    "would_pnl",      # filled by outcome_checker.py
    "spot_at_expiry", "price_move_pct", "miss_pct",  # filled by outcome_checker.py
    "loss_margin_pct", "loss_category",              # filled by outcome_checker.py; tau-scaled quality labels
    # 2026-07-04: honest p_up rebuild (btc_p_up_v3_model.py) — SHADOW ONLY.
    # Market-level score, one value per cycle, BTC only. NO decision path may
    # read this column until shadow data confirms the replay (approved step).
    "p_up_v3",       # BTC honest v3 hour-level p_up [SHADOW — keep at end for header order]
    # 2026-07-06: regime HMM built on p_up_v3's own level/momentum/6h-trend.
    # Backfilled against 2,995 real taken BTC hourly trades (Apr-Jul, 8-13
    # distinct weeks per state): rising+YES loses (-$2,234), rising+NO wins
    # (+$2,404), crashing+YES wins (+$1,255), crashing+NO loses (-$1,189).
    "pup_v3_hmm_state",  # "rising"/"neutral"/"crashing", BTC only
    # 2026-07-06: honest ETH p_up rebuild (eth_p_up_v1_model.py) — SHADOW ONLY.
    # Asset-specific feature set (A16+C13), NOT BTC's shape. NO decision path
    # reads this pending the same real-trade backfill validation BTC's v3 got.
    "eth_p_up_v1",
    # 2026-07-06: honest SOL p_up rebuild (sol_p_up_v1_model.py). Asset-specific
    # feature set (A16 ONLY -- neither BTC's nor ETH's shape). Backfilled +
    # gated same day (sol_pup_v1_agreement_gate).
    "sol_p_up_v1",
    # 2026-07-08: CoinGlass flow-regime HMM state (BTC hourly, 0-6). Drives
    # btc_cg_flow_no_gate (block NO in states 1/2/5 unless kc rescue).
    "cg_flow_state",
    # 2026-07-08: 5m Bollinger bandwidth (BTC hourly). Drives
    # hmm_pup_v3_crashing_no_gate's rescue (bb_width_5m>=0.0077 lets NO trade).
    "bb_width_5m",
    # 2026-07-09: SOL hourly VWAP MTF HMM state (0-7; blank for BTC/ETH).
    # Drives sol_1h_vwap_s2_no_gate (block) + sol_1h_vwap_s3_no_boost (x1.25).
    "vwap_1h_state",
    # 2026-07-09: macro HMM regime posteriors (BTC only; the p_up calibration
    # blend weights). Never logged anywhere before -- the scan-archive version
    # was dead code, which forced the stale-parquet reconstruction mistake.
    "macro_regime_bull",
    "macro_regime_sdwy",
    "macro_regime_bear",
]
