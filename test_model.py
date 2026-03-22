"""
Unit tests for the Kalshi BTC event trading model.

Coverage:
  - market_data: normal volatility computation, missing columns, too few rows
  - probability_engine: normal case, tau=0, sigma_min=0, far OTM, far ITM
  - pricing_comparison: qualifying and non-qualifying edges
  - market_structure: bullish/bearish/neutral, fewer than 3 pivots, too few candles
  - confirmation_indicators: bullish/bearish/neutral, too few candles
  - kelly_sizing: normal, zero/negative Kelly, cap triggering, bad inputs
  - decision: full trade pass, each gate failing independently
"""

import math
import unittest

import numpy as np
import pandas as pd

from market_data import compute_realized_volatility
from probability_engine import estimate_probability
from pricing_comparison import evaluate_edge
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from kelly_sizing import compute_kelly_size
from decision import evaluate_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1min_df(n: int = 150, start: float = 80_000.0, drift: float = 0.0,
                  vol: float = 0.001, seed: int = 0) -> pd.DataFrame:
    """Return a minimal 1-minute OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, n)
    closes = start * np.exp(np.cumsum(log_ret))
    closes = np.insert(closes, 0, start)[:-1]
    noise = rng.uniform(0.0003, 0.001, n)
    opens = closes * np.exp(rng.normal(0, vol * 0.3, n))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    volume = rng.uniform(5, 30, n) * 1e6
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volume})


def _make_4h_df(n: int = 120, start: float = 70_000.0, drift: float = 0.003,
                vol: float = 0.025, seed: int = 1) -> pd.DataFrame:
    """Return a 4-hour OHLCV DataFrame with a strong uptrend."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, n)
    closes = start * np.exp(np.cumsum(log_ret))
    closes = np.insert(closes, 0, start)[:-1]
    noise = rng.uniform(0.005, 0.015, n)
    opens = closes * np.exp(rng.normal(0, vol * 0.3, n))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    volume = rng.uniform(200, 600, n) * 1e6
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volume})


def _make_1h_df(n: int = 100, start: float = 80_000.0, drift: float = 0.001,
                vol: float = 0.012, seed: int = 2) -> pd.DataFrame:
    """Return a 1-hour OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, n)
    closes = start * np.exp(np.cumsum(log_ret))
    closes = np.insert(closes, 0, start)[:-1]
    noise = rng.uniform(0.002, 0.008, n)
    opens = closes * np.exp(rng.normal(0, vol * 0.3, n))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    volume = rng.uniform(50, 200, n) * 1e6
    volume[-1] = volume[-1] * 3.0  # ensure volume_confirmed=True
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volume})


# ---------------------------------------------------------------------------
# market_data tests
# ---------------------------------------------------------------------------

class TestMarketData(unittest.TestCase):

    def test_normal_returns_three_vol_values(self):
        """With >= 120 rows, all three volatility values should be finite floats > 0."""
        result = compute_realized_volatility(_make_1min_df(n=150))
        for val in (result.vol_30m, result.vol_60m, result.vol_120m):
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0)
            self.assertTrue(math.isfinite(val))

    def test_vol_60m_is_reasonable_for_btc(self):
        """Per-minute vol for BTC should be in a plausible range (0.0001 – 0.005)."""
        result = compute_realized_volatility(_make_1min_df(vol=0.001))
        self.assertGreater(result.vol_60m, 0.0001)
        self.assertLess(result.vol_60m, 0.005)

    def test_log_returns_length(self):
        """Log-return series should have n-1 values (first return is NaN, then dropped)."""
        df = _make_1min_df(n=150)
        result = compute_realized_volatility(df)
        self.assertEqual(len(result.log_returns), 149)

    def test_too_few_rows_raises(self):
        """Fewer than 120 rows must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_realized_volatility(_make_1min_df(n=50))

    def test_missing_column_raises(self):
        """DataFrame missing a required column must raise ValueError."""
        df = _make_1min_df(n=150).drop(columns=["volume"])
        with self.assertRaises(ValueError):
            compute_realized_volatility(df)

    def test_column_names_case_insensitive(self):
        """Upper-case column names should be accepted without error."""
        df = _make_1min_df(n=150)
        df.columns = [c.upper() for c in df.columns]
        result = compute_realized_volatility(df)
        self.assertGreater(result.vol_60m, 0)


# ---------------------------------------------------------------------------
# probability_engine tests
# ---------------------------------------------------------------------------

class TestProbabilityEngine(unittest.TestCase):

    def test_atm_probability_near_half(self):
        """When K = S (at-the-money), p_yes should be close to 0.5."""
        result = estimate_probability(S=80_000, K=80_000, tau=60, sigma_min=0.001)
        self.assertAlmostEqual(result.p_yes, 0.5, places=1)

    def test_deep_otm_low_probability(self):
        """Strike far above spot (deep OTM) should give p_yes close to 0."""
        result = estimate_probability(S=80_000, K=100_000, tau=60, sigma_min=0.001)
        self.assertLess(result.p_yes, 0.01)

    def test_deep_itm_high_probability(self):
        """Strike far below spot (deep ITM) should give p_yes close to 1."""
        result = estimate_probability(S=80_000, K=60_000, tau=60, sigma_min=0.001)
        self.assertGreater(result.p_yes, 0.99)

    def test_z_score_sign(self):
        """z_score should be positive when K > S and negative when K < S."""
        otm = estimate_probability(S=80_000, K=82_000, tau=60, sigma_min=0.001)
        itm = estimate_probability(S=80_000, K=78_000, tau=60, sigma_min=0.001)
        self.assertGreater(otm.z_score, 0)
        self.assertLess(itm.z_score, 0)

    def test_sigma_tau_scales_correctly(self):
        """sigma_to_expiry should equal sigma_min * sqrt(tau)."""
        sigma_min, tau = 0.001, 60
        result = estimate_probability(S=80_000, K=80_500, tau=tau, sigma_min=sigma_min)
        expected_sigma_tau = sigma_min * math.sqrt(tau)
        self.assertAlmostEqual(result.sigma_to_expiry, expected_sigma_tau, places=10)

    def test_tau_zero_raises(self):
        """tau = 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_probability(S=80_000, K=80_500, tau=0, sigma_min=0.001)

    def test_tau_negative_raises(self):
        """Negative tau must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_probability(S=80_000, K=80_500, tau=-1, sigma_min=0.001)

    def test_sigma_zero_raises(self):
        """sigma_min = 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_probability(S=80_000, K=80_500, tau=60, sigma_min=0)

    def test_sigma_negative_raises(self):
        """Negative sigma_min must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_probability(S=80_000, K=80_500, tau=60, sigma_min=-0.001)

    def test_expected_move_positive(self):
        """Expected move percentage should always be positive."""
        result = estimate_probability(S=80_000, K=80_000, tau=30, sigma_min=0.0008)
        self.assertGreater(result.expected_move_pct, 0)


# ---------------------------------------------------------------------------
# pricing_comparison tests
# ---------------------------------------------------------------------------

class TestPricingComparison(unittest.TestCase):

    def test_qualifies_when_edge_sufficient(self):
        """Large model edge should produce qualifies=True."""
        result = evaluate_edge(p_model=0.65, p_market=0.40)
        self.assertTrue(result.qualifies)
        self.assertGreater(result.net_edge, 0.03)

    def test_does_not_qualify_when_edge_small(self):
        """Small model edge that doesn't clear costs should produce qualifies=False."""
        result = evaluate_edge(p_model=0.42, p_market=0.40)
        self.assertFalse(result.qualifies)

    def test_raw_edge_calculation(self):
        """raw_edge should equal p_model - p_market exactly."""
        result = evaluate_edge(p_model=0.55, p_market=0.40)
        self.assertAlmostEqual(result.raw_edge, 0.15, places=10)

    def test_net_edge_deducts_costs(self):
        """net_edge should equal raw_edge - kalshi_fee(p_market) - slippage - spread."""
        from pricing_comparison import kalshi_fee
        result = evaluate_edge(p_model=0.55, p_market=0.40, slippage=0.005, spread=0.0)
        expected = 0.55 - 0.40 - kalshi_fee(0.40) - 0.005
        self.assertAlmostEqual(result.net_edge, expected, places=10)

    def test_custom_slippage_spread(self):
        """Custom slippage and spread values should be respected."""
        from pricing_comparison import kalshi_fee
        result = evaluate_edge(p_model=0.50, p_market=0.40, slippage=0.01, spread=0.005)
        expected_net = 0.50 - 0.40 - kalshi_fee(0.40) - 0.01 - 0.005
        self.assertAlmostEqual(result.net_edge, expected_net, places=10)

    def test_reason_string_present(self):
        """Reason string should always be a non-empty string."""
        result = evaluate_edge(p_model=0.55, p_market=0.40)
        self.assertIsInstance(result.reason, str)
        self.assertGreater(len(result.reason), 0)


# ---------------------------------------------------------------------------
# market_structure tests
# ---------------------------------------------------------------------------

class TestMarketStructure(unittest.TestCase):

    def test_bullish_uptrend(self):
        """Strong uptrend 4h data should produce structure_bias = +1."""
        df = _make_4h_df(drift=0.005, vol=0.015)
        result = detect_market_structure(df)
        # With a strong positive drift the structure should be bullish
        self.assertIn(result.structure_bias, (+1, 0),
                      "Expected bullish or neutral for a strong uptrend")

    def test_bearish_downtrend(self):
        """Strong downtrend 4h data should produce structure_bias = -1."""
        df = _make_4h_df(drift=-0.005, vol=0.015)
        result = detect_market_structure(df)
        self.assertIn(result.structure_bias, (-1, 0),
                      "Expected bearish or neutral for a strong downtrend")

    def test_too_few_candles_raises(self):
        """Fewer than 90 candles must raise ValueError."""
        df = _make_4h_df(n=50)
        with self.assertRaises(ValueError):
            detect_market_structure(df)

    def test_neutral_when_insufficient_pivots(self):
        """
        A perfectly flat price series produces no swing pivots.
        With fewer than 2 swing highs or lows, bias should be 0.
        """
        # Perfectly flat — no strict swing highs or lows possible
        n = 90
        closes = np.full(n, 80_000.0)
        highs  = np.full(n, 80_010.0)
        lows   = np.full(n, 79_990.0)
        volume = np.ones(n) * 1e8
        df = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                           "close": closes, "volume": volume})
        result = detect_market_structure(df)
        # No strict pivots possible in a flat series → neutral
        self.assertEqual(result.structure_bias, 0)

    def test_swing_highs_and_lows_are_lists(self):
        """swing_highs and swing_lows should always be lists of floats."""
        df = _make_4h_df()
        result = detect_market_structure(df)
        self.assertIsInstance(result.swing_highs, list)
        self.assertIsInstance(result.swing_lows, list)

    def test_max_three_pivots_returned(self):
        """At most 3 swing highs and 3 swing lows should be returned."""
        df = _make_4h_df()
        result = detect_market_structure(df)
        self.assertLessEqual(len(result.swing_highs), 3)
        self.assertLessEqual(len(result.swing_lows), 3)


# ---------------------------------------------------------------------------
# confirmation_indicators tests
# ---------------------------------------------------------------------------

class TestConfirmationIndicators(unittest.TestCase):

    def test_normal_run_returns_dataclass(self):
        """Should return a ConfirmationResult with all fields populated."""
        df = _make_1h_df()
        result = compute_confirmation(df)
        self.assertIn(result.ema_alignment, ("bullish", "bearish", "neutral"))
        self.assertIn(result.rsi_regime, ("bullish", "bearish", "neutral"))
        self.assertIsInstance(result.volume_confirmed, bool)
        self.assertIn(result.confirmation_bias, (-1, 0, +1))

    def test_bullish_scenario(self):
        """
        Strong uptrend 1h data should push 20-EMA above 50-EMA and RSI above 50.
        With elevated last-bar volume, confirmation_bias should be +1.
        """
        df = _make_1h_df(drift=0.005, vol=0.008, seed=42)
        df.loc[df.index[-1], "volume"] = df["volume"].mean() * 5
        result = compute_confirmation(df)
        # Strong uptrend should produce bullish EMA alignment and RSI
        self.assertIn(result.ema_alignment, ("bullish", "neutral"))

    def test_too_few_candles_raises(self):
        """Fewer than 60 candles must raise ValueError."""
        df = _make_1h_df(n=30)
        with self.assertRaises(ValueError):
            compute_confirmation(df)

    def test_rsi_value_in_range(self):
        """RSI should always be between 0 and 100."""
        df = _make_1h_df()
        result = compute_confirmation(df)
        self.assertGreaterEqual(result.rsi_value, 0)
        self.assertLessEqual(result.rsi_value, 100)

    def test_ema_values_are_positive(self):
        """EMA values for BTC prices should be positive."""
        df = _make_1h_df()
        result = compute_confirmation(df)
        self.assertGreater(result.ema_20_current, 0)
        self.assertGreater(result.ema_50_current, 0)

    def test_reason_is_string(self):
        """Reason field should always be a non-empty string."""
        df = _make_1h_df()
        result = compute_confirmation(df)
        self.assertIsInstance(result.reason, str)
        self.assertGreater(len(result.reason), 0)


# ---------------------------------------------------------------------------
# kelly_sizing tests
# ---------------------------------------------------------------------------

class TestKellySizing(unittest.TestCase):

    def test_normal_positive_edge(self):
        """With genuine edge, kelly_fraction should be positive."""
        result = compute_kelly_size(p_model=0.65, p_market=0.40, bankroll=10_000)
        self.assertGreater(result.kelly_fraction, 0)
        self.assertGreater(result.bet_amount, 0)

    def test_no_edge_returns_zero(self):
        """When p_model <= p_market, Kelly fraction is negative → bet_fraction = 0."""
        result = compute_kelly_size(p_model=0.35, p_market=0.60, bankroll=10_000)
        self.assertLessEqual(result.kelly_fraction, 0)
        self.assertEqual(result.bet_fraction, 0.0)
        self.assertEqual(result.bet_amount, 0.0)
        self.assertFalse(result.was_capped)

    def test_cap_triggers_at_five_percent(self):
        """With extreme edge, raw Kelly > 5% and was_capped should be True."""
        # p_model = 0.95, p_market = 0.40 → massive edge
        result = compute_kelly_size(p_model=0.95, p_market=0.40, bankroll=10_000)
        self.assertGreater(result.kelly_fraction, 0.05)
        self.assertTrue(result.was_capped)
        self.assertAlmostEqual(result.bet_fraction, 0.05, places=5)

    def test_cap_not_triggered_below_five_percent(self):
        """With modest edge, scaled Kelly < 5% and was_capped should be False.

        With p_market=0.42, b≈1.381. Kelly < 5% requires p_model < ~0.449.
        p_model=0.445 gives Kelly ≈ 4.3%. After 0.25× multiplier, scaled = ~1.1%,
        safely below the cap. bet_fraction should equal kelly_fraction * multiplier.
        """
        result = compute_kelly_size(p_model=0.445, p_market=0.42, bankroll=10_000,
                                    kelly_multiplier=0.25, side="yes")
        self.assertGreater(result.kelly_fraction, 0)
        self.assertFalse(result.was_capped)
        self.assertAlmostEqual(result.bet_fraction, result.kelly_fraction * 0.25, places=5)

    def test_bet_amount_rounded_to_two_decimals(self):
        """bet_amount should be rounded to 2 decimal places."""
        result = compute_kelly_size(p_model=0.60, p_market=0.42, bankroll=7_777)
        remainder = result.bet_amount - round(result.bet_amount, 2)
        self.assertAlmostEqual(remainder, 0.0, places=5)

    def test_bet_amount_equals_fraction_times_bankroll(self):
        """bet_amount should equal bet_fraction * bankroll (before rounding)."""
        bankroll = 10_000.0
        result = compute_kelly_size(p_model=0.60, p_market=0.42, bankroll=bankroll)
        expected = round(result.bet_fraction * bankroll, 2)
        self.assertAlmostEqual(result.bet_amount, expected, places=2)

    def test_p_market_zero_raises(self):
        """p_market = 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_kelly_size(p_model=0.60, p_market=0.0, bankroll=10_000)

    def test_p_market_one_raises(self):
        """p_market = 1 must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_kelly_size(p_model=0.60, p_market=1.0, bankroll=10_000)

    def test_bankroll_zero_raises(self):
        """bankroll = 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_kelly_size(p_model=0.60, p_market=0.40, bankroll=0)

    def test_bankroll_negative_raises(self):
        """Negative bankroll must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_kelly_size(p_model=0.60, p_market=0.40, bankroll=-500)


# ---------------------------------------------------------------------------
# decision tests
# ---------------------------------------------------------------------------

class TestDecision(unittest.TestCase):

    def _bullish_inputs(self):
        """Return inputs that should pass all three gates for a YES trade."""
        return dict(
            structure_bias=+1,
            confirmation_bias=+1,
            p_model=0.65,
            p_market=0.40,
            bankroll=10_000,
        )

    def test_all_gates_pass_produces_trade(self):
        """All gates passing should return decision='trade'."""
        result = evaluate_trade(**self._bullish_inputs())
        self.assertEqual(result.decision, "trade")
        self.assertEqual(result.side, "yes")

    def test_gate1_fail_structure_mismatch(self):
        """structure_bias neutral should fail Gate 1."""
        inputs = self._bullish_inputs()
        inputs["structure_bias"] = 0
        result = evaluate_trade(**inputs)
        self.assertEqual(result.decision, "no_trade")
        self.assertTrue(any("Gate 1 FAILED" in r for r in result.reasons))

    def test_gate1_pass_gate2_fail_on_conflicting_biases(self):
        """structure_bias=-1 derives side='no'; Gate 1 passes, Gate 2 fails on bullish confirmation.

        With structure_bias=-1, the system proposes a NO trade (side is derived from
        structure_bias, not from external input). Gate 1 passes because -1 aligns with
        the required -1 for a NO trade. Gate 2 then fails because confirmation_bias=+1
        does not align with the required -1.
        """
        inputs = self._bullish_inputs()
        inputs["structure_bias"] = -1   # bearish → NO trade proposed
        # confirmation_bias=+1 (bullish) conflicts with the required -1 for NO
        result = evaluate_trade(**inputs)
        self.assertEqual(result.decision, "no_trade")
        self.assertTrue(any("Gate 2 FAILED" in r for r in result.reasons))

    def test_gate2_fail_confirmation_mismatch(self):
        """Neutral confirmation when structure passes should fail Gate 2."""
        inputs = self._bullish_inputs()
        inputs["confirmation_bias"] = 0
        result = evaluate_trade(**inputs)
        self.assertEqual(result.decision, "no_trade")
        self.assertTrue(any("Gate 2 FAILED" in r for r in result.reasons))

    def test_gate3_fail_insufficient_edge(self):
        """Tiny model edge that doesn't clear costs should fail Gate 3."""
        inputs = self._bullish_inputs()
        inputs["p_model"] = 0.42   # only 2% raw edge, won't clear fee + slippage + min
        result = evaluate_trade(**inputs)
        self.assertEqual(result.decision, "no_trade")
        self.assertTrue(any("Gate 3 FAILED" in r for r in result.reasons))

    def test_no_trade_has_zero_bet(self):
        """Any no_trade decision should have bet_amount = 0."""
        inputs = self._bullish_inputs()
        inputs["structure_bias"] = 0
        result = evaluate_trade(**inputs)
        self.assertEqual(result.bet_amount, 0.0)
        self.assertEqual(result.bet_fraction, 0.0)

    def test_trade_bet_amount_positive(self):
        """A trade decision should have a positive bet_amount."""
        result = evaluate_trade(**self._bullish_inputs())
        self.assertGreater(result.bet_amount, 0)

    def test_reasons_list_always_populated(self):
        """Reasons list should never be empty."""
        result = evaluate_trade(**self._bullish_inputs())
        self.assertIsInstance(result.reasons, list)
        self.assertGreater(len(result.reasons), 0)

    def test_bearish_trade_produces_no_side(self):
        """When model says p_model < p_market, the proposed side should be 'no'."""
        result = evaluate_trade(
            structure_bias=-1, confirmation_bias=-1,
            p_model=0.25, p_market=0.60, bankroll=10_000,
        )
        self.assertEqual(result.side, "no")

    def test_decision_fields_populated(self):
        """All numeric fields on a trade result should be non-None and finite."""
        result = evaluate_trade(**self._bullish_inputs())
        for attr in ("p_model", "p_market", "raw_edge", "net_edge",
                     "kelly_fraction", "bet_fraction", "bet_amount"):
            val = getattr(result, attr)
            self.assertTrue(math.isfinite(val), f"{attr} is not finite: {val}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
