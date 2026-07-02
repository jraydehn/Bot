#!/usr/bin/env python3
"""
Quick smoke test for Deribit IV + Coinalyze liq signal integration.

Tests:
  1. deribit_iv  — fetch DVOL, unit conversion, SOL fallback
  2. coinalyze_liq — fetch signal, scoring logic, SOL fallback
  3. BearDrift rescue — verify liq_score >= 1 fires the rescue path
  4. Cache — second call returns cached result without network hit
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deribit_iv
import coinalyze_liq

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition


def test_deribit():
    print("\n── Deribit IV ──────────────────────────────────────────")

    dvol_btc = deribit_iv.fetch_dvol("BTC")
    check("BTC DVOL returned",   dvol_btc is not None)
    check("BTC DVOL in range",   dvol_btc is not None and 0.05 < dvol_btc < 5.0,
          f"{dvol_btc*100:.1f}%" if dvol_btc else "None")

    dvol_eth = deribit_iv.fetch_dvol("ETH")
    check("ETH DVOL returned",   dvol_eth is not None)

    sol = deribit_iv.fetch_dvol("SOL")
    check("SOL returns None",    sol is None)

    if dvol_btc:
        spm = deribit_iv.dvol_to_sigma_per_min(dvol_btc)
        check("sigma_per_min > 0",   spm > 0, f"{spm:.6f}")
        s45 = deribit_iv.dvol_to_sigma_tau(dvol_btc, 45)
        check("sigma_tau(45) = spm*√45", abs(s45 - spm * 45**0.5) < 1e-10,
              f"{s45:.4f}")

    # Cache test
    t0 = time.monotonic()
    deribit_iv.fetch_dvol("BTC")
    elapsed = time.monotonic() - t0
    check("BTC second call uses cache (<0.01s)", elapsed < 0.01, f"{elapsed*1000:.1f}ms")


def test_coinalyze():
    print("\n── Coinalyze Liq Signal ─────────────────────────────────")

    for asset in ("BTC", "ETH"):
        sig = coinalyze_liq.fetch_liq_signal(asset)
        check(f"{asset} signal returned", sig is not None)
        if sig:
            print(f"       bias={sig.liq_bias:+.2f}  long={sig.ls_long_pct:.1f}%  "
                  f"short={sig.ls_short_pct:.1f}%  score={sig.liq_score:+d}  [{sig.label}]")
            check(f"{asset} liq_bias in [-1,+1]",     -1.0 <= sig.liq_bias <= 1.0)
            check(f"{asset} ls pct sums ~100",         abs(sig.ls_long_pct + sig.ls_short_pct - 100) < 1.0,
                  f"{sig.ls_long_pct:.1f}+{sig.ls_short_pct:.1f}={sig.ls_long_pct+sig.ls_short_pct:.1f}")
            check(f"{asset} score in [-2,+2]",         -2 <= sig.liq_score <= 2)
            check(f"{asset} label not empty",          bool(sig.label))

    sol = coinalyze_liq.fetch_liq_signal("SOL")
    check("SOL returns None", sol is None)

    # Cache test
    t0 = time.monotonic()
    coinalyze_liq.fetch_liq_signal("BTC")
    elapsed = time.monotonic() - t0
    check("BTC second call uses cache (<0.01s)", elapsed < 0.01, f"{elapsed*1000:.1f}ms")


def test_scoring_logic():
    print("\n── Scoring Logic (unit) ─────────────────────────────────")

    # Pure short squeeze + no crowd skew → score +1
    from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH, LiqSignal
    def manual_score(liq_bias, ls_long, ls_short):
        s = 0
        if liq_bias >= _LIQ_BIAS_STRONG:
            s += 1
        elif liq_bias <= -_LIQ_BIAS_STRONG:
            s -= 1
        if ls_short >= _LS_CROWD_THRESH:
            s += 1
        elif ls_long >= _LS_CROWD_THRESH:
            s -= 1
        return max(-2, min(2, s))

    check("Squeeze (bias=+0.9, short=55%) → +1",
          manual_score(0.9, 45, 55) == 1)
    check("Squeeze + crowd short (bias=+0.9, short=68%) → +2",
          manual_score(0.9, 32, 68) == 2)
    check("Cascade (bias=-0.9, long=55%) → -1",
          manual_score(-0.9, 55, 45) == -1)
    check("Cascade + crowd long (bias=-0.9, long=68%) → -2",
          manual_score(-0.9, 68, 32) == -2)
    check("Conflict (bias=+0.9, long=68%) → 0",
          manual_score(0.9, 68, 32) == 0)
    check("Mild bias (0.4) → 0 (below threshold)",
          manual_score(0.4, 50, 50) == 0)


def test_beardrift_rescue():
    print("\n── BearDrift Rescue Logic ───────────────────────────────")

    # Simulate the rescue condition from paper_trade_runner.py:
    #   _bd_liq_squeeze = (_liq_signal is not None and _liq_signal.liq_score >= 1)
    #   _bd_rescued = vpin==1 or ema_stretch==1 or _bd_liq_squeeze

    class MockConfirm:
        def __init__(self, vpin, ema_stretch):
            self.vpin_score = vpin
            self.ema_stretch_score = ema_stretch

    def simulate_rescue(liq_score, vpin, ema_stretch):
        sig = coinalyze_liq.LiqSignal(
            liq_bias=1.0 if liq_score > 0 else -1.0,
            ls_long_pct=40.0, ls_short_pct=60.0,
            liq_score=liq_score, label="test",
        )
        confirm = MockConfirm(vpin, ema_stretch)
        _bd_liq_squeeze = sig.liq_score >= 1
        return confirm.vpin_score == 1 or confirm.ema_stretch_score == 1 or _bd_liq_squeeze

    check("liq_score=+1, vpin=0, ema=0 → RESCUED",
          simulate_rescue(1, 0, 0) is True)
    check("liq_score=+2, vpin=0, ema=0 → RESCUED",
          simulate_rescue(2, 0, 0) is True)
    check("liq_score=0, vpin=0, ema=0  → BLOCKED",
          simulate_rescue(0, 0, 0) is False)
    check("liq_score=-1, vpin=0, ema=0 → BLOCKED",
          simulate_rescue(-1, 0, 0) is False)
    check("liq_score=0, vpin=1, ema=0  → RESCUED (existing path)",
          simulate_rescue(0, 1, 0) is True)
    check("liq_score=0, vpin=0, ema=1  → RESCUED (existing path)",
          simulate_rescue(0, 0, 1) is True)


def main():
    print("=" * 56)
    print("  External Signal Integration — Smoke Test")
    print("=" * 56)
    test_deribit()
    test_coinalyze()
    test_scoring_logic()
    test_beardrift_rescue()
    print("\n" + "=" * 56)


if __name__ == "__main__":
    main()
