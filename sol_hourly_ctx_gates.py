"""Transferable SOL hourly context gates for the v7/v8 challenger books.
2026-07-30.

Faithful ports of the production decision-layer gates that do NOT depend on
the production model's probability (those — Gate 0 saturation, Gate 3
min-edge, Gate OTM — are replaced by each challenger's own edge rule).
Thresholds copied from decision.py / paper_trade_runner.py as of 2026-07-30:

  CS   OTM YES (offset_pct>0) needs composite_p_up >= 0.55, UNLESS the SOL
       rescue holds (offset_pct <= 0.62% and composite_trend <= 1).
  NS   OTM NO (offset_pct<0) needs composite_p_up <= 0.45 (SOL threshold).
  RR   YES: pm/(1-pm) > 3.0 (pm>0.75) blocked unconditionally.
       NO: (1-pm)/pm < 0.33 or > 4.0 blocked unless edge >= 0.08 (the
       exception uses the CHALLENGER's own fee-adj edge — faithful transfer).
  LS   contrarian side (YES at pm<0.5 / NO at pm>0.5) blocked unless
       ls_long_pct >= 71.65 (sol_contrarian_ls_gate).

Gate CI is NOT ported — removed for SOL 2026-07-16 (project memory).

Challenger runners tag every booked trade with the failed-gate list in a
`ctx_gates` column but still book the trade — the gated book is a filtered
VIEW (dashboard/analysis), so no data is lost either way.
"""
import numpy as np


def ctx_gate_fails(side: str, p_market: float, offset_pct: float,
                   composite_p_up: float, composite_trend: float,
                   ls_long_pct: float, edge: float) -> str:
    """Return '+'-joined failed context gates ('' = all pass)."""
    fails = []
    pm = float(p_market)
    off = float(offset_pct) if offset_pct == offset_pct else np.nan
    cpu = float(composite_p_up) if composite_p_up == composite_p_up else np.nan
    ctr = float(composite_trend) if composite_trend == composite_trend else np.nan
    ls = float(ls_long_pct) if ls_long_pct == ls_long_pct else np.nan

    if side == "yes":
        if off == off and off > 0 and cpu == cpu and cpu < 0.55:
            rescued = (off <= 0.0062) and (ctr == ctr and ctr <= 1)
            if not rescued:
                fails.append("CS")
        if pm > 0.75:
            fails.append("RR")
        if pm < 0.5 and not (ls == ls and ls >= 71.65):
            fails.append("LS")
    else:  # no
        if off == off and off < 0 and cpu == cpu and cpu > 0.45:
            fails.append("NS")
        rr = (1 - pm) / pm
        if (rr < 0.33 or rr > 4.0) and edge < 0.08:
            fails.append("RR")
        if pm > 0.5 and not (ls == ls and ls >= 71.65):
            fails.append("LS")
    return "+".join(fails)


def ctx_gate_fails_btc(side: str, p_market: float, offset_pct: float,
                       composite_p_up: float, edge: float,
                       rr_exception: float = 0.055) -> str:
    """BTC/ETH variant of the transferable context gates (decision.py as of
    07-31): CS OTM YES needs composite_p_up>=0.55 (no SOL rescue); CI ITM
    YES blocked if composite_p_up<0.45 (live for BTC/ETH — only SOL removed
    it); NS OTM NO needs composite_p_up<=0.50; R:R YES pm>0.75 blocked, NO
    out-of-bounds unless edge>=rr_exception (BTC NO 0.055; ETH/others
    0.08). No contrarian-LS gate (SOL-only)."""
    fails = []
    pm = float(p_market)
    off = float(offset_pct) if offset_pct == offset_pct else np.nan
    cpu = float(composite_p_up) if composite_p_up == composite_p_up else np.nan

    if side == "yes":
        if off == off and off > 0 and cpu == cpu and cpu < 0.55:
            fails.append("CS")
        if off == off and off < 0 and cpu == cpu and cpu < 0.45:
            fails.append("CI")
        if pm > 0.75:
            fails.append("RR")
    else:
        if off == off and off < 0 and cpu == cpu and cpu > 0.50:
            fails.append("NS")
        rr = (1 - pm) / pm
        if (rr < 0.33 or rr > 4.0) and edge < rr_exception:
            fails.append("RR")
    return "+".join(fails)
