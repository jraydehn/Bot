"""Seed-ensemble shim for the BTC 15m refresh shadow model. 2026-08-02.

Lives in its own module so pickle can resolve the class at load time inside
paper_trade_runner_15m.py (the artifact dict's ["model"] is an instance of
this class; pickle auto-imports this module by name on load).

Averaging predict_proba across 5 fixed-seed fits directly addresses the
seed-noise problem documented in feedback_lgbm_single_fit_not_a_finding —
a single LGBM fit's run-to-run PnL std (~$2.8k on this book) rivals the
effects being sought; the mean of 5 is a lower-variance estimator of the
same model, with no selection step (no seed-picking = nothing fit to
recent data).
"""
import numpy as np


class SeedEnsemble:
    """predict_proba = mean of members' predict_proba. sklearn-duck-typed
    just enough for compute_p_model_15m's dict-artifact path."""

    def __init__(self, members):
        self.members = list(members)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)
