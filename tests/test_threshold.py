"""Unit tests for the score-thresholding helpers."""
import numpy as np

from liqsignal import config
from liqsignal.analysis import (apply_threshold, expected_value_threshold,
                                fit_score_threshold)


def test_expected_value_threshold():
    score = np.array([-1.0, 0.5, -0.2, 2.0])
    np.testing.assert_array_equal(expected_value_threshold(score), [1, 0, 1, 0])
    # with a positive cost, the +0.5 trade is also filtered
    np.testing.assert_array_equal(expected_value_threshold(score, cost=0.6), [1, 1, 1, 0])


def test_apply_threshold():
    score = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(apply_threshold(score, 1.5), [1, 1, 0, 0])


def test_fit_score_threshold_separable():
    # score nearly equals pnl: a positive cutoff should keep the winners and drop losers.
    rng = np.random.default_rng(0)
    n = 20_000
    pnl = rng.normal(0, 1, n)
    score = pnl + rng.normal(0, 0.1, n)         # informative score
    w = np.ones(n)
    # monotonic timestamps spanning ~50 days so the purged time folds are well-separated
    ts = np.linspace(0, 50 * config.DAY_US, n).astype(np.int64)
    thr, cv = fit_score_threshold(score, pnl, w, ts, step=1.0, turnover_floor=0.0,
                                  n_splits=4, min_keep_frac=0.05)
    assert thr > 0.0                             # filters predicted-losers
    # applying the threshold lifts the kept mean above the all mean
    f = apply_threshold(score, thr)
    kept = pnl[f == 0]
    assert kept.mean() > pnl.mean()
    assert cv > 0.0
