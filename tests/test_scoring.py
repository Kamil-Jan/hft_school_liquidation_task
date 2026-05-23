"""Spec-critical: Score, PnL_all/kept/filtered and the turnover constraint."""
import numpy as np

from liqsignal.scoring import evaluate_filter, weighted_mean


def test_weighted_mean_drops_nan():
    pnl = np.array([1.0, np.nan, 3.0])
    w = np.array([1.0, 5.0, 1.0])
    assert np.isclose(weighted_mean(pnl, w), 2.0)  # (1+3)/2, nan row ignored


def test_evaluate_filter_hand_example():
    pnl = np.array([1.0, -1.0, 2.0, -2.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    f = np.array([0, 1, 0, 1])               # keep idx 0,2 ; filter idx 1,3
    r = evaluate_filter(pnl, w, f, n_days=1)

    assert np.isclose(r.pnl_all, 0.0)        # (1-1+2-2)/4
    assert np.isclose(r.pnl_kept, 1.5)       # (1+2)/2
    assert np.isclose(r.pnl_filtered, -1.5)  # (-1-2)/2
    assert np.isclose(r.score, 1.5)
    assert np.isclose(r.frac_filtered_w, 0.5)
    assert np.isclose(r.frac_filtered_n, 0.5)
    assert np.isclose(r.kept_turnover_per_day, 2.0)  # kept weight 2 / 1 day
    assert r.constraint_ok is False                   # 2 < 500_000


def test_turnover_scale_and_constraint():
    pnl = np.zeros(10)
    w = np.full(10, 1e5)
    f = np.zeros(10, dtype=int)              # keep everything
    # kept weight = 1e6; with scale 1 over 1 day -> 1e6/day >= 5e5 -> OK
    assert evaluate_filter(pnl, w, f, n_days=1, turnover_scale=1.0).constraint_ok
    # over 10 days -> 1e5/day < 5e5 -> violation
    assert not evaluate_filter(pnl, w, f, n_days=10, turnover_scale=1.0).constraint_ok


def test_nan_pnl_rows_excluded():
    pnl = np.array([1.0, np.nan, -1.0])
    w = np.array([1.0, 99.0, 1.0])
    f = np.array([0, 0, 0])
    r = evaluate_filter(pnl, w, f, n_days=1)
    assert r.n == 2 and np.isclose(r.pnl_all, 0.0)
