"""Walk-forward harness: fold boundaries/embargo, and the run loop reproducing a
hand-computed Score on a tiny synthetic panel."""
import numpy as np
import polars as pl

from liqsignal import backtest, config
from liqsignal.splits import month_label, month_starts, walk_forward_folds


def test_month_starts_span_and_endpoint():
    start = config._utc_us(2025, 11, 1)
    end = config._utc_us(2026, 4, 29)
    bounds = month_starts(start, end)
    # Nov..Apr month starts, then the exact data end appended
    assert bounds[0] == start
    assert config._utc_us(2026, 4, 1) in bounds
    assert bounds[-1] == end
    assert bounds == sorted(bounds)


def test_walk_forward_folds_embargo_and_oos_months():
    folds = walk_forward_folds(n_oos=3)
    assert len(folds) == 3
    emb = config.SPLIT_EMBARGO_S * config.US
    months = [month_label(oos) for _, _, oos, _ in folds]
    assert months == ["2026-02", "2026-03", "2026-04"]
    for train_start, train_end, oos_start, oos_end in folds:
        assert train_start == config.TRAIN_START          # expanding window anchored at data start
        assert train_end == oos_start - emb               # embargo purged before the OOS month
        assert oos_start < oos_end


def test_run_walk_forward_matches_manual_score():
    # Build a 2-month synthetic panel: score perfectly ranks pnl, so filtering the
    # negative-score trades must lift kept above all (a positive Score). Timestamps are
    # spread across each month so the purged-CV threshold has non-degenerate time folds.
    n = 4000
    rng = np.random.default_rng(0)
    jan_ts = np.linspace(config._utc_us(2026, 1, 1), config._utc_us(2026, 1, 31), n)
    feb_ts = np.linspace(config._utc_us(2026, 2, 1), config._utc_us(2026, 2, 28), n)
    ts = np.concatenate([jan_ts, feb_ts]).astype(np.int64)
    pnl = rng.normal(0, 1.0, 2 * n)
    panel = pl.DataFrame({
        "timestamp": ts,
        "pnl_30": pnl,
        "w": np.full(2 * n, 50_000.0),            # realistic notional so the turnover floor clears
        "day": (ts // config.DAY_US).astype(np.int64),
        # one feature equal to pnl so the regressor's score ranks trades correctly
        "feat": pnl + rng.normal(0, 0.05, 2 * n),
    })
    folds = [(config._utc_us(2026, 1, 1), config._utc_us(2026, 2, 1) - config.SPLIT_EMBARGO_S * config.US,
              config._utc_us(2026, 2, 1), config._utc_us(2026, 3, 1))]
    rows = backtest.run_walk_forward(panel, step=300, tau=30,
                                     fit_fn=backtest.hgbr_fit_fn(features=["feat"], min_keep_frac=0.05),
                                     folds=folds)
    assert len(rows) == 1
    r = rows[0]
    assert r["month"] == "2026-02"
    assert r["score"] > 0.0                       # informative score ⇒ positive OOS Score
    assert 0.0 < r["frac_filt"] < 1.0


def test_summarize_aggregates_mean_std():
    long = pl.DataFrame({
        "sym": ["btc", "btc"], "spec": ["a", "a"], "tau": [30, 30],
        "month": ["2026-02", "2026-03"], "score": [1.0, 3.0], "constraint_ok": [True, True],
    })
    s = backtest.summarize(long)
    assert s.height == 1
    assert abs(s["mean_score"][0] - 2.0) < 1e-9
    assert s["n_folds"][0] == 2
    assert bool(s["all_ok"][0])


def _btc_panel(n: int = 4000):
    """Two-month BTC-priced synthetic panel; ``feat`` ranks pnl so the model is informative.
    Price ≫ 10k so the fit_fns infer sym='btc' (mirroring ``signal()``)."""
    rng = np.random.default_rng(0)
    jan_ts = np.linspace(config._utc_us(2026, 1, 1), config._utc_us(2026, 1, 31), n)
    feb_ts = np.linspace(config._utc_us(2026, 2, 1), config._utc_us(2026, 2, 28), n)
    ts = np.concatenate([jan_ts, feb_ts]).astype(np.int64)
    pnl = rng.normal(0, 1.0, 2 * n)
    return pl.DataFrame({
        "timestamp": ts, "pnl_30": pnl, "w": np.full(2 * n, 50_000.0),
        "price": np.full(2 * n, 50_000.0), "day": (ts // config.DAY_US).astype(np.int64),
        "feat": pnl + rng.normal(0, 0.05, 2 * n),
        "noise": rng.normal(0, 1.0, 2 * n),
    })


_FOLD = [(config._utc_us(2026, 1, 1), config._utc_us(2026, 2, 1) - config.SPLIT_EMBARGO_S * config.US,
          config._utc_us(2026, 2, 1), config._utc_us(2026, 3, 1))]


def test_features_specs_compares_shipped_legs():
    # Lock the feature-selection gate to shipped-vs-shipped (guards against a regression to
    # the old MSE-vs-MSE comparison, which would judge features on an estimator we don't ship).
    assert [name for name, _ in backtest.features_specs()] == \
        ["shipped_all_features", "shipped_curated_features"]


def test_shipped_features_fit_fn_uses_feature_set(monkeypatch):
    monkeypatch.setitem(config.FEATURE_SETS, ("btc", 30), ["feat"])
    rows = backtest.run_walk_forward(_btc_panel(), step=300, tau=30,
                                     fit_fn=backtest.shipped_features_fit_fn(), folds=_FOLD)
    assert len(rows) == 1 and rows[0]["month"] == "2026-02"
    assert 0.0 <= rows[0]["frac_filt"] < 1.0          # the curated single feature drives the filter


def test_shipped_features_falls_back_to_all_when_empty(monkeypatch):
    # With FEATURE_SETS empty the curated leg must be identical to the plain shipped leg.
    monkeypatch.setattr(config, "FEATURE_SETS", {})
    panel = _btc_panel()
    all_fit = backtest.shipped_fit_fn()(panel, 30, 300)
    cur_fit = backtest.shipped_features_fit_fn()(panel, 30, 300)
    assert abs(all_fit[1] - cur_fit[1]) < 1e-12       # same threshold
    np.testing.assert_allclose(all_fit[0](panel), cur_fit[0](panel))   # same predictions
