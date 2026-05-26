"""Tests for per-(symbol, tau) model persistence — including the fitted Score-max
threshold that signal() applies by default."""
import numpy as np
import joblib
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from liqsignal import config, model


def _toy_panel(tau: int, n: int = 3000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 0.2, n)
    ts = np.linspace(0, 60 * config.DAY_US, n).astype(np.int64)
    return pl.DataFrame({"timestamp": ts, "w": np.full(n, 1000.0), f"pnl_{tau}": y, "feat": x})


def test_threshold_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)

    mdl = HistGradientBoostingRegressor()  # unfitted is fine; we only round-trip the blob
    model.save(mdl, ["f0", "f1"], 30, "btc", threshold=0.25)

    _, feats = model.load(30, "btc")
    assert feats == ["f0", "f1"]
    assert model.load_threshold(30, "btc") == 0.25


def test_per_symbol_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    model.save(HistGradientBoostingRegressor(), ["a"], 120, "btc", threshold=1.0)
    model.save(HistGradientBoostingRegressor(), ["a", "b"], 120, "eth", threshold=2.0)

    assert model.model_path(120, "btc").name == "model_btc_120.joblib"
    assert model.load_threshold(120, "btc") == 1.0
    assert model.load_threshold(120, "eth") == 2.0
    assert model.load(120, "eth")[1] == ["a", "b"]
    assert model.load(30, "btc") == (None, None)    # not trained → absent


def test_sample_weight_override_changes_fit():
    # A regressor that ignores w (default) vs one driven entirely toward the high-x
    # region by a sample_weight concentrated there should predict differently at x=1.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    y = np.where(x > 0.5, 5.0, -5.0) + rng.normal(0, 0.1, n)
    panel = pl.DataFrame({"pnl_30": y, "w": np.ones(n), "x": x})

    base, _ = model.train_markout_model(panel, 30, features=["x"])
    sw = np.where(x > 0.5, 100.0, 1.0)          # weight the high-x region heavily
    weighted, _ = model.train_markout_model(panel, 30, features=["x"], sample_weight=sw)

    grid = pl.DataFrame({"x": np.array([0.1, 0.9])})
    assert not np.allclose(model.predict_markout(base, grid, ["x"]),
                           model.predict_markout(weighted, grid, ["x"]))


def test_fit_model_dispatches_hgbr_variants(monkeypatch):
    # MAE-loss and recency-weighted cells both resolve to a HistGBR estimator.
    monkeypatch.setitem(config.MODEL_SPECS, ("btc", 30), {"kind": "hgbr", "loss": "absolute_error"})
    monkeypatch.setitem(config.MODEL_SPECS, ("btc", 300), {"kind": "hgbr", "recency_halflife_days": 30})
    for tau in (30, 300):
        mdl, kind, feats = model.fit_model(_toy_panel(tau), tau, "btc", features=["feat"])
        assert kind == "hgbr" and feats == ["feat"]
        assert isinstance(mdl, HistGradientBoostingRegressor)


def test_recency_weight_decays_with_age():
    p = _toy_panel(30)
    w = model.recency_weight(p, halflife_days=30)
    # most recent trade keeps ~full weight; oldest is decayed below it
    assert w[-1] > w[0]
    assert np.isclose(w[-1], p["w"][-1], rtol=1e-3)


def test_fit_model_lgbm_quantile_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setitem(config.MODEL_SPECS, ("eth", 120), {"kind": "lgbm_quantile", "alpha": 0.6})
    p = _toy_panel(120)

    mdl, kind, feats = model.fit_model(p, 120, "eth", features=["feat"])
    assert kind == "lgbm_quantile"

    model.save(mdl, feats, 120, "eth", threshold=0.1, kind=kind)
    m2, f2 = model.load(120, "eth")
    assert f2 == ["feat"]
    # submission path (feature dict) matches the panel path, and both run on the lgbm model
    feat_dict = {"feat": p["feat"].to_numpy()}
    np.testing.assert_allclose(model.predict_from_features(m2, feat_dict, f2),
                               model.predict_markout(m2, p, f2))


def test_load_threshold_absent_or_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)

    assert model.load_threshold(120, "btc") is None   # no file

    # legacy blob saved before threshold persistence (no "threshold" key)
    joblib.dump({"model": HistGradientBoostingRegressor(), "features": ["f0"], "tau": 300},
                tmp_path / "model_btc_300.joblib")
    assert model.load_threshold(300, "btc") is None
