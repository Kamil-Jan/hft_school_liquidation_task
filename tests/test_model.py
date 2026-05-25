"""Tests for per-(symbol, tau) model persistence — including the fitted Score-max
threshold that signal() applies by default."""
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from liqsignal import config, model


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


def test_load_threshold_absent_or_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)

    assert model.load_threshold(120, "btc") is None   # no file

    # legacy blob saved before threshold persistence (no "threshold" key)
    joblib.dump({"model": HistGradientBoostingRegressor(), "features": ["f0"], "tau": 300},
                tmp_path / "model_btc_300.joblib")
    assert model.load_threshold(300, "btc") is None
