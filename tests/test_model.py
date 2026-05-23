"""Tests for model persistence — including the fitted Score-max threshold that
signal() applies by default."""
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from liqsignal import config, model


def test_threshold_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "model_{tau}.joblib")

    mdl = HistGradientBoostingRegressor()  # unfitted is fine; we only round-trip the blob
    model.save(mdl, ["f0", "f1"], 30, threshold=0.25)

    _, feats = model.load(30)
    assert feats == ["f0", "f1"]
    assert model.load_threshold(30) == 0.25


def test_load_threshold_absent_or_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "model_{tau}.joblib")

    # no file at all
    assert model.load_threshold(120) is None

    # legacy blob saved before threshold persistence (no "threshold" key)
    joblib.dump({"model": HistGradientBoostingRegressor(), "features": ["f0"], "tau": 300},
                tmp_path / "model_300.joblib")
    assert model.load_threshold(300) is None
