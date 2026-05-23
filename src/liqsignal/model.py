"""Combined predicted-markout model: one regressor per horizon.

A ``HistGradientBoostingRegressor`` predicts the maker markout ``pnl_i(tau)`` (bps)
from the engineered features, trained with **sample weights** ``w_i`` (the spec's
clipped notional) so the fit targets the same PnL the score optimises — per the
task's weighted-classification hint and paper 1's PnL-not-accuracy lesson. Models
are pooled across symbols (features are scale-free / venue-agnostic) and persisted
per tau so the submission can load them.

The predicted markout is a *signed* score: positive ⇒ expected-profitable maker
trade. The keep/filter decision (expected-value or Score-maximising cutoff) lives
in :mod:`liqsignal.analysis`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import feature_columns

MODEL_PATH = config.ARTIFACTS_DIR / "model_{tau}.joblib"


def _matrix(panel: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix as float64 (HistGBR handles NaNs natively)."""
    return panel.select(features).to_numpy().astype(np.float64)


def train_markout_model(panel: pl.DataFrame, tau: int, *, features: list[str] | None = None,
                        **hgbr_kwargs) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Fit a sample-weighted markout regressor for one horizon on ``panel``.

    Rows with non-finite markout (spec-excluded trades) are dropped. Returns the
    fitted model and the feature-column list it was trained on.
    """
    features = features or feature_columns(panel.columns)
    y = panel[f"pnl_{tau}"].to_numpy()
    w = panel["w"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(w)
    X = _matrix(panel, features)[mask]

    params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, random_state=0)
    params.update(hgbr_kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y[mask], sample_weight=w[mask])
    return model, features


def predict_markout(model: HistGradientBoostingRegressor, panel: pl.DataFrame,
                    features: list[str]) -> np.ndarray:
    """Predicted markout (bps) for every row of ``panel``."""
    return model.predict(_matrix(panel, features))


def predict_from_features(model: HistGradientBoostingRegressor, feats: dict[str, np.ndarray],
                          features: list[str]) -> np.ndarray:
    """Predicted markout from a feature dict (submission path)."""
    X = np.column_stack([feats[c].astype(np.float64) for c in features])
    return model.predict(X)


def save(model: HistGradientBoostingRegressor, features: list[str], tau: int,
         *, threshold: float | None = None) -> Path:
    """Persist a fitted model, its feature columns, and (optionally) the fitted
    Score-maximising keep/filter ``threshold`` so the submission applies the same
    operating point that the report measured."""
    config.ensure_artifacts()
    path = Path(str(MODEL_PATH).format(tau=tau))
    joblib.dump({"model": model, "features": features, "tau": tau,
                 "threshold": threshold}, path)
    return path


def load(tau: int):
    """Return ``(model, features)`` for a horizon, or ``(None, None)`` if absent."""
    path = Path(str(MODEL_PATH).format(tau=tau))
    if not path.exists():
        return None, None
    blob = joblib.load(path)
    return blob["model"], blob["features"]


def load_threshold(tau: int) -> float | None:
    """Return the persisted Score-maximising threshold for a horizon.

    ``None`` if no model is saved, or the model predates threshold persistence —
    callers then fall back to the expected-value cutoff.
    """
    path = Path(str(MODEL_PATH).format(tau=tau))
    if not path.exists():
        return None
    return joblib.load(path).get("threshold")
