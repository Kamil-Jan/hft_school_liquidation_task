"""Predicted-markout model: one regressor per (symbol, horizon).

A ``HistGradientBoostingRegressor`` predicts the maker markout ``pnl_i(tau)`` (bps)
from the engineered features, trained with **sample weights** ``w_i`` (the spec's
clipped notional) so the fit targets the same PnL the score optimises — per the
task's weighted-classification hint and paper 1's PnL-not-accuracy lesson. A model
is fit per ``(symbol, tau)`` and persisted as ``model_<sym>_<tau>.joblib`` so the
submission can load the one matching the symbol it is scoring.

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


def _matrix(panel: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix as float64 (HistGBR handles NaNs natively)."""
    return panel.select(features).to_numpy().astype(np.float64)


def train_markout_model(panel: pl.DataFrame, tau: int, *, features: list[str] | None = None,
                        sample_weight: np.ndarray | None = None,
                        **hgbr_kwargs) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Fit a sample-weighted markout regressor for one horizon on ``panel``.

    Rows with non-finite markout (spec-excluded trades) are dropped. Weights default to
    the spec's clipped notional ``w``; pass ``sample_weight`` (aligned to ``panel`` rows)
    to override — e.g. a recency-decayed weight for regime robustness. ``hgbr_kwargs``
    pass straight to the estimator (e.g. ``loss="absolute_error"``, ``monotonic_cst=...``).
    Returns the fitted model and the feature-column list it was trained on.
    """
    features = features or feature_columns(panel.columns)
    y = panel[f"pnl_{tau}"].to_numpy()
    w = panel["w"].to_numpy() if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(w)
    X = _matrix(panel, features)[mask]

    params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, random_state=0)
    params.update(hgbr_kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y[mask], sample_weight=w[mask])
    return model, features


def recency_weight(panel: pl.DataFrame, halflife_days: float) -> np.ndarray:
    """Notional weight decayed by trade age: ``w · exp(-age_days / halflife)``.

    Age is measured from the most recent trade in ``panel``, so the fit leans on the
    regime closest to the end of the train window without discarding older data.
    """
    ts = panel["timestamp"].to_numpy().astype(np.float64)
    w = panel["w"].to_numpy().astype(np.float64)
    age_days = (ts.max() - ts) / config.DAY_US
    return w * np.exp(-age_days / float(halflife_days))


def fit_lgbm_quantile(panel: pl.DataFrame, tau: int, alpha: float, *,
                      features: list[str] | None = None, **lgbm_kwargs):
    """Fit a sample-weighted LightGBM quantile regressor (predicts the ``alpha`` quantile
    of markout). ``alpha`` > 0.5 ⇒ conservative — it only ranks a trade high when even its
    lower-quantile markout is good, which makes it robust to regime sign-flips. Returns
    ``(model, features)``; the model exposes ``.predict`` so the submission path is uniform."""
    import lightgbm as lgb  # dev/experiment dependency; only needed for the quantile cells

    features = features or feature_columns(panel.columns)
    y = panel[f"pnl_{tau}"].to_numpy()
    w = panel["w"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(w)
    X = _matrix(panel, features)[mask]
    params = dict(objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05,
                  num_leaves=31, min_child_samples=200, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    params.update(lgbm_kwargs)
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(X, y[mask], sample_weight=w[mask])
    return mdl, features


def fit_model(panel: pl.DataFrame, tau: int, symbol: str, *, features: list[str] | None = None):
    """Fit the estimator chosen for ``(symbol, tau)`` by the walk-forward study.

    Dispatches on ``config.MODEL_SPECS`` (falls back to ``DEFAULT_MODEL_SPEC`` = HGBR-MSE):
    HistGBR with an optional ``loss`` (e.g. ``absolute_error``) and/or ``recency_halflife_days``
    sample weighting, or LightGBM quantile (``alpha``). Returns ``(model, kind, features)``.
    """
    spec = config.MODEL_SPECS.get((symbol, tau), config.DEFAULT_MODEL_SPEC)
    kind = spec["kind"]
    if kind == "hgbr":
        kwargs = {k: spec[k] for k in ("loss",) if k in spec}
        sw = (recency_weight(panel, spec["recency_halflife_days"])
              if "recency_halflife_days" in spec else None)
        mdl, feats = train_markout_model(panel, tau, features=features, sample_weight=sw, **kwargs)
        return mdl, kind, feats
    if kind == "lgbm_quantile":
        mdl, feats = fit_lgbm_quantile(panel, tau, spec["alpha"], features=features)
        return mdl, kind, feats
    raise ValueError(f"unknown model spec kind {kind!r} for ({symbol}, {tau})")


def predict_markout(model, panel: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Predicted score (markout bps, or a markout quantile for LightGBM) for every row of
    ``panel``. Higher ⇒ keep. Uniform across estimator kinds — they all expose ``.predict``."""
    return model.predict(_matrix(panel, features))


def predict_from_features(model, feats: dict[str, np.ndarray],
                          features: list[str]) -> np.ndarray:
    """Predicted score from a feature dict (submission path). ``model.predict`` is uniform
    across HistGBR and LightGBM, so this path is estimator-agnostic."""
    X = np.column_stack([feats[c].astype(np.float64) for c in features])
    return model.predict(X)


def model_path(tau: int, symbol: str) -> Path:
    """Artifact path for the per-symbol model ``model_<sym>_<tau>.joblib``."""
    return config.ARTIFACTS_DIR / f"model_{symbol}_{tau}.joblib"


def save(model, features: list[str], tau: int, symbol: str,
         *, threshold: float | None = None, kind: str = "hgbr") -> Path:
    """Persist a fitted per-(symbol, tau) model, its feature columns, the estimator
    ``kind`` (for transparency; inference is uniform), and (optionally) the fitted
    Score-maximising keep/filter ``threshold`` so the submission applies the same
    operating point that the report measured.

    Note: a ``kind="lgbm_quantile"`` blob unpickles only where ``lightgbm`` imports — the
    grader environment must have it (see scripts/patch_lightgbm.py / README)."""
    config.ensure_artifacts()
    path = model_path(tau, symbol)
    joblib.dump({"model": model, "features": features, "tau": tau,
                 "symbol": symbol, "threshold": threshold, "kind": kind}, path)
    return path


def load(tau: int, symbol: str):
    """Return ``(model, features)`` for a (symbol, horizon), or ``(None, None)`` if absent."""
    path = model_path(tau, symbol)
    if not path.exists():
        return None, None
    blob = joblib.load(path)
    return blob["model"], blob["features"]


def load_threshold(tau: int, symbol: str) -> float | None:
    """Return the persisted Score-maximising threshold for a (symbol, horizon).

    ``None`` if no model is saved, or the model predates threshold persistence —
    callers then fall back to the expected-value cutoff.
    """
    path = model_path(tau, symbol)
    if not path.exists():
        return None
    return joblib.load(path).get("threshold")
