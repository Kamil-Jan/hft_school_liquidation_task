"""Walk-forward (expanding-window) out-of-sample evaluation harness.

The shipped model is judged on a single validation month (March) and a single test
month (April) — too thin to tell a real regime-robust improvement from noise, and the
exact setup that let an earlier feature-selection pass overfit validation. This module
re-scores **any model spec** on several *consecutive held-out months* (Feb, Mar, Apr by
default) so a change is adopted only if it lifts the **mean out-of-sample Score**.

It operates on the existing sampled panels (`artifacts/panel_<sym>.parquet`): every row
already carries ``timestamp`` / ``w`` / ``pnl_{30,120,300}`` / the features, so a fold is
just a timestamp mask — no panel rebuild and no ``config`` edits.

The seam every experiment plugs into is a **fit function**::

    fit_fn(train_panel, tau, step) -> (predict_score, threshold)

where ``predict_score(panel) -> np.ndarray`` is a signed/keep-high score (predicted
markout, calibrated probability, predicted quantile, …) and ``threshold`` is the cutoff
below which a trade is filtered. Estimators stay isolated here (experiment-only) until a
winner is promoted into the production ``model`` module.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from . import analysis, config, model
from .scoring import evaluate_filter
from .splits import month_label, walk_forward_folds

# (predict_score, threshold) — predict_score keeps high scores, filters score < threshold.
PredictScore = Callable[[pl.DataFrame], np.ndarray]
FitFn = Callable[[pl.DataFrame, int, int], "tuple[PredictScore, float]"]


# ---------------------------------------------------------------------------
# Core walk-forward loop
# ---------------------------------------------------------------------------
def run_walk_forward(panel: pl.DataFrame, step: int, tau: int, fit_fn: FitFn, *,
                     folds: list[tuple[int, int, int, int]] | None = None
                     ) -> list[dict]:
    """Score one ``fit_fn`` on every walk-forward fold for one horizon.

    For each fold, fit on the train mask, predict the OOS month, apply the fold's
    threshold, and score with :func:`scoring.evaluate_filter` (turnover rescaled by the
    panel sampling ``step``). Returns one row of metrics per fold.
    """
    folds = folds or walk_forward_folds()
    rows: list[dict] = []
    for tr_start, tr_end, oos_start, oos_end in folds:
        train = panel.filter((pl.col("timestamp") >= tr_start) & (pl.col("timestamp") < tr_end))
        oos = panel.filter((pl.col("timestamp") >= oos_start) & (pl.col("timestamp") < oos_end))
        if train.height == 0 or oos.height == 0:
            continue
        predict_score, thr = fit_fn(train, tau, step)
        sc = np.asarray(predict_score(oos), dtype=np.float64)
        f = (sc < thr).astype(np.int8)
        res = evaluate_filter(oos[f"pnl_{tau}"].to_numpy(), oos["w"].to_numpy(), f,
                              n_days=max(1, oos["day"].n_unique()), turnover_scale=step)
        rows.append({
            "month": month_label(oos_start), "tau": tau, "threshold": float(thr),
            "score": res.score, "pnl_kept": res.pnl_kept, "pnl_all": res.pnl_all,
            "kept_turn_per_day": res.kept_turnover_per_day, "constraint_ok": res.constraint_ok,
            "frac_filt": res.frac_filtered_n, "n": res.n,
        })
    return rows


def evaluate_specs(panels: dict[str, pl.DataFrame], steps: dict[str, int],
                   specs: list[tuple[str, FitFn]], *, taus: tuple[int, ...] = config.TAUS,
                   folds: list[tuple[int, int, int, int]] | None = None) -> pl.DataFrame:
    """Run several named specs over all symbols/horizons; return a long per-fold frame.

    Columns: ``sym, spec, tau, month, score, pnl_kept, pnl_all, kept_turn_per_day,
    constraint_ok, frac_filt, n, threshold``.
    """
    out: list[dict] = []
    for sym, panel in panels.items():
        for tau in taus:
            for name, fit_fn in specs:
                for row in run_walk_forward(panel, steps[sym], tau, fit_fn, folds=folds):
                    out.append({"sym": sym, "spec": name, **row})
    return pl.DataFrame(out)


def summarize(long: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-fold results to mean/std/min OOS Score per (sym, spec, tau)."""
    return (long.group_by(["sym", "spec", "tau"])
            .agg(mean_score=pl.col("score").mean(),
                 std_score=pl.col("score").std(),
                 min_score=pl.col("score").min(),
                 n_folds=pl.len(),
                 all_ok=pl.col("constraint_ok").all())
            .sort(["sym", "tau", "spec"]))


# ---------------------------------------------------------------------------
# Experiment estimator factories (experiment-only; not the shipped model path)
# ---------------------------------------------------------------------------
def _fit_threshold(train: pl.DataFrame, tau: int, train_score: np.ndarray, step: int,
                   *, min_keep_frac: float = 0.05) -> float:
    """Purged-CV Score-maximising cutoff on the fold's train rows (reuses analysis)."""
    thr, _ = analysis.fit_score_threshold(
        train_score, train[f"pnl_{tau}"].to_numpy(), train["w"].to_numpy(),
        train["timestamp"].to_numpy(), step=step, min_keep_frac=min_keep_frac)
    return thr


def hgbr_fit_fn(*, features: list[str] | None = None, min_keep_frac: float = 0.05,
                **hgbr_kwargs) -> FitFn:
    """Baseline-style fit: sample-weighted HistGBR markout regressor + purged-CV threshold.

    ``hgbr_kwargs`` pass through to :func:`model.train_markout_model` (e.g.
    ``loss="absolute_error"``, ``monotonic_cst=...``), so most experiments are a thin
    wrapper over this.
    """
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        mdl, feats = model.train_markout_model(train, tau, features=features, **hgbr_kwargs)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


# ---- Phase 2: regime-robustness factories --------------------------------------------
def _recency_weight(train: pl.DataFrame, halflife_days: float) -> np.ndarray:
    """Notional weight decayed by trade age: ``w · exp(-age_days / halflife)``.

    Age is measured from the most recent train trade, so the fit leans on the regime
    closest to deployment without discarding older data.
    """
    ts = train["timestamp"].to_numpy().astype(np.float64)
    w = train["w"].to_numpy().astype(np.float64)
    age_days = (ts.max() - ts) / config.DAY_US
    return w * np.exp(-age_days / float(halflife_days))


def recency_hgbr_fit_fn(*, halflife_days: float, features: list[str] | None = None,
                        min_keep_frac: float = 0.05, **hgbr_kwargs) -> FitFn:
    """HistGBR markout regressor with recency-decayed sample weights."""
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        sw = _recency_weight(train, halflife_days)
        mdl, feats = model.train_markout_model(train, tau, features=features,
                                               sample_weight=sw, **hgbr_kwargs)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


# Sign-known features: by construction higher ⇒ more aligned with the taker side ⇒ the
# liquidation-reversion thesis predicts higher markout. Constrained monotone-increasing.
MONOTONIC_UP: tuple[str, ...] = (
    "binance_liqalign_5s", "binance_liqalign_30s", "binance_liqalign_300s",
    "bybit_liqalign_5s", "bybit_liqalign_30s", "bybit_liqalign_300s",
    "xexch_liqalign_30s", "xexch_liqalign_300s",
    "tfi_aligned_30s", "tfi_aligned_300s", "signed_vol_mom_30s", "signed_vol_mom_300s",
    "ret_1s_signed", "ret_5s_signed", "ret_30s_signed", "micro_signed_bps", "obi_signed",
)


def monotonic_hgbr_fit_fn(*, features: list[str] | None = None, up: tuple[str, ...] = MONOTONIC_UP,
                          min_keep_frac: float = 0.05, **hgbr_kwargs) -> FitFn:
    """HistGBR with +1 monotonic constraints on the sign-known (taker-aligned) features."""
    up_set = set(up)

    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        from .features import feature_columns
        feats = features or feature_columns(train.columns)
        cst = np.array([1 if f in up_set else 0 for f in feats], dtype=int)
        mdl, feats = model.train_markout_model(train, tau, features=feats,
                                               monotonic_cst=cst, **hgbr_kwargs)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


# ---- Phase 3: objective-reframing factories ------------------------------------------
def _design(panel: pl.DataFrame, tau: int, feats: list[str]):
    """(X, y_markout, w, finite-mask) for an estimator fit."""
    y = panel[f"pnl_{tau}"].to_numpy()
    w = panel["w"].to_numpy()
    X = panel.select(feats).to_numpy().astype(np.float64)
    return X, y, w, np.isfinite(y) & np.isfinite(w)


def clf_fit_fn(*, features: list[str] | None = None, min_keep_frac: float = 0.05,
               **clf_kwargs) -> FitFn:
    """Good/bad-trade classifier: target ``pnl_tau > 0``, score = P(good).

    Note: the score-max threshold is invariant to any monotone calibration (it only
    reranks-preservingly), so isotonic/Platt calibration would not change the harness
    Score — it matters only for a fixed-probability or expected-value cutoff (Phase 4).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        from .features import feature_columns
        feats = features or feature_columns(train.columns)
        X, y, w, mask = _design(train, tau, feats)
        params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                      min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                      validation_fraction=0.1, random_state=0)
        params.update(clf_kwargs)
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X[mask], (y[mask] > 0).astype(int), sample_weight=w[mask])

        def predict_score(panel: pl.DataFrame) -> np.ndarray:
            Xp = panel.select(feats).to_numpy().astype(np.float64)
            return clf.predict_proba(Xp)[:, 1]
        thr = _fit_threshold(train, tau, predict_score(train), step, min_keep_frac=min_keep_frac)
        return predict_score, thr
    return fit


def lgbm_quantile_fit_fn(*, alpha: float = 0.5, features: list[str] | None = None,
                         min_keep_frac: float = 0.05, **lgbm_kwargs) -> FitFn:
    """LightGBM quantile regressor: predicts the ``alpha`` quantile of markout (alpha>0.5
    ⇒ conservative, keeps only confidently-positive trades)."""
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        import lightgbm as lgb

        from .features import feature_columns
        feats = features or feature_columns(train.columns)
        X, y, w, mask = _design(train, tau, feats)
        params = dict(objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05,
                      num_leaves=31, min_child_samples=200, reg_lambda=1.0, n_jobs=-1, verbose=-1)
        params.update(lgbm_kwargs)
        reg = lgb.LGBMRegressor(**params)
        reg.fit(X[mask], y[mask], sample_weight=w[mask])

        def predict_score(panel: pl.DataFrame) -> np.ndarray:
            return reg.predict(panel.select(feats).to_numpy().astype(np.float64))
        thr = _fit_threshold(train, tau, predict_score(train), step, min_keep_frac=min_keep_frac)
        return predict_score, thr
    return fit


# ---------------------------------------------------------------------------
# Spec collections (named for the runner / report)
# ---------------------------------------------------------------------------
def baseline_specs() -> list[tuple[str, FitFn]]:
    """The current shipped model, reproduced for the harness (the comparison baseline)."""
    return [("baseline_hgbr_mse", hgbr_fit_fn())]


def regime_specs() -> list[tuple[str, FitFn]]:
    """Phase 2: recency weighting (halflife sweep) + monotonic constraints vs baseline."""
    return [
        ("baseline_hgbr_mse", hgbr_fit_fn()),
        ("recency_hl90", recency_hgbr_fit_fn(halflife_days=90)),
        ("recency_hl60", recency_hgbr_fit_fn(halflife_days=60)),
        ("recency_hl30", recency_hgbr_fit_fn(halflife_days=30)),
        ("monotonic_up", monotonic_hgbr_fit_fn()),
    ]


def objective_specs() -> list[tuple[str, FitFn]]:
    """Phase 3: robust regression / classification / quantile vs baseline."""
    return [
        ("baseline_hgbr_mse", hgbr_fit_fn()),
        ("hgbr_mae", hgbr_fit_fn(loss="absolute_error")),
        ("hgbr_clf", clf_fit_fn()),
        ("lgbm_quantile_a50", lgbm_quantile_fit_fn(alpha=0.5)),
        ("lgbm_quantile_a60", lgbm_quantile_fit_fn(alpha=0.6)),
    ]


def features_fit_fn(min_keep_frac: float = 0.05) -> FitFn:
    """MSE-HGBR using ``config.FEATURE_SETS[(sym, tau)]`` when present (else all features).

    Retained for reference only — ``features_specs`` now gates on the *shipped* estimator
    (``shipped_features_fit_fn``), so feature wins are measured on what we deploy, not on
    plain MSE-HGBR. The symbol is inferred from the train panel's price level (BTC ≫ ETH).
    """
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        from .features import feature_columns
        sym = "btc" if float(train["price"].median()) > 10_000 else "eth"
        feats = config.FEATURE_SETS.get((sym, tau)) or feature_columns(train.columns)
        mdl, feats = model.train_markout_model(train, tau, features=feats)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


def shipped_features_fit_fn(min_keep_frac: float = 0.05) -> FitFn:
    """Deployed per-(sym,τ) estimator (``model.fit_model`` on ``config.MODEL_SPECS``) trained
    on the curated ``config.FEATURE_SETS[(sym, tau)]`` (else all features).

    This is the honest feature-selection gate: it differs from ``shipped_fit_fn`` *only* in
    the feature set, so any OOS gain is attributable to feature curation, not the estimator.
    With ``FEATURE_SETS`` empty the two legs are identical (a useful sanity check).
    """
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        from .features import feature_columns
        sym = "btc" if float(train["price"].median()) > 10_000 else "eth"
        feats = config.FEATURE_SETS.get((sym, tau)) or feature_columns(train.columns)
        mdl, _kind, feats = model.fit_model(train, tau, sym, features=feats)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


def features_specs() -> list[tuple[str, FitFn]]:
    """Phase 1: shipped per-cell estimator on ALL features vs on the curated ``FEATURE_SETS``.
    Both legs use ``model.fit_model`` so the comparison isolates the feature set."""
    return [("shipped_all_features", shipped_fit_fn()),
            ("shipped_curated_features", shipped_features_fit_fn())]


def shipped_fit_fn(min_keep_frac: float = 0.05) -> FitFn:
    """The actually-deployed fit: dispatches via ``model.fit_model`` on the per-(sym,τ)
    ``config.MODEL_SPECS`` (symbol inferred from price, as ``signal()`` does). Use this to
    validate the shipped configuration end-to-end on OOS."""
    def fit(train: pl.DataFrame, tau: int, step: int) -> tuple[PredictScore, float]:
        sym = "btc" if float(train["price"].median()) > 10_000 else "eth"
        mdl, _kind, feats = model.fit_model(train, tau, sym)
        thr = _fit_threshold(train, tau, model.predict_markout(mdl, train, feats), step,
                             min_keep_frac=min_keep_frac)
        return (lambda panel: model.predict_markout(mdl, panel, feats)), thr
    return fit


def shipped_specs() -> list[tuple[str, FitFn]]:
    """Baseline vs the shipped per-cell estimators (config.MODEL_SPECS) — the deployed config."""
    return [("baseline_hgbr_mse", hgbr_fit_fn()), ("shipped", shipped_fit_fn())]


SPEC_SETS: dict[str, "Callable[[], list[tuple[str, FitFn]]]"] = {
    "baseline": baseline_specs,
    "regime": regime_specs,
    "objective": objective_specs,
    "features": features_specs,
    "shipped": shipped_specs,
}
