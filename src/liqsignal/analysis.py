"""Reusable analysis helpers for the conditional-markout studies.

Kept separate from plotting/printing so both scripts and notebooks can call them.
The single-feature filter is fit on *train* (direction and keep-threshold) and the
same rule is applied to *validation* — no per-split leakage.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config
from .scoring import ScoreResult, evaluate_filter


def load_panel(symbol: str) -> tuple[pl.DataFrame, int]:
    """Load a feature panel and its sampling step from artifacts."""
    panel = pl.read_parquet(config.ARTIFACTS_DIR / f"panel_{symbol}.parquet")
    step = int(pl.read_parquet(config.ARTIFACTS_DIR / f"panel_meta_{symbol}.parquet")["step"][0])
    return panel, step


def weighted_markout(df: pl.DataFrame, pnl_col: str, w_col: str = "w") -> float:
    """w-weighted mean markout over finite rows (= PnL_all on this subset)."""
    d = df.filter(pl.col(pnl_col).is_finite() & pl.col(w_col).is_finite())
    if d.height == 0:
        return float("nan")
    return float((d[pnl_col] * d[w_col]).sum() / d[w_col].sum())


def conditional_markout(df: pl.DataFrame, feature: str, pnl_col: str,
                        n_quantiles: int = 5) -> pl.DataFrame:
    """w-weighted markout per quantile bucket of ``feature``."""
    d = df.filter(pl.col(feature).is_finite() & pl.col(pnl_col).is_finite())
    labels = [str(i) for i in range(n_quantiles)]
    d = d.with_columns(pl.col(feature).qcut(n_quantiles, labels=labels,
                                            allow_duplicates=True).alias("bucket"))
    return (d.group_by("bucket")
            .agg(lo=pl.col(feature).min(), hi=pl.col(feature).max(), n=pl.len(),
                 wpnl=(pl.col(pnl_col) * pl.col("w")).sum() / pl.col("w").sum())
            .sort("bucket"))


def _weighted_cov_sign(feat: np.ndarray, pnl: np.ndarray, w: np.ndarray) -> float:
    fm = (feat * w).sum() / w.sum()
    pm = (pnl * w).sum() / w.sum()
    cov = (w * (feat - fm) * (pnl - pm)).sum()
    return 1.0 if cov >= 0 else -1.0


def fit_keep_best(train: pl.DataFrame, feature: str, pnl_col: str,
                  keep_frac: float) -> tuple[float, float]:
    """Fit a one-feature 'keep the best-ranked trades' rule on the train split.

    Returns ``(direction, threshold)``: keep rows whose ``direction*feature`` is in
    the top ``keep_frac`` (i.e. filter those below ``threshold``).
    """
    d = train.filter(pl.col(feature).is_finite() & pl.col(pnl_col).is_finite())
    feat, pnl, w = d[feature].to_numpy(), d[pnl_col].to_numpy(), d["w"].to_numpy()
    direction = _weighted_cov_sign(feat, pnl, w)
    threshold = float(np.quantile(direction * feat, 1.0 - keep_frac))
    return direction, threshold


def apply_keep_best(df: pl.DataFrame, feature: str, direction: float,
                    threshold: float) -> np.ndarray:
    """0/1 filter (1 = filter out) for a fitted keep-best rule."""
    rank = direction * df[feature].to_numpy()
    return (rank < threshold).astype(np.int8)


def score_split(df: pl.DataFrame, pnl_col: str, f: np.ndarray, step: int) -> ScoreResult:
    """Score a filter on a panel subset, rescaling turnover by the sampling step."""
    return evaluate_filter(df[pnl_col].to_numpy(), df["w"].to_numpy(), f,
                           n_days=df["day"].n_unique(), turnover_scale=step)


# ---------------------------------------------------------------------------
# Thresholding a predicted-markout score (replaces the keep-N% rule)
# ---------------------------------------------------------------------------
# A score is a predicted markout (bps); we KEEP high-score trades and FILTER low.
# f_i = 1 (filter) where score_i < threshold.

def expected_value_threshold(score: np.ndarray, cost: float = 0.0) -> np.ndarray:
    """Parameter-free rule: filter trades whose predicted markout is below ``cost``
    (default 0 ⇒ drop expected losers). Returns a 0/1 array (1 = filter)."""
    return (np.asarray(score) < cost).astype(np.int8)


def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    """0/1 filter (1 = filter out) for a score cutoff: filter where score < threshold."""
    return (np.asarray(score) < threshold).astype(np.int8)


def fit_score_threshold(score: np.ndarray, pnl: np.ndarray, w: np.ndarray, ts: np.ndarray,
                        *, step: float = 1.0, turnover_floor: float = config.TURNOVER_MIN_PER_DAY,
                        n_grid: int = 60, n_splits: int = 5, embargo_s: int | None = None,
                        min_keep_frac: float = 0.0) -> tuple[float, float]:
    """Choose the score cutoff that maximises Score = PnL_kept - PnL_all, via
    **purged + embargoed time-series CV**.

    Candidate cutoffs are quantiles of ``score``. Rows are sorted by ``ts`` (epoch
    µs) and split into ``n_splits`` *contiguous time blocks*; each candidate is
    scored on every held-out block with an ``embargo_s``-second margin purged at the
    internal boundaries, so a trade whose markout window (≤ max τ) reaches into the
    neighbouring block can't inflate the estimate. The cutoff with the best mean
    held-out Score (subject to the turnover floor and an optional ``min_keep_frac``
    guard on every block) is returned as ``(threshold, cv_score)``.

    Random k-fold would mix adjacent, overlapping-markout trades across folds and
    badly overstate the Score; contiguous purged folds give an honest estimate.
    Fit on train; apply the returned threshold to validation/test via
    :func:`apply_threshold`.
    """
    score = np.asarray(score, dtype=np.float64)
    pnl = np.asarray(pnl, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ts = np.asarray(ts)
    valid = np.isfinite(score) & np.isfinite(pnl) & np.isfinite(w)
    score, pnl, w, ts = score[valid], pnl[valid], w[valid], ts[valid]
    if len(score) == 0:
        return 0.0, float("nan")

    embargo_us = (max(config.TAUS) if embargo_s is None else embargo_s) * config.US
    order = np.argsort(ts, kind="stable")
    score, pnl, w, ts = score[order], pnl[order], w[order], ts[order]
    bounds = np.linspace(0, len(score), n_splits + 1).astype(int)

    candidates = np.unique(np.quantile(score, np.linspace(0.0, 0.99, n_grid)))
    best_thr, best_cv = float(candidates[0]), -np.inf
    for thr in candidates:
        fold_scores = []
        ok = True
        for k in range(n_splits):
            a, b = int(bounds[k]), int(bounds[k + 1])
            if b <= a:
                continue
            tt = ts[a:b]
            keep = np.ones(b - a, dtype=bool)
            if k > 0:                       # purge rows abutting the previous block
                keep &= (tt - tt[0]) >= embargo_us
            if k < n_splits - 1:            # purge rows whose markout reaches into the next block
                keep &= (tt[-1] - tt) >= embargo_us
            if not keep.any():
                continue
            sc, pn, ww, tk = score[a:b][keep], pnl[a:b][keep], w[a:b][keep], tt[keep]
            f = apply_threshold(sc, thr)
            n_days = max(1, len(np.unique(tk // config.DAY_US)))
            res = evaluate_filter(pn, ww, f, n_days=n_days, turnover_scale=step)
            if res.kept_turnover_per_day < turnover_floor or (1.0 - res.frac_filtered_n) < min_keep_frac:
                ok = False
                break
            fold_scores.append(res.score)
        if ok and fold_scores:
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_cv:
                best_cv, best_thr = mean_score, float(thr)
    return best_thr, best_cv
