"""Scoring: the Score metric and turnover constraint exactly as specified.

For one horizon, given per-trade markout ``pnl``, weights ``w = min(notional, cap)``
and a binary filter ``f`` (1 = filter out, 0 = keep)::

    PnL_all      = sum_i w_i pnl_i               / sum_i w_i
    PnL_kept     = sum_i (1-f_i) w_i pnl_i        / sum_i (1-f_i) w_i
    PnL_filtered = sum_i  f_i    w_i pnl_i        / sum_i  f_i    w_i
    Score        = PnL_kept - PnL_all                       (maximise)

    KeptTurnoverPerDay = sum_i (1-f_i) w_i / n_days   >= 500_000   (constraint)

Trades with NaN markout (excluded by the spec's "beyond available BBO" rule) are
dropped before aggregating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TURNOVER_MIN_PER_DAY


@dataclass(frozen=True)
class ScoreResult:
    """All spec metrics for one (filter, horizon)."""
    pnl_all: float
    pnl_kept: float
    pnl_filtered: float
    score: float
    kept_turnover_per_day: float
    constraint_ok: bool
    frac_filtered_w: float   # fraction of weight filtered
    frac_filtered_n: float   # fraction of trades filtered
    n: int                   # number of (valid) trades scored

    def __str__(self) -> str:  # compact, for logs
        return (f"Score={self.score:+.3f}  PnL_kept={self.pnl_kept:+.3f}  "
                f"PnL_all={self.pnl_all:+.3f}  keptTurn/day={self.kept_turnover_per_day:,.0f}  "
                f"{'OK' if self.constraint_ok else 'VIOLATION'}")


def weighted_mean(pnl: np.ndarray, w: np.ndarray) -> float:
    """w-weighted mean of finite entries (= PnL_all)."""
    m = np.isfinite(pnl) & np.isfinite(w)
    return float((w[m] * pnl[m]).sum() / w[m].sum())


def evaluate_filter(pnl: np.ndarray, w: np.ndarray, f: np.ndarray, n_days: float,
                    *, turnover_scale: float = 1.0) -> ScoreResult:
    """Score a filter on aligned ``pnl`` / ``w`` / ``f`` arrays.

    ``turnover_scale`` multiplies the kept turnover when the inputs are a uniform
    *sample* of a larger population (pass ``full_n / sample_n``), so the per-day
    floor is comparable to the full data. Use 1.0 for full-population inputs.
    """
    valid = np.isfinite(pnl) & np.isfinite(w)
    pnl, w = pnl[valid], w[valid]
    f = f[valid].astype(float)
    keep = 1.0 - f

    sw = w.sum()
    sw_keep = float((w * keep).sum())
    sw_filt = float((w * f).sum())

    pnl_all = float((w * pnl).sum() / sw)
    pnl_kept = float((w * keep * pnl).sum() / sw_keep) if sw_keep > 0 else float("nan")
    pnl_filt = float((w * f * pnl).sum() / sw_filt) if sw_filt > 0 else float("nan")
    kept_turn = sw_keep * turnover_scale / n_days

    return ScoreResult(
        pnl_all=pnl_all,
        pnl_kept=pnl_kept,
        pnl_filtered=pnl_filt,
        score=pnl_kept - pnl_all,
        kept_turnover_per_day=kept_turn,
        constraint_ok=kept_turn >= TURNOVER_MIN_PER_DAY,
        frac_filtered_w=sw_filt / sw,
        frac_filtered_n=float(f.sum() / len(f)),
        n=int(len(f)),
    )
