"""Train / validation / test / out-of-range assignment from timestamps.

All boundaries come from :mod:`liqsignal.config` (`TRAIN_START`, `VAL_START`,
`TEST_START`, `SPLIT_END`, `USE_TEST`, `SPLIT_EMBARGO_S`). :func:`split_windows`
is the single source of truth; the NumPy function and the Polars expression are both
built from it, so the two can never disagree on the boundaries or the toggle.

Leak safety: ``SPLIT_EMBARGO_S`` seconds are dropped at the trailing edge of every
window (those rows become ``"other"``), so a trade whose markout horizon (≤ max τ)
would reach into the next split is excluded rather than contaminating it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config

TRAIN = "train"
VAL = "val"
TEST = "test"
OTHER = "other"


def split_windows() -> list[tuple[str, int, int]]:
    """Ordered ``(label, start, end)`` windows (right edge exclusive) per config.

    Three windows when ``config.USE_TEST`` is set, otherwise two (the test window
    folds into validation). Windows are disjoint and contiguous.
    """
    if config.USE_TEST:
        return [
            (TRAIN, config.TRAIN_START, config.VAL_START),
            (VAL, config.VAL_START, config.TEST_START),
            (TEST, config.TEST_START, config.SPLIT_END),
        ]
    return [
        (TRAIN, config.TRAIN_START, config.VAL_START),
        (VAL, config.VAL_START, config.SPLIT_END),
    ]


def assign_split(ts_us: np.ndarray) -> np.ndarray:
    """Map epoch-microsecond timestamps to ``train`` / ``val`` / ``test`` / ``other``."""
    ts = np.asarray(ts_us)
    emb = config.SPLIT_EMBARGO_S * config.US
    out = np.full(ts.shape, OTHER, dtype="<U5")
    for label, start, end in split_windows():
        out[(ts >= start) & (ts < end - emb)] = label
    return out


def split_expr(ts_col: str = "timestamp") -> pl.Expr:
    """Polars expression equivalent of :func:`assign_split`, aliased ``"split"``."""
    ts = pl.col(ts_col)
    emb = config.SPLIT_EMBARGO_S * config.US
    expr = pl.lit(OTHER)
    for label, start, end in split_windows():  # windows disjoint ⇒ fold order is irrelevant
        expr = pl.when((ts >= start) & (ts < end - emb)).then(pl.lit(label)).otherwise(expr)
    return expr.alias("split")
