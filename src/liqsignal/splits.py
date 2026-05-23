"""Train / validation / out-of-range assignment from timestamps.

Provided both as a NumPy function (for array pipelines) and a Polars expression
(for lazy/streaming pipelines) so the two never disagree on the boundaries.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config

TRAIN = "train"
VAL = "val"
OTHER = "other"


def assign_split(ts_us: np.ndarray) -> np.ndarray:
    """Map epoch-microsecond timestamps to ``"train"`` / ``"val"`` / ``"other"``."""
    return np.where(
        (ts_us >= config.TRAIN_START) & (ts_us < config.VAL_START), TRAIN,
        np.where((ts_us >= config.VAL_START) & (ts_us < config.VAL_END), VAL, OTHER),
    )


def split_expr(ts_col: str = "timestamp") -> pl.Expr:
    """Polars expression equivalent of :func:`assign_split`, aliased ``"split"``."""
    ts = pl.col(ts_col)
    return (
        pl.when((ts >= config.TRAIN_START) & (ts < config.VAL_START)).then(pl.lit(TRAIN))
        .when((ts >= config.VAL_START) & (ts < config.VAL_END)).then(pl.lit(VAL))
        .otherwise(pl.lit(OTHER))
        .alias("split")
    )
