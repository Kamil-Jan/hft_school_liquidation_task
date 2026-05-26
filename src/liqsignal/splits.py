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

import datetime as dt

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


# ---------------------------------------------------------------------------
# Walk-forward (expanding-window) out-of-sample folds
# ---------------------------------------------------------------------------
# Independent of the train/val/test split above: used by the backtest harness to
# judge a model spec on several *consecutive held-out months*. Each fold trains on
# everything before an OOS month (minus an embargo so no train markout window reaches
# into it) and is scored on that one month. Operates on the existing panels via
# timestamp masks — no panel rebuild, no config edits.

def month_starts(start_us: int, end_us: int) -> list[int]:
    """UTC month-start boundaries spanning ``[start_us, end_us]`` (µs), end included.

    The first element is the month start at/just before ``start_us``-month; the last
    element is exactly ``end_us`` (the data's exclusive right edge) so the trailing
    partial month is bounded.
    """
    d = dt.datetime.fromtimestamp(start_us / config.US, tz=dt.timezone.utc)
    cur = dt.datetime(d.year, d.month, 1, tzinfo=dt.timezone.utc)
    out: list[int] = []
    while True:
        us = int(cur.timestamp()) * config.US
        if us > end_us:
            break
        out.append(us)
        cur = dt.datetime(cur.year + cur.month // 12, cur.month % 12 + 1, 1,
                          tzinfo=dt.timezone.utc)
    if not out or out[-1] < end_us:
        out.append(end_us)
    return out


def walk_forward_folds(*, train_start: int | None = None, data_end: int | None = None,
                       n_oos: int = 3, embargo_s: int | None = None
                       ) -> list[tuple[int, int, int, int]]:
    """Expanding-window folds as ``(train_start, train_end, oos_start, oos_end)`` (µs).

    The last ``n_oos`` calendar months of ``[train_start, data_end)`` become OOS
    months in turn; each fold trains on ``[train_start, oos_start - embargo)`` and is
    scored on ``[oos_start, oos_end)`` (right edge exclusive). Defaults span the
    configured data range (Nov→Apr ⇒ OOS = Feb, Mar, Apr).
    """
    train_start = config.TRAIN_START if train_start is None else train_start
    data_end = config.SPLIT_END if data_end is None else data_end
    emb = (config.SPLIT_EMBARGO_S if embargo_s is None else embargo_s) * config.US
    bounds = month_starts(train_start, data_end)
    # consecutive (oos_start, oos_end) month pairs that begin strictly after train_start
    pairs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i] > train_start]
    return [(train_start, oos_start - emb, oos_start, oos_end)
            for oos_start, oos_end in pairs[-n_oos:]]


def month_label(us: int) -> str:
    """``"YYYY-MM"`` UTC label for an epoch-µs timestamp (fold/month tagging)."""
    d = dt.datetime.fromtimestamp(us / config.US, tz=dt.timezone.utc)
    return f"{d.year}-{d.month:02d}"
