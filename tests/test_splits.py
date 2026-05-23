"""Train/validation boundaries (NumPy and Polars must agree)."""
import numpy as np
import polars as pl

from liqsignal import config
from liqsignal.splits import OTHER, TRAIN, VAL, assign_split, split_expr


def test_boundary_assignment():
    ts = np.array([
        config.TRAIN_START - 1,   # other (before)
        config.TRAIN_START,       # train (inclusive start)
        config.VAL_START - 1,     # train (last us before val)
        config.VAL_START,         # val (inclusive start)
        config.VAL_END - 1,       # val (last us)
        config.VAL_END,           # other (exclusive end)
    ], dtype=np.int64)
    assert list(assign_split(ts)) == [OTHER, TRAIN, TRAIN, VAL, VAL, OTHER]


def test_numpy_and_polars_agree():
    ts = np.linspace(config.TRAIN_START - config.DAY_US,
                     config.VAL_END + config.DAY_US, 1000).astype(np.int64)
    np_labels = assign_split(ts)
    pl_labels = pl.DataFrame({"timestamp": ts}).select(split_expr())["split"].to_numpy()
    assert (np_labels == pl_labels).all()
