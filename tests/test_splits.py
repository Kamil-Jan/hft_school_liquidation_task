"""Train/validation/test boundaries: NumPy and Polars must agree, embargo purges
the gap before each boundary, and USE_TEST toggles the test window."""
import numpy as np
import polars as pl

from liqsignal import config
from liqsignal.splits import OTHER, TEST, TRAIN, VAL, assign_split, split_expr, split_windows

EMB = config.SPLIT_EMBARGO_S * config.US


def test_boundary_assignment_with_embargo(monkeypatch):
    monkeypatch.setattr(config, "USE_TEST", True)
    ts = np.array([
        config.TRAIN_START - 1,        # other (before train)
        config.TRAIN_START,            # train (inclusive start)
        config.VAL_START - EMB - 1,    # train (last us before the purge gap)
        config.VAL_START - EMB,        # other (embargo gap before val)
        config.VAL_START - 1,          # other (still in the gap)
        config.VAL_START,              # val (inclusive start)
        config.TEST_START - EMB - 1,   # val (before the gap)
        config.TEST_START - 1,         # other (gap before test)
        config.TEST_START,             # test (inclusive start)
        config.SPLIT_END - EMB - 1,    # test (before the trailing gap)
        config.SPLIT_END - 1,          # other (trailing gap)
        config.SPLIT_END,              # other (exclusive end)
    ], dtype=np.int64)
    assert list(assign_split(ts)) == [
        OTHER, TRAIN, TRAIN, OTHER, OTHER, VAL, VAL, OTHER, TEST, TEST, OTHER, OTHER]


def test_numpy_and_polars_agree():
    ts = np.linspace(config.TRAIN_START - config.DAY_US,
                     config.SPLIT_END + config.DAY_US, 5000).astype(np.int64)
    np_labels = assign_split(ts)
    pl_labels = pl.DataFrame({"timestamp": ts}).select(split_expr())["split"].to_numpy()
    assert (np_labels == pl_labels).all()


def test_use_test_on_has_three_windows(monkeypatch):
    monkeypatch.setattr(config, "USE_TEST", True)
    assert [w[0] for w in split_windows()] == [TRAIN, VAL, TEST]


def test_use_test_off_folds_test_into_val(monkeypatch):
    monkeypatch.setattr(config, "USE_TEST", False)
    assert [w[0] for w in split_windows()] == [TRAIN, VAL]
    # a timestamp inside the former test window is now validation, and nothing is test
    t = np.array([config.TEST_START + config.DAY_US], dtype=np.int64)
    assert assign_split(t)[0] == VAL
    grid = np.linspace(config.TRAIN_START, config.SPLIT_END - EMB - 1, 2000).astype(np.int64)
    assert TEST not in set(assign_split(grid))
