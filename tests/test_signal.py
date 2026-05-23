"""Submission-contract tests for signal()."""
import numpy as np
import polars as pl
import pytest

from liqsignal.config import TAUS
from liqsignal.signal import signal


def _trades(n):
    return pl.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64),
        "side": ["buy", "sell"] * (n // 2),
        "price": np.full(n, 100.0),
        "amount": np.full(n, 1.0),
    })


def test_filter_fn_path_shapes():
    tr = _trades(6)
    empty = pl.DataFrame()
    # custom filter: filter every other trade
    def fn(trades, bbo, lb, ly, tau):
        return (np.arange(trades.height) % 2).astype(np.int8)
    out = signal(tr, empty, empty, empty, filter_fn=fn)
    assert set(out) == set(TAUS)
    for tau in TAUS:
        assert out[tau].shape == (6,)
        assert set(np.unique(out[tau])) <= {0, 1}


def test_filter_fn_wrong_shape_raises():
    tr = _trades(4)
    empty = pl.DataFrame()
    with pytest.raises(ValueError):
        signal(tr, empty, empty, empty, filter_fn=lambda *a: np.zeros(3, np.int8))
