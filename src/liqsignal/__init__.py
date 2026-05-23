"""liqsignal — liquidation-driven maker-trade filter for the Binance perpetual.

A small research package implementing the task in ``description.md``: filter
Binance maker trades using trade / BBO / liquidation data so that the kept trades
have a better markout than the unfiltered baseline, subject to a daily turnover
floor.

Module map
----------
config     paths, universe, and the frozen task-spec constants
splits     train / validation assignment (NumPy + Polars)
io         data access: lazy scans, materialised arrays, batched trade iterator
markout    spec-critical maker-PnL math (forward-filled mid, markout in bps)
scoring    Score, PnL_all/kept/filtered and the turnover constraint
features   feature engineering + sampled feature-panel assembly
baselines  full-data PnL_all and turnover/day
signal     the submission entry point (keep-all baseline; pluggable filter)

Typical use
-----------
>>> from liqsignal import io, markout, scoring
>>> book = io.load_book_top("btc")
>>> # ... compute pnl, weights, filter ...
>>> scoring.evaluate_filter(pnl, w, f, n_days=62)
"""
from __future__ import annotations

from . import analysis, baselines, config, features, io, markout, scoring, signal, splits
from .scoring import ScoreResult, evaluate_filter, weighted_mean
from .signal import signal as run_signal

# `model` and `report` pull in scikit-learn / matplotlib; import lazily to keep the
# core (io/markout/scoring) usable without them.

__version__ = "0.1.0"

__all__ = [
    "analysis", "baselines", "config", "features", "io", "markout", "scoring",
    "signal", "splits",
    "ScoreResult", "evaluate_filter", "weighted_mean", "run_signal",
]
