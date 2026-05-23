"""Submission entry point.

The grader calls :func:`signal` with the four per-symbol frames (same schema as
the public files) and expects, for each horizon, a 0/1 array of length
``len(trades)`` (1 = filter out, 0 = keep).

Default behaviour: if per-horizon models are present in ``artifacts/`` they are
loaded and applied (feature context built once, features + predictions computed in
memory-bounded batches). Each model carries the **fitted Score-maximising
threshold** from training, which is applied by default; only if a model predates
threshold persistence does it fall back to the expected-value cutoff (``cost``).
If models are absent it falls back to the keep-all baseline. A custom per-horizon
``filter_fn`` can be supplied to override entirely.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import polars as pl

from .config import TAUS

# A FilterFn returns the 0/1 filter for one horizon given the four input frames.
FilterFn = Callable[[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, int], np.ndarray]

BATCH = 20_000_000


def signal(trades: pl.DataFrame, bbo: pl.DataFrame,
           liq_binance: pl.DataFrame, liq_bybit: pl.DataFrame,
           *, filter_fn: Optional[FilterFn] = None,
           thresholds: Optional[dict[int, float]] = None,
           cost: float = 0.0) -> dict[int, np.ndarray]:
    """Return ``{tau: 0/1 array of length len(trades)}``.

    * ``filter_fn`` set  → generic per-horizon path (used for experiments).
    * else, trained models present → model + threshold path.
    * else → keep-all baseline.

    ``thresholds`` optionally supplies a per-tau score cutoff (e.g. a fitted
    Score-maximising threshold); when omitted, the threshold persisted alongside
    each model is used, falling back to the expected-value rule (cutoff ``cost``)
    where none was saved.
    """
    n = trades.height

    if filter_fn is not None:
        out = {}
        for tau in TAUS:
            f = np.asarray(filter_fn(trades, bbo, liq_binance, liq_bybit, tau), dtype=np.int8)
            if f.shape != (n,):
                raise ValueError(f"filter_fn returned {f.shape}, expected ({n},) for tau={tau}")
            out[tau] = f
        return out

    return _model_signal(trades, bbo, liq_binance, liq_bybit,
                         thresholds=thresholds, cost=cost)


def _model_signal(trades, bbo, liq_binance, liq_bybit, *, thresholds, cost) -> dict[int, np.ndarray]:
    # Lazy imports so the keep-all baseline never requires scikit-learn.
    from . import features, io, model
    from .markout import trade_sign

    n = trades.height
    loaded = {tau: model.load(tau) for tau in TAUS}
    if any(m is None for m, _ in loaded.values()):
        warnings.warn("no trained models in artifacts/ — returning keep-all baseline")
        return {tau: np.zeros(n, dtype=np.int8) for tau in TAUS}

    # Default to each model's persisted Score-maximising threshold; None entries
    # (older models) fall back to the expected-value cutoff `cost` below.
    if thresholds is None:
        thresholds = {tau: model.load_threshold(tau) for tau in TAUS}
        thresholds = {tau: thr for tau, thr in thresholds.items() if thr is not None}

    ctx = features.build_context(
        io.book_top_from_frame(bbo),
        io.liquidations_from_frame(liq_binance, "binance"),
        io.liquidations_from_frame(liq_bybit, "bybit"),
    )
    t_all = trades["timestamp"].to_numpy()
    sign_all = trade_sign(trades["side"].to_numpy())
    price_all = trades["price"].to_numpy()

    out = {tau: np.zeros(n, dtype=np.int8) for tau in TAUS}
    for start in range(0, n, BATCH):
        sl = slice(start, min(start + BATCH, n))
        feats = features.compute_features(ctx, t_all[sl], sign_all[sl], price_all[sl])
        for tau in TAUS:
            mdl, cols = loaded[tau]
            score = model.predict_from_features(mdl, feats, cols)
            cut = cost if thresholds is None else thresholds.get(tau, cost)
            out[tau][sl] = (score < cut).astype(np.int8)
    return out
