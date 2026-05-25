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

# Feature computation materialises ~50 float64 columns per row, so the batch size
# bounds peak memory: ~50 × 8 B × BATCH ≈ 2 GB of feature matrix at 5M. Kept well
# under the 16 GB budget (the BBO/liq context + trade frame sit alongside it).
BATCH = 5_000_000


def _infer_symbol(trades: pl.DataFrame, bbo: pl.DataFrame) -> Optional[str]:
    """Best-effort symbol detection so per-symbol models can be selected without a
    symbol argument: use a ``ticker`` column if present, else the price level
    (BTC trades in the tens of thousands, ETH in the thousands — they never overlap)."""
    for fr in (trades, bbo):
        if "ticker" in fr.columns and fr.height:
            t = str(fr["ticker"][0]).lower()
            if "btc" in t:
                return "btc"
            if "eth" in t:
                return "eth"
    if trades.height:
        return "btc" if float(trades["price"].median()) > 10_000 else "eth"
    return None


def signal(trades: pl.DataFrame, bbo: pl.DataFrame,
           liq_binance: pl.DataFrame, liq_bybit: pl.DataFrame,
           *, filter_fn: Optional[FilterFn] = None,
           thresholds: Optional[dict[int, float]] = None,
           cost: float = 0.0, symbol: Optional[str] = None) -> dict[int, np.ndarray]:
    """Return ``{tau: 0/1 array of length len(trades)}``.

    * ``filter_fn`` set  → generic per-horizon path (used for experiments).
    * else, trained models present → model + threshold path.
    * else → keep-all baseline.

    ``thresholds`` optionally supplies a per-tau score cutoff (e.g. a fitted
    Score-maximising threshold); when omitted, the threshold persisted alongside
    each model is used, falling back to the expected-value rule (cutoff ``cost``)
    where none was saved. ``symbol`` (``"btc"``/``"eth"``) selects the per-symbol model;
    when omitted it is inferred from the data (a ``ticker`` column if present, else the
    price level). If no model exists for the symbol, returns the keep-all baseline.
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
                         thresholds=thresholds, cost=cost, symbol=symbol)


def _model_signal(trades, bbo, liq_binance, liq_bybit, *, thresholds, cost, symbol=None) -> dict[int, np.ndarray]:
    # Lazy imports so the keep-all baseline never requires scikit-learn.
    from . import features, io, model
    from .markout import trade_sign

    n = trades.height
    sym = symbol if symbol is not None else _infer_symbol(trades, bbo)
    if sym is None:
        warnings.warn("could not infer symbol — returning keep-all baseline")
        return {tau: np.zeros(n, dtype=np.int8) for tau in TAUS}

    loaded = {tau: model.load(tau, sym) for tau in TAUS}
    if any(m is None for m, _ in loaded.values()):
        warnings.warn(f"no trained models for '{sym}' in artifacts/ — returning keep-all baseline")
        return {tau: np.zeros(n, dtype=np.int8) for tau in TAUS}

    # Default to each model's persisted Score-maximising threshold; None entries
    # (older models) fall back to the expected-value cutoff `cost` below.
    if thresholds is None:
        thresholds = {tau: model.load_threshold(tau, sym) for tau in TAUS}
        thresholds = {tau: thr for tau, thr in thresholds.items() if thr is not None}

    t_all = trades["timestamp"].to_numpy()
    side_all = trades["side"].to_numpy()
    sign_all = trade_sign(side_all)
    price_all = trades["price"].to_numpy()
    # Build the 1s trade-flow grid from the passed trades (the panel uses the full-tape
    # grid; here the passed frame *is* the data). Built once; reused across batches.
    flow = io.flow_grid_from_trades(t_all, side_all == "buy", trades["amount"].to_numpy())
    ctx = features.build_context(
        io.book_top_from_frame(bbo),
        io.liquidations_from_frame(liq_binance, "binance"),
        io.liquidations_from_frame(liq_bybit, "bybit"),
        flow=flow,
    )

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
