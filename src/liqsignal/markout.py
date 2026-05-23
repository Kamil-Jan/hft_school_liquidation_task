"""Maker markout — the spec-critical price math.

For trade ``i`` and horizon ``tau`` (seconds), with ``s_i = +1`` for a taker buy
(maker sell) and ``-1`` for a taker sell (maker buy)::

    m_i(tau)   = forward-filled Binance mid at t_i + tau
    pnl_i(tau) = -s_i * (m_i(tau) - p_i) / p_i * 1e4 + REBATE_BPS      [bps]

A trade is *excluded* (pnl = NaN) when ``t_i + tau`` falls before the first or
after the last available BBO tick, per the spec.

Everything here is pure (no I/O) and operates on plain NumPy arrays so it can be
unit-tested against hand-computed examples.
"""
from __future__ import annotations

import numpy as np

from .config import REBATE_BPS, US


def trade_sign(side: np.ndarray) -> np.ndarray:
    """+1 for taker buy (maker sell), -1 for taker sell (maker buy)."""
    return np.where(side == "buy", 1, -1).astype(np.int8)


def last_index_at(sorted_ts: np.ndarray, query_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last entry of ``sorted_ts`` at-or-before each query time.

    Returns ``(idx, valid)`` where ``valid`` is False when the query precedes the
    first entry or follows the last (the spec's "beyond available BBO" rule). The
    returned indices are clipped into range; callers must mask with ``valid``.
    """
    idx = np.searchsorted(sorted_ts, query_us, side="right") - 1
    valid = (idx >= 0) & (query_us <= sorted_ts[-1])
    return np.clip(idx, 0, len(sorted_ts) - 1), valid


def forward_fill_mid(bbo_ts: np.ndarray, bbo_mid: np.ndarray,
                     query_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last observed mid at-or-before each query time.

    Returns ``(mid, valid)``; ``mid`` is NaN wherever ``valid`` is False.
    """
    idx, valid = last_index_at(bbo_ts, query_us)
    mid = bbo_mid[idx].astype(np.float64)
    mid[~valid] = np.nan
    return mid, valid


def markout_bps(price: np.ndarray, sign: np.ndarray, mid_tau: np.ndarray) -> np.ndarray:
    """Maker PnL in bps. NaN propagates from excluded ``mid_tau`` entries."""
    return -sign * (mid_tau - price) / price * 1e4 + REBATE_BPS


def compute_markout(trade_ts: np.ndarray, sign: np.ndarray, price: np.ndarray,
                    bbo_ts: np.ndarray, bbo_mid: np.ndarray, tau: int) -> np.ndarray:
    """Maker markout (bps) for every trade at horizon ``tau`` seconds."""
    mid_tau, valid = forward_fill_mid(bbo_ts, bbo_mid, trade_ts + tau * US)
    pnl = markout_bps(price, sign, mid_tau)
    pnl[~valid] = np.nan
    return pnl
