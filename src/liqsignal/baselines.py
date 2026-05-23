"""Full-data baselines: ``PnL_all`` and clipped turnover/day per symbol, split and
horizon — the numbers any filter must beat (Score > 0) while staying above the
turnover floor.

Memory-bounded: the BBO mid is held in RAM once and the multi-hundred-million-row
trade files are streamed in PyArrow batches, accumulating weighted sums via
``searchsorted`` forward-fill.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config, io
from .markout import forward_fill_mid, markout_bps
from .splits import TRAIN, VAL, assign_split


def compute_baselines(symbols: tuple[str, ...] = config.SYMBOLS, *,
                      batch_size: int = 20_000_000) -> pl.DataFrame:
    """Return a tidy frame with one row per (symbol, split, tau)."""
    rows: list[dict] = []
    for symbol in symbols:
        book = io.load_book_top(symbol)
        bts, bmid = book.ts, book.mid
        # accumulators keyed by (split, tau) -> [sum_w, sum_w_pnl, n]
        acc: dict[tuple[str, int], list] = {}
        days: dict[str, set] = {TRAIN: set(), VAL: set()}

        for batch in io.iter_trade_batches(symbol, batch_size=batch_size):
            t, price, amount = batch["timestamp"], batch["price"], batch["amount"]
            sign = np.where(batch["is_buy"], 1, -1)
            w = np.minimum(price * amount, config.NOTIONAL_CAP)
            label = assign_split(t)
            day = t // config.DAY_US
            for split in (TRAIN, VAL):
                in_split = label == split
                if not in_split.any():
                    continue
                days[split].update(np.unique(day[in_split]).tolist())
                for tau in config.TAUS:
                    mid_tau, valid = forward_fill_mid(bts, bmid, t + tau * config.US)
                    pnl = markout_bps(price, sign, mid_tau)
                    m = in_split & valid
                    a = acc.setdefault((split, tau), [0.0, 0.0, 0])
                    a[0] += float(w[m].sum())
                    a[1] += float((w[m] * pnl[m]).sum())
                    a[2] += int(m.sum())

        for (split, tau), (sum_w, sum_w_pnl, n) in acc.items():
            n_days = len(days[split])
            rows.append(dict(
                sym=symbol, split=split, tau=tau, n=n, n_days=n_days,
                pnl_all_bps=round(sum_w_pnl / sum_w, 4),
                clipped_turnover_per_day=round(sum_w / n_days, 0),
                total_clipped_turnover=sum_w,
            ))
        del book

    return pl.DataFrame(rows).sort(["sym", "tau", "split"])
