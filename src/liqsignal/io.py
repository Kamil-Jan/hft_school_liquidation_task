"""Data access layer.

Thin, well-typed loaders for the four raw parquet sources, plus the two access
patterns the rest of the package needs:

* small, fully-materialised arrays for the BBO and liquidation feeds (used for
  ``searchsorted`` forward-fill and proximity features), and
* a memory-bounded batch iterator over the multi-hundred-million-row trade files.

All timestamps stay in epoch microseconds; nothing here mutates the spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from . import config


# ---------------------------------------------------------------------------
# Lightweight containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BookTop:
    """Top-of-book snapshots for one symbol, sorted ascending by ``ts``."""
    ts: np.ndarray          # int64 epoch-us
    mid: np.ndarray         # float64
    spread: np.ndarray      # float32 (ask_price - bid_price)
    bid_amount: np.ndarray  # float32
    ask_amount: np.ndarray  # float32

    @property
    def last_ts(self) -> int:
        return int(self.ts[-1])


@dataclass(frozen=True)
class Liquidations:
    """Liquidation events for one feed, sorted ascending by ``ts``.

    ``signed_notional`` is +notional for buy-side (upward-pressure) liquidations
    and -notional for sell-side, so a windowed sum gives net liquidation pressure.
    For the Bybit feed ``ts`` already includes the +200 ms availability shift.
    """
    ts: np.ndarray               # int64 epoch-us
    side: np.ndarray             # str ('buy'/'sell')
    price: np.ndarray            # float64 (liquidation print price)
    signed_notional: np.ndarray  # float64


# ---------------------------------------------------------------------------
# Lazy scans (for streaming aggregations)
# ---------------------------------------------------------------------------
def scan(source: str, symbol: str) -> pl.LazyFrame:
    """Lazy scan of a raw parquet source (no data read until collected)."""
    return pl.scan_parquet(config.dataset_path(source, symbol))


def row_count(source: str, symbol: str) -> int:
    """Exact row count from parquet metadata (cheap)."""
    return pq.read_metadata(config.dataset_path(source, symbol)).num_rows


def mid_expr() -> pl.Expr:
    """Best-bid/offer mid price expression."""
    return ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid")


# ---------------------------------------------------------------------------
# Frame -> array constructors (work on in-memory frames, e.g. the submission inputs)
# ---------------------------------------------------------------------------
def book_top_from_frame(df: pl.DataFrame) -> BookTop:
    """Build a sorted :class:`BookTop` from a BBO frame (public schema)."""
    book = BookTop(
        ts=df["timestamp"].to_numpy(),
        mid=((df["bid_price"] + df["ask_price"]) / 2.0).to_numpy(),
        spread=(df["ask_price"] - df["bid_price"]).to_numpy().astype(np.float32),
        bid_amount=df["bid_amount"].to_numpy().astype(np.float32),
        ask_amount=df["ask_amount"].to_numpy().astype(np.float32),
    )
    if not np.all(np.diff(book.ts) >= 0):
        raise ValueError("BBO frame is not sorted by timestamp")
    return book


def liquidations_from_frame(df: pl.DataFrame, exchange: str, *, shift_bybit: bool = True) -> Liquidations:
    """Build a sorted, signed :class:`Liquidations` from a liquidation frame.

    For ``exchange == "bybit"`` and ``shift_bybit`` True, the +200 ms availability
    delay is applied before sorting. The Bybit feed is not natively time-sorted.
    """
    ts = df["timestamp"].to_numpy().copy()
    if exchange == "bybit" and shift_bybit:
        ts = ts + config.BYBIT_DELAY_US
    side = df["side"].to_numpy()
    price = df["price"].to_numpy()
    signed_notional = (price * df["amount"].to_numpy()) * np.where(side == "buy", 1.0, -1.0)
    order = np.argsort(ts, kind="stable")
    return Liquidations(ts=ts[order], side=side[order], price=price[order],
                        signed_notional=signed_notional[order])


# ---------------------------------------------------------------------------
# Materialised loaders (read a symbol's full file, then delegate to the above)
# ---------------------------------------------------------------------------
def load_book_top(symbol: str) -> BookTop:
    """Load the full BBO feed for ``symbol`` as sorted arrays."""
    return book_top_from_frame(pl.read_parquet(
        config.dataset_path("bbo", symbol),
        columns=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"]))


def load_liquidations(exchange: str, symbol: str, *, shift_bybit: bool = True) -> Liquidations:
    """Load a liquidation feed for ``symbol`` as sorted, signed arrays."""
    d = pl.read_parquet(config.dataset_path(f"liq_{exchange}", symbol),
                        columns=["timestamp", "side", "price", "amount"])
    return liquidations_from_frame(d, exchange, shift_bybit=shift_bybit)


def sample_trades(symbol: str, target_rows: int) -> tuple[pl.DataFrame, int]:
    """Deterministic uniform sample of ~``target_rows`` trades (every k-th row).

    Returns the sampled frame and the sampling ``step`` (= full_n / sample_n),
    which downstream scoring uses to rescale summed quantities like turnover.
    """
    total = row_count("trades", symbol)
    step = max(1, total // target_rows)
    sample = (
        scan("trades", symbol)
        .with_row_index("ridx")
        .filter(pl.col("ridx") % step == 0)
        .select("timestamp", "side", "price", "amount")
        .collect(engine="streaming")
    )
    return sample, step


def iter_trade_batches(symbol: str, *, batch_size: int = 20_000_000
                       ) -> Iterator[dict[str, np.ndarray]]:
    """Yield trade batches as numpy arrays, bounding peak memory.

    Each yielded dict has keys ``timestamp``, ``price``, ``amount`` and
    ``is_buy`` (bool). Used to score the full trade files without materialising
    them whole.
    """
    import pyarrow.compute as pc
    pf = pq.ParquetFile(config.dataset_path("trades", symbol))
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=["timestamp", "side", "price", "amount"]):
        yield {
            "timestamp": batch.column("timestamp").to_numpy(zero_copy_only=False),
            "price": batch.column("price").to_numpy(zero_copy_only=False),
            "amount": batch.column("amount").to_numpy(zero_copy_only=False),
            "is_buy": pc.equal(batch.column("side"), "buy").to_numpy(zero_copy_only=False),
        }
