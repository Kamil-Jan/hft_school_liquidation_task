#!/usr/bin/env python
"""Build the 1s trade-flow grid → ``artifacts/flow_grid_<sym>.parquet``.

Streams the full trade tape per symbol and aggregates per UTC second:
``signed_vol`` (buy − sell amount), ``tot_vol`` (total amount), ``cnt`` (count).
The flow features in ``features.compute_features`` look up windowed prefix-sums
over this grid — the panel uses the full-tape grid built here, while ``signal()``
builds the same grid in memory from the passed trades (``io.flow_grid_from_trades``).

Usage:  python scripts/build_flow_grid.py [--symbols btc eth]
"""
from __future__ import annotations

import argparse
import time

import polars as pl

from liqsignal import config, io


def build(sym: str) -> None:
    t0 = time.time()
    is_buy = pl.col("side") == "buy"
    signed = pl.when(is_buy).then(pl.col("amount")).otherwise(-pl.col("amount"))
    g = (io.scan("trades", sym)
         .with_columns(sec=(pl.col("timestamp") // config.US))
         .group_by("sec")
         .agg(signed_vol=signed.sum().cast(pl.Float32),
              tot_vol=pl.col("amount").sum().cast(pl.Float32),
              cnt=pl.len().cast(pl.Int32))
         .sort("sec")
         .collect(engine="streaming"))
    out = config.ensure_artifacts() / f"flow_grid_{sym}.parquet"
    g.write_parquet(out)
    print(f"  [{sym}] {g.height:,} seconds -> {out}  ({time.time()-t0:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    args = p.parse_args()
    for sym in args.symbols:
        build(sym)


if __name__ == "__main__":
    main()
