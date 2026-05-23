#!/usr/bin/env python
"""EDA pass 1: streaming aggregates from the large trades/BBO files.

Writes small artifact tables (per-minute aggregates, log histograms, summary
stats, quantiles, spread distributions) consumed by the exploration notebook.
Everything is exact (group-by) except quantiles, computed on a deterministic
~3M-row subsample.

Usage:  python scripts/eda/precompute_aggregates.py
"""
from __future__ import annotations

import time

import polars as pl

from liqsignal import config, io

ART = config.ensure_artifacts()
MIN_US = 60 * config.US


def log_hist(lf: pl.LazyFrame, value: pl.Expr, per_decade: int = 10) -> pl.DataFrame:
    """Exact histogram of a positive quantity on a log10 grid."""
    binned = (lf.filter(value > 0)
              .select(b=(value.log10() * per_decade).floor().cast(pl.Int32))
              .group_by("b").agg(count=pl.len()).sort("b").collect(engine="streaming"))
    return binned.with_columns(bin_lo=(10.0 ** (pl.col("b") / per_decade)),
                               bin_hi=(10.0 ** ((pl.col("b") + 1) / per_decade)))


def quantile_sample(lf: pl.LazyFrame, exprs: dict[str, pl.Expr], total: int,
                    target: int = 3_000_000) -> pl.DataFrame:
    step = max(1, total // target)
    return (lf.with_row_index("ridx").filter(pl.col("ridx") % step == 0)
            .select(**exprs).collect(engine="streaming"))


def do_trades(sym: str) -> None:
    t0 = time.time()
    lf = io.scan("trades", sym)
    notional = pl.col("price") * pl.col("amount")
    is_buy = pl.col("side") == "buy"

    minute = (lf.with_columns(m=(pl.col("timestamp") // MIN_US)).group_by("m").agg(
        n=pl.len(), sum_amount=pl.col("amount").sum(), sum_notional=notional.sum(),
        buy_n=is_buy.sum(),
        buy_amount=pl.when(is_buy).then(pl.col("amount")).otherwise(0.0).sum(),
        buy_notional=pl.when(is_buy).then(notional).otherwise(0.0).sum(),
        price_first=pl.col("price").first(), price_last=pl.col("price").last(),
        price_min=pl.col("price").min(), price_max=pl.col("price").max(),
    ).sort("m").collect(engine="streaming"))
    minute.write_parquet(ART / f"trades_minute_{sym}.parquet")
    total = int(minute["n"].sum())

    log_hist(lf, pl.col("amount")).write_parquet(ART / f"trades_amount_hist_{sym}.parquet")
    log_hist(lf, notional).write_parquet(ART / f"trades_notional_hist_{sym}.parquet")

    lf.select(
        count=pl.len(), amount_mean=pl.col("amount").mean(), amount_std=pl.col("amount").std(),
        amount_min=pl.col("amount").min(), amount_max=pl.col("amount").max(),
        notional_mean=notional.mean(), notional_min=notional.min(), notional_max=notional.max(),
        price_min=pl.col("price").min(), price_max=pl.col("price").max(),
        buy_n=is_buy.sum(), zero_amount=(pl.col("amount") <= 0).sum(),
    ).collect(engine="streaming").write_parquet(ART / f"trades_stats_{sym}.parquet")

    samp = quantile_sample(lf, dict(amount=pl.col("amount"), notional=notional,
                                    price=pl.col("price")), total)
    pl.DataFrame([dict(q=q, amount=samp["amount"].quantile(q), notional=samp["notional"].quantile(q),
                       price=samp["price"].quantile(q))
                  for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999)]) \
      .write_parquet(ART / f"trades_quantiles_{sym}.parquet")
    print(f"  [trades {sym}] {total:,} trades, {time.time()-t0:.1f}s")


def do_bbo(sym: str) -> None:
    t0 = time.time()
    lf = io.scan("bbo", sym)
    mid = (pl.col("bid_price") + pl.col("ask_price")) / 2
    spread = pl.col("ask_price") - pl.col("bid_price")
    rel_bps = spread / mid * 1e4

    minute = (lf.with_columns(m=(pl.col("timestamp") // MIN_US)).group_by("m").agg(
        n=pl.len(), mid_first=mid.first(), mid_last=mid.last(),
        mid_min=mid.min(), mid_max=mid.max(), spread_mean=spread.mean(),
        rel_bps_mean=rel_bps.mean(), bid_amt_mean=pl.col("bid_amount").mean(),
        ask_amt_mean=pl.col("ask_amount").mean(),
        crossed=(pl.col("bid_price") >= pl.col("ask_price")).sum(),
        locked=(pl.col("bid_price") == pl.col("ask_price")).sum(),
    ).sort("m").collect(engine="streaming"))
    minute.write_parquet(ART / f"bbo_minute_{sym}.parquet")
    total = int(minute["n"].sum())

    sp = (lf.select(b=(rel_bps.clip(0, 50) * 10).round().cast(pl.Int32))
          .group_by("b").agg(count=pl.len()).sort("b").collect(engine="streaming"))
    sp.with_columns(bps=(pl.col("b") / 10.0)).write_parquet(ART / f"bbo_spread_hist_{sym}.parquet")

    lf.select(
        count=pl.len(), spread_mean=spread.mean(), spread_min=spread.min(), spread_max=spread.max(),
        rel_bps_mean=rel_bps.mean(), bid_amt_mean=pl.col("bid_amount").mean(),
        ask_amt_mean=pl.col("ask_amount").mean(),
        crossed=(pl.col("bid_price") >= pl.col("ask_price")).sum(),
        locked=(pl.col("bid_price") == pl.col("ask_price")).sum(),
        nonpos_spread=(spread <= 0).sum(),
    ).collect(engine="streaming").write_parquet(ART / f"bbo_stats_{sym}.parquet")

    samp = quantile_sample(lf, dict(rel_bps=rel_bps, spread=spread,
                                    bid_amount=pl.col("bid_amount"),
                                    ask_amount=pl.col("ask_amount")), total)
    pl.DataFrame([dict(q=q, rel_bps=samp["rel_bps"].quantile(q), spread=samp["spread"].quantile(q),
                       bid_amount=samp["bid_amount"].quantile(q), ask_amount=samp["ask_amount"].quantile(q))
                  for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)]) \
      .write_parquet(ART / f"bbo_quantiles_{sym}.parquet")
    print(f"  [bbo {sym}] {total:,} ticks, {time.time()-t0:.1f}s")


def main() -> None:
    t0 = time.time()
    for sym in config.SYMBOLS:
        do_trades(sym)
        do_bbo(sym)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
