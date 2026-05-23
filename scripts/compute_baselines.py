#!/usr/bin/env python
"""Compute full-data baselines and write ``artifacts/baselines.parquet``.

Usage:  python scripts/compute_baselines.py [--symbols btc eth] [--batch-size N]
"""
from __future__ import annotations

import argparse

import polars as pl

from liqsignal import config
from liqsignal.baselines import compute_baselines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--batch-size", type=int, default=20_000_000)
    args = p.parse_args()

    base = compute_baselines(tuple(args.symbols), batch_size=args.batch_size)
    out = config.ensure_artifacts() / "baselines.parquet"
    base.write_parquet(out)

    pl.Config.set_tbl_rows(40)
    print(base.select("sym", "split", "tau", "n", "n_days",
                      "pnl_all_bps", "clipped_turnover_per_day"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
