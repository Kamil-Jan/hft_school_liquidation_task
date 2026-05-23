#!/usr/bin/env python
"""Build sampled feature panels and write ``artifacts/panel_<sym>.parquet``
(plus ``panel_meta_<sym>.parquet`` recording the sampling step).

Usage:  python scripts/build_panel.py [--symbols btc eth] [--rows 3000000]
"""
from __future__ import annotations

import argparse

import polars as pl

from liqsignal import config
from liqsignal.features import build_feature_panel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--rows", type=int, default=3_000_000, help="target sample size per symbol")
    args = p.parse_args()

    art = config.ensure_artifacts()
    for sym in args.symbols:
        panel, step = build_feature_panel(sym, target_rows=args.rows)
        panel.write_parquet(art / f"panel_{sym}.parquet")
        pl.DataFrame([{"sym": sym, "step": step, "sample_n": panel.height,
                       "total_n": step * panel.height}]).write_parquet(art / f"panel_meta_{sym}.parquet")
        print(f"  {sym}: panel {panel.shape}  step={step}  -> artifacts/panel_{sym}.parquet")


if __name__ == "__main__":
    main()
