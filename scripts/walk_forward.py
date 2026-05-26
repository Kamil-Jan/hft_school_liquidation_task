#!/usr/bin/env python
"""Walk-forward out-of-sample evaluation: judge model specs on several held-out months.

Runs each named spec through the expanding-window backtest (default OOS = Feb, Mar, Apr),
prints per-fold Scores and a mean ± std summary per (symbol, tau), and writes
``artifacts/report/walkforward.parquet``. The first spec is the shipped baseline so every
other number is read against it.

Usage:  python scripts/walk_forward.py [--symbols btc eth] [--n-oos 3]
"""
from __future__ import annotations

import argparse
import os
import warnings

# Silence LightGBM's nameless-array warning here and in joblib/loky workers (see train_model.py).
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import polars as pl

from liqsignal import analysis, backtest, config
from liqsignal.splits import walk_forward_folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--n-oos", type=int, default=3, help="number of trailing OOS months")
    p.add_argument("--specs", default="baseline", choices=sorted(backtest.SPEC_SETS),
                   help="which spec set to compare (baseline | regime | objective | features | shipped)")
    p.add_argument("--taus", nargs="+", type=int, default=list(config.TAUS))
    args = p.parse_args()

    panels, steps = {}, {}
    for sym in args.symbols:
        panels[sym], steps[sym] = analysis.load_panel(sym)

    folds = walk_forward_folds(n_oos=args.n_oos)
    from liqsignal.splits import month_label
    print(f"walk-forward folds (OOS months): "
          f"{', '.join(month_label(oos) for _, _, oos, _ in folds)}")
    specs = backtest.SPEC_SETS[args.specs]()
    print(f"spec set '{args.specs}': {', '.join(name for name, _ in specs)}; "
          f"symbols={args.symbols}; taus={args.taus}\n")

    long = backtest.evaluate_specs(panels, steps, specs, taus=tuple(args.taus), folds=folds)

    out_dir = config.ensure_artifacts() / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    long.write_parquet(out_dir / f"walkforward_{args.specs}.parquet")

    pl.Config.set_tbl_rows(200)
    summary = backtest.summarize(long)
    print("=== mean OOS Score per (sym, tau, spec) ===")
    print(summary)
    print("\n=== per-fold Score (sym, tau, spec, month) ===")
    print(long.select(["sym", "tau", "spec", "month", "score", "kept_turn_per_day",
                       "frac_filt", "constraint_ok"]).sort(["sym", "tau", "spec", "month"]))
    print(f"\nwrote -> {out_dir / f'walkforward_{args.specs}.parquet'}")


if __name__ == "__main__":
    main()
