#!/usr/bin/env python
"""Conditional-markout study + single-feature filter sweep.

Reports, per symbol/horizon: the baseline to beat, w-weighted markout by feature
quantile (train vs val), and the train-fit / val-applied Score of a 'keep the
best-ranked trades' rule at several selectivity levels.

Usage:  python scripts/run_study.py [--symbols btc eth] [--tau 120]
"""
from __future__ import annotations

import argparse

import polars as pl

from liqsignal import analysis, config
from liqsignal.splits import TRAIN, VAL

CANDIDATES = ["obi_signed", "px_vs_mid_bps", "ret_1s_signed", "ret_5s_signed",
              "bybit_liqpress_30s", "binance_liqpress_30s"]
KEEP_FRACS = (0.5, 0.2, 0.1, 0.05)


def print_baselines() -> None:
    path = config.ARTIFACTS_DIR / "baselines.parquet"
    if not path.exists():
        print("(run scripts/compute_baselines.py first for full-data baselines)\n")
        return
    b = pl.read_parquet(path).with_columns(
        turnover_headroom=(pl.col("clipped_turnover_per_day") / config.TURNOVER_MIN_PER_DAY))
    pl.Config.set_tbl_rows(40)
    print("=== FULL-DATA BASELINES (PnL_all to beat; turnover headroom vs 500k/day floor) ===")
    print(b.select("sym", "split", "tau", "pnl_all_bps",
                  "clipped_turnover_per_day", "turnover_headroom"))
    print()


def study_conditional(symbol: str, tau: int) -> None:
    panel, step = analysis.load_panel(symbol)
    tr = panel.filter(pl.col("split") == TRAIN)
    va = panel.filter(pl.col("split") == VAL)
    pnl_col = f"pnl_{tau}"
    print(f"\n################ {symbol.upper()}  tau={tau}s  (sample n={panel.height:,}, step={step}) ################")
    print(f"  baseline PnL_all  train={analysis.weighted_markout(tr, pnl_col):+.3f}  "
          f"val={analysis.weighted_markout(va, pnl_col):+.3f} bps")
    print("\n  w-weighted markout by feature quintile (train):")
    for feat in CANDIDATES:
        g = analysis.conditional_markout(tr, feat, pnl_col)
        cells = "  ".join(f"Q{r['bucket']}:{r['wpnl']:+.2f}" for r in g.iter_rows(named=True))
        print(f"    {feat:22s} {cells}")


def study_sweep(symbol: str, tau: int) -> None:
    panel, step = analysis.load_panel(symbol)
    tr = panel.filter(pl.col("split") == TRAIN)
    va = panel.filter(pl.col("split") == VAL)
    pnl_col = f"pnl_{tau}"
    print(f"\n  keep-best filters (fit on train, applied to val), tau={tau}s")
    print(f"  {'feature':22s} {'keep%':>6s} | {'trScore':>8s} | {'vaScore':>8s} {'vaKept':>8s} "
          f"{'vaTurn/day':>14s} ok")
    for feat in CANDIDATES:
        for keep in KEEP_FRACS:
            direction, threshold = analysis.fit_keep_best(tr, feat, pnl_col, keep)
            f_tr = analysis.apply_keep_best(tr, feat, direction, threshold)
            f_va = analysis.apply_keep_best(va, feat, direction, threshold)
            r_tr = analysis.score_split(tr, pnl_col, f_tr, step)
            r_va = analysis.score_split(va, pnl_col, f_va, step)
            print(f"  {feat:22s} {keep*100:5.0f}% | {r_tr.score:+8.3f} | "
                  f"{r_va.score:+8.3f} {r_va.pnl_kept:+8.3f} "
                  f"{r_va.kept_turnover_per_day:14,.0f} {'Y' if r_va.constraint_ok else 'N'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--tau", type=int, default=120, choices=config.TAUS)
    args = p.parse_args()

    print_baselines()
    for sym in args.symbols:
        study_conditional(sym, args.tau)
    for sym in args.symbols:
        study_sweep(sym, args.tau)


if __name__ == "__main__":
    main()
