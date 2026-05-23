#!/usr/bin/env python
"""Train the per-horizon markout models, fit thresholds, score panels, and emit the report.

Pools BTC+ETH train rows (features are scale-free), fits one HistGBR per tau with
sample weights w_i, chooses the Score-maximising cutoff by CV on train, scores both
panels, persists models to artifacts/, and writes artifacts/report/report.md.

Usage:  python scripts/train_model.py [--symbols btc eth] [--min-keep-frac 0.05]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from liqsignal import analysis, config, model, report
from liqsignal.features import feature_columns
from liqsignal.splits import TRAIN, VAL


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--min-keep-frac", type=float, default=0.05)
    args = p.parse_args()

    panels, steps = {}, {}
    for sym in args.symbols:
        panel, step = analysis.load_panel(sym)
        panels[sym], steps[sym] = panel, step
    features = feature_columns(next(iter(panels.values())).columns)
    print(f"{len(features)} features; symbols={args.symbols}")

    pooled_train = pl.concat([p.filter(pl.col("split") == TRAIN) for p in panels.values()])

    thresholds: dict[int, float] = {}
    models: dict[int, object] = {}
    for tau in config.TAUS:
        mdl, feats = model.train_markout_model(pooled_train, tau, features=features)
        models[tau] = mdl
        # score every panel
        for sym in panels:
            scores = model.predict_markout(mdl, panels[sym], feats)
            panels[sym] = panels[sym].with_columns(pl.Series(f"score_{tau}", scores))
        # fit Score-maximising threshold on pooled train (CV)
        ptr = pooled_train.with_columns(
            pl.Series(f"score_{tau}", model.predict_markout(mdl, pooled_train, feats)))
        thr, cv = analysis.fit_score_threshold(
            ptr[f"score_{tau}"].to_numpy(), ptr[f"pnl_{tau}"].to_numpy(),
            ptr["w"].to_numpy(), ptr["day"].to_numpy(),
            min_keep_frac=args.min_keep_frac)
        thresholds[tau] = thr
        # persist with the threshold so signal() applies this exact operating point
        model.save(mdl, feats, tau, threshold=thr)
        print(f"  tau={tau}: score-max threshold={thr:+.4f} bps (CV Score={cv:+.4f})")

    out = report.generate(panels, steps, models, features, thresholds)
    print(f"\nwrote report -> {out}")

    # concise stdout summary (validation Score by method)
    metrics = pl.read_parquet(config.ARTIFACTS_DIR / "report" / "metrics.parquet")
    pl.Config.set_tbl_rows(80)
    print(metrics.filter(pl.col("method") != "baseline_keep_all")
          .pivot(values="score", index=["sym", "tau"], on="method", aggregate_function="first")
          .sort(["sym", "tau"]))


if __name__ == "__main__":
    main()
