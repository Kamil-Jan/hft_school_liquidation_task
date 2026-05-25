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


def _top_features(mdl, panels, features: list[str], tau: int, n_keep: int,
                  n_sample: int = 60_000) -> list[str]:
    """Top-``n_keep`` features by permutation importance on the validation sample.

    HistGBR has no native importances, so we permute on held-out validation rows
    (the same machinery the report uses) and keep the highest-impact columns.
    """
    from sklearn.inspection import permutation_importance
    va = pl.concat([p.filter((pl.col("split") == VAL) & pl.col(f"pnl_{tau}").is_finite())
                    for p in panels.values()])
    va = va.sample(min(n_sample, va.height), seed=0)
    X = va.select(features).to_numpy().astype(np.float64)
    imp = permutation_importance(mdl, X, va[f"pnl_{tau}"].to_numpy(),
                                 sample_weight=va["w"].to_numpy(),
                                 n_repeats=3, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:n_keep]
    return [features[i] for i in sorted(order)]


def _fit_tau(train: pl.DataFrame, tau: int, features: list[str], args,
             panel_for_imp: pl.DataFrame, step: float, symbol: str):
    """Fit one (symbol, horizon) on ``train`` (optionally prune + refit), fit the
    purged-CV threshold, persist, and return ``(model, feats, threshold, cv)``."""
    mdl, feats = model.train_markout_model(train, tau, features=features)
    if args.n_features is not None and 0 < args.n_features < len(features):
        feats = _top_features(mdl, {symbol: panel_for_imp}, features, tau, args.n_features)
        mdl, feats = model.train_markout_model(train, tau, features=feats)
    ptr = train.with_columns(pl.Series(f"score_{tau}", model.predict_markout(mdl, train, feats)))
    thr, cv = analysis.fit_score_threshold(
        ptr[f"score_{tau}"].to_numpy(), ptr[f"pnl_{tau}"].to_numpy(),
        ptr["w"].to_numpy(), ptr["timestamp"].to_numpy(),
        step=step, min_keep_frac=args.min_keep_frac)
    model.save(mdl, feats, tau, symbol, threshold=thr)
    return mdl, feats, thr, cv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--min-keep-frac", type=float, default=0.05)
    p.add_argument("--n-features", type=int, default=config.N_FEATURES,
                   help="keep top-N features (permutation importance) per (sym,tau); default keep all")
    args = p.parse_args()

    panels, steps = {}, {}
    for sym in args.symbols:
        panel, step = analysis.load_panel(sym)
        panels[sym], steps[sym] = panel, step
    features = feature_columns(next(iter(panels.values())).columns)
    prune = args.n_features is not None and 0 < args.n_features < len(features)
    use_sets = (not prune) and bool(config.FEATURE_SETS)
    mode = (f"pruning to top-{args.n_features}" if prune
            else "curated config.FEATURE_SETS" if use_sets else "all features")
    print(f"{len(features)} features; symbols={args.symbols}; per-symbol models; {mode}")

    # one model per (symbol, tau). Feature set precedence: --n-features > FEATURE_SETS > all.
    models: dict = {}
    feats_by: dict = {}
    thresholds: dict = {}
    for tau in config.TAUS:
        for sym in args.symbols:
            tr = panels[sym].filter(pl.col("split") == TRAIN)
            feat_in = config.FEATURE_SETS.get((sym, tau), features) if use_sets else features
            mdl, feats, thr, cv = _fit_tau(tr, tau, feat_in, args, panels[sym], steps[sym], sym)
            panels[sym] = panels[sym].with_columns(
                pl.Series(f"score_{tau}", model.predict_markout(mdl, panels[sym], feats)))
            models[(sym, tau)], feats_by[(sym, tau)], thresholds[(sym, tau)] = mdl, feats, thr
            print(f"  {sym} tau={tau}: {len(feats)} feats, thr={thr:+.4f} bps (CV Score={cv:+.4f})")

    out = report.generate(panels, steps, models, feats_by, thresholds)
    print(f"\nwrote report -> {out}")

    # concise stdout summary (held-out Score by method, per split)
    metrics = pl.read_parquet(config.ARTIFACTS_DIR / "report" / "metrics.parquet")
    pl.Config.set_tbl_rows(80)
    print(metrics.filter(pl.col("method") != "baseline_keep_all")
          .pivot(values="score", index=["sym", "split", "tau"], on="method", aggregate_function="first")
          .sort(["sym", "split", "tau"]))


if __name__ == "__main__":
    main()
