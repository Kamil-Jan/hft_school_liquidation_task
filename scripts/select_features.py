#!/usr/bin/env python
"""Derive per-(symbol, tau) feature lists from the feature-selection study (notebook 02)
and print a dict ready to paste into ``config.FEATURE_SETS``.

Pipeline (the careful order):
  1. **Redundancy filter first** — per symbol, cluster features by |corr|>0.75 (same as
     notebook §3); collapse each correlated block to a single representative so importance
     isn't split across near-duplicates.
  2. **Rank + select** — greedily take features by stabilised validation permutation
     importance (8 repeats), skipping any whose cluster is already represented, up to N
     (N = the sweep-optimal count per model).
  3. **Verify** — report each curated set's validation Score vs the all-73 baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance

from liqsignal import analysis, config, model, scoring
from liqsignal.features import feature_columns
from liqsignal.splits import TRAIN, VAL

NMAP = {("btc", 30): 5, ("btc", 120): 10, ("btc", 300): 15,
        ("eth", 30): 40, ("eth", 120): 40, ("eth", 300): 25}
CORR_THRESH = 0.75

panels, steps = {}, {}
for s in config.SYMBOLS:
    panels[s], steps[s] = analysis.load_panel(s)
FEATURES = feature_columns(next(iter(panels.values())).columns)


def median_impute(X):
    med = np.nanmedian(X, axis=0)
    return np.nan_to_num(np.where(np.isnan(X), med, X))


def cluster_map(sym: str, thresh: float = CORR_THRESH) -> dict[str, int]:
    """feature -> cluster id; members of a cluster have |corr| > thresh (per symbol)."""
    p = panels[sym].filter(pl.col("split") == TRAIN).sample(150_000, seed=0)
    R = np.corrcoef(median_impute(p.select(FEATURES).to_numpy().astype(float)), rowvar=False)
    R = np.nan_to_num(R)
    Z = linkage(squareform(1 - np.abs(R), checks=False), method="average")
    cl = fcluster(Z, t=1 - thresh, criterion="distance")
    return dict(zip(FEATURES, cl))


def val_importance(sym: str, tau: int, n: int = 100_000, repeats: int = 8) -> pd.Series:
    mdl, feats = model.load(tau, sym)
    p = panels[sym].filter((pl.col("split") == VAL) & pl.col(f"pnl_{tau}").is_finite()).sample(
        min(n, panels[sym].filter(pl.col("split") == VAL).height), seed=0)
    imp = permutation_importance(mdl, p.select(feats).to_numpy().astype(float),
                                 p[f"pnl_{tau}"].to_numpy(), sample_weight=p["w"].to_numpy(),
                                 n_repeats=repeats, random_state=0, n_jobs=-1).importances_mean
    return pd.Series(imp, index=feats).sort_values(ascending=False)


def select(sym: str, tau: int, n: int, cmap: dict[str, int]) -> list[str]:
    """Top-n features by val importance, one per correlated cluster (redundancy-filtered)."""
    imp = val_importance(sym, tau)
    sel, used = [], set()
    for f in imp.index:
        if cmap[f] in used:
            continue                      # cluster already represented -> skip duplicate
        sel.append(f); used.add(cmap[f])
        if len(sel) >= n:
            break
    return sel


def val_score(sym: str, tau: int, feats: list[str]) -> float:
    tr = panels[sym].filter((pl.col("split") == TRAIN) & pl.col(f"pnl_{tau}").is_finite()).sample(
        min(500_000, panels[sym].filter(pl.col("split") == TRAIN).height), seed=0)
    mdl, feats = model.train_markout_model(tr, tau, features=feats)
    va = panels[sym].filter(pl.col("split") == VAL)
    sc = model.predict_markout(mdl, va, feats)
    f = analysis.expected_value_threshold(sc)
    return scoring.evaluate_filter(va[f"pnl_{tau}"].to_numpy(), va["w"].to_numpy(), f,
                                   n_days=va["day"].n_unique(), turnover_scale=steps[sym]).score


def main() -> None:
    cmaps = {s: cluster_map(s) for s in config.SYMBOLS}
    for s in config.SYMBOLS:
        ncl = len(set(cmaps[s].values()))
        print(f"{s}: {len(FEATURES)} features -> {ncl} clusters (|r|>{CORR_THRESH})")
    sets = {}
    for (sym, tau), n in NMAP.items():
        sel = select(sym, tau, n, cmaps[sym])
        sets[(sym, tau)] = sel
        print(f"\n{sym} t{tau:>3}: N={len(sel)}  valScore sel={val_score(sym, tau, sel):+.3f}  "
              f"all73={val_score(sym, tau, FEATURES):+.3f}")
        print("   " + ", ".join(sel))
    print("\n# --- paste into config.py ---")
    print("FEATURE_SETS = {")
    for (sym, tau), fl in sets.items():
        print(f'    ("{sym}", {tau}): {fl!r},')
    print("}")


if __name__ == "__main__":
    main()
