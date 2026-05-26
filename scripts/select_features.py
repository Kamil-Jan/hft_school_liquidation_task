#!/usr/bin/env python
"""Derive per-(symbol, tau) feature lists **leak-free** and print a dict ready to paste
into ``config.FEATURE_SETS``.

The earlier version ranked permutation importance on the *validation* split and so
overfit it (validation improved, held-out test fell — a val-selection leak). This version
ranks importance on a **train-internal selection fold**: TRAIN is split by time into an
earlier *fit block* and a later *selection block* (with a max-τ embargo between); the model
is fit on the fit block and importance is permuted on the selection block. Validation **and**
test are never touched here — the real adoption gate is the walk-forward harness
(`python scripts/walk_forward.py --specs features` after populating ``FEATURE_SETS``).

Two estimators, deliberately different jobs:
  * **RANKER (MSE-HGBR).** ``fold_importance`` orders features by permutation importance
    using a plain MSE HistGBR. Ranking is order-only, so the stable/fast estimator is the
    right tool; switching it to the per-cell deployed estimators would only add noise.
  * **JUDGE (deployed estimator).** ``internal_score`` and the N-sweep refit with
    ``model.fit_model`` — the *shipped* per-(sym,τ) estimator (``config.MODEL_SPECS``) — so
    the chosen N is the N that helps the model we actually deploy (not MSE-HGBR).

Pipeline (the careful order):
  1. **Redundancy filter first** — per symbol, cluster features by |corr|>0.75 on the fit
     block; collapse each correlated block to one representative so importance isn't split
     across near-duplicates.
  2. **Rank + sweep N** — greedily take features by selection-fold permutation importance
     (one per cluster); for each candidate N score the curated set train-internally with the
     deployed estimator; ``pick_n`` keeps the smallest N at (near-)peak Score (a parsimony
     knee that *discovers* the BTC-few / ETH-many shape instead of hardcoding it).
  3. **Emit** — write ``feature_selection_sweep.parquet`` + ``feature_importance_rank.parquet``
     and print the paste-ready ``FEATURE_SETS`` dict.
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
from liqsignal.splits import TRAIN

CORR_THRESH = 0.75
TRAIN_INTERNAL_FRAC = 0.80   # earlier 80% of TRAIN = fit block; last 20% = selection block
N_GRID = (5, 8, 12, 15, 20, 25, 30, 40, 73)   # 73 ≈ "all (cluster-filtered)"; capped at #clusters

# Populated by load_panels() (kept module-level so the small helpers reference them, but
# loading is deferred out of import so pick_n/sweep_n stay importable without any I/O).
panels: dict[str, pl.DataFrame] = {}
steps: dict[str, int] = {}
FEATURES: list[str] = []


def load_panels() -> None:
    """Load each symbol's panel + sampling step and the shared feature column list."""
    global FEATURES
    for s in config.SYMBOLS:
        panels[s], steps[s] = analysis.load_panel(s)
    FEATURES = feature_columns(next(iter(panels.values())).columns)


def train_internal_blocks(sym: str, frac: float = TRAIN_INTERNAL_FRAC):
    """Split TRAIN by time into (fit_block, selection_block), embargoed at the cut.

    The fit block trains the model; the selection block (held-out *within train*) is where
    importance is permuted and the internal Score is read — so validation/test stay pristine.
    """
    tr = panels[sym].filter(pl.col("split") == TRAIN)
    cut = int(tr["timestamp"].quantile(frac))
    emb = config.SPLIT_EMBARGO_S * config.US
    fit_block = tr.filter(pl.col("timestamp") < cut - emb)
    sel_block = tr.filter(pl.col("timestamp") >= cut)
    return fit_block, sel_block


def median_impute(X):
    med = np.nanmedian(X, axis=0)
    return np.nan_to_num(np.where(np.isnan(X), med, X))


def cluster_map(fit_block: pl.DataFrame, thresh: float = CORR_THRESH) -> dict[str, int]:
    """feature -> cluster id; members of a cluster have |corr| > thresh (on the fit block)."""
    p = fit_block.sample(min(150_000, fit_block.height), seed=0)
    R = np.corrcoef(median_impute(p.select(FEATURES).to_numpy().astype(float)), rowvar=False)
    R = np.nan_to_num(R)
    Z = linkage(squareform(1 - np.abs(R), checks=False), method="average")
    cl = fcluster(Z, t=1 - thresh, criterion="distance")
    return dict(zip(FEATURES, cl))


def fold_importance(sym: str, tau: int, fit_block: pl.DataFrame, sel_block: pl.DataFrame,
                    n: int = 100_000, repeats: int = 8) -> pd.Series:
    """Permutation importance on the train-internal selection block (leak-free **ranking**).

    Uses a plain MSE HistGBR on purpose: this step only *orders* features, and the stable,
    fast MSE fit is the right ranker. The per-cell deployed estimator is used to *judge* N
    (``internal_score``), not to rank. The model trains on the fit block, then importance is
    measured on the unseen selection block — an honest within-train estimate.
    """
    mdl, feats = model.train_markout_model(fit_block, tau)
    p = sel_block.filter(pl.col(f"pnl_{tau}").is_finite()).sample(
        min(n, sel_block.filter(pl.col(f"pnl_{tau}").is_finite()).height), seed=0)
    imp = permutation_importance(mdl, p.select(feats).to_numpy().astype(float),
                                 p[f"pnl_{tau}"].to_numpy(), sample_weight=p["w"].to_numpy(),
                                 n_repeats=repeats, random_state=0, n_jobs=-1).importances_mean
    return pd.Series(imp, index=feats).sort_values(ascending=False)


def select(imp: pd.Series, n: int, cmap: dict[str, int]) -> list[str]:
    """Top-n features by importance, one per correlated cluster (redundancy-filtered)."""
    sel, used = [], set()
    for f in imp.index:
        if cmap[f] in used:
            continue                      # cluster already represented -> skip duplicate
        sel.append(f); used.add(cmap[f])
        if len(sel) >= n:
            break
    return sel


def internal_score(sym: str, tau: int, fit_block: pl.DataFrame, sel_block: pl.DataFrame,
                   feats: list[str]) -> float:
    """Train on the fit block with the **deployed** estimator, score on the selection block.

    Dispatches via ``model.fit_model`` (the same per-(sym,τ) estimator ``signal()`` ships),
    so the N we pick is tuned for the model we deploy — not the MSE-HGBR ranker. The
    threshold is the parameter-free ``score >= 0`` rule (estimator-agnostic; the proper
    purged-CV threshold is applied later by the walk-forward gate).
    """
    mdl, _kind, feats = model.fit_model(fit_block, tau, sym, features=feats)
    sc = model.predict_markout(mdl, sel_block, feats)
    f = analysis.expected_value_threshold(sc)
    return scoring.evaluate_filter(sel_block[f"pnl_{tau}"].to_numpy(), sel_block["w"].to_numpy(),
                                   f, n_days=sel_block["day"].n_unique(),
                                   turnover_scale=steps[sym]).score


def sweep_n(sym: str, tau: int, fit_block: pl.DataFrame, sel_block: pl.DataFrame,
            imp: pd.Series, cmap: dict[str, int], grid=N_GRID) -> list[dict]:
    """Leak-free N-sweep: for each candidate N take top-N (one-per-cluster) features and
    score them train-internally with the *deployed* estimator. Returns one row per distinct
    selected-set size: ``{N, n_selected, internal_score, selected_features}``."""
    rows, seen = [], set()
    for n in sorted({min(n, len(FEATURES)) for n in grid}):
        sel = select(imp, n, cmap)
        if len(sel) in seen:              # grid value above #clusters -> same set; skip dup
            continue
        seen.add(len(sel))
        rows.append({"N": n, "n_selected": len(sel),
                     "internal_score": internal_score(sym, tau, fit_block, sel_block, sel),
                     "selected_features": sel})
    return rows


def pick_n(rows: list[dict], *, tol: float = 0.02) -> dict:
    """Choose the smallest selected-set within ``tol`` bps of the best internal Score (knee
    rule → fewest features that capture the peak). Pure: depends only on ``rows``."""
    best = max(r["internal_score"] for r in rows)
    eligible = [r for r in rows if r["internal_score"] >= best - tol]
    return min(eligible, key=lambda r: r["n_selected"])


def main() -> None:
    load_panels()
    blocks = {s: train_internal_blocks(s) for s in config.SYMBOLS}
    cmaps = {s: cluster_map(blocks[s][0]) for s in config.SYMBOLS}
    for s in config.SYMBOLS:
        fb, sb = blocks[s]
        print(f"{s}: fit={fb.height:,} sel={sb.height:,} rows; "
              f"{len(FEATURES)} features -> {len(set(cmaps[s].values()))} clusters (|r|>{CORR_THRESH})")

    sweep_rows, rank_rows, sets = [], [], {}
    for sym in config.SYMBOLS:
        for tau in config.TAUS:
            fb, sb = blocks[sym]
            imp = fold_importance(sym, tau, fb, sb)
            rows = sweep_n(sym, tau, fb, sb, imp, cmaps[sym])
            all_score = internal_score(sym, tau, fb, sb, FEATURES)
            chosen = pick_n(rows)
            sets[(sym, tau)] = chosen["selected_features"]

            for rank, (feat, val) in enumerate(imp.items()):
                rank_rows.append({"sym": sym, "tau": tau, "feature": feat,
                                  "rank": rank, "importance": float(val)})
            for r in rows:
                sweep_rows.append({"sym": sym, "tau": tau, "N": r["N"],
                                   "n_selected": r["n_selected"],
                                   "internal_score": r["internal_score"],
                                   "internal_score_all": all_score,
                                   "chosen": r["n_selected"] == chosen["n_selected"],
                                   "selected_features": r["selected_features"]})

            print(f"\n{sym} t{tau:>3}: chosen N={chosen['n_selected']}  sel-block Score: "
                  f"sel={chosen['internal_score']:+.3f}  all={all_score:+.3f}  "
                  f"(sweep: " + " ".join(f"{r['n_selected']}={r['internal_score']:+.2f}" for r in rows) + ")")
            print("   " + ", ".join(chosen["selected_features"]))

    out = config.ensure_artifacts() / "report"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(sweep_rows, schema_overrides={"selected_features": pl.List(pl.Utf8)}
                 ).write_parquet(out / "feature_selection_sweep.parquet")
    pl.DataFrame(rank_rows).write_parquet(out / "feature_importance_rank.parquet")
    print(f"\nwrote -> {out / 'feature_selection_sweep.parquet'}, "
          f"{out / 'feature_importance_rank.parquet'}")

    print("\n# --- paste into config.py, then judge with `scripts/walk_forward.py --specs features` ---")
    print("FEATURE_SETS = {")
    for (sym, tau), fl in sets.items():
        print(f'    ("{sym}", {tau}): {fl!r},')
    print("}")


if __name__ == "__main__":
    main()
