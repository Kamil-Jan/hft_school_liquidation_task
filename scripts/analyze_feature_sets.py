#!/usr/bin/env python
"""Explain *why* the selected features help, per (symbol, tau) — a read-only diagnostic.

For each feature in ``config.FEATURE_SETS[(sym, tau)]`` (or, when a cell kept all features,
its top-``TOP_K`` by leak-free importance rank) reports three evidence axes:
  * **rank / importance** — from ``feature_importance_rank.parquet`` (the MSE-HGBR ranker,
    permuted on the train-internal selection block).
  * **train edge** — ``analysis.conditional_markout`` top-minus-bottom-quintile w-markout on
    the TRAIN split (bps) + its sign + a monotonicity trend (does PnL stratify with the
    feature?). This is the single-feature edge the model leans on.
  * **regime survival** — how many calendar months the per-month top-minus-bottom edge keeps
    the TRAIN sign (mirrors ``report.regime_by_month``): a feature whose edge survives more
    months is a more regime-robust reason to keep it.

Writes ``artifacts/report/feature_explanations.parquet`` and prints per-cell tables.
Run after ``select_features.py`` (needs the rank artifact) and after staging the adopted
``config.FEATURE_SETS``.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from liqsignal import analysis, config
from liqsignal.features import feature_columns
from liqsignal.splits import TRAIN

TOP_K = 12   # features to explain when neither a curated set nor a sweep pick exists
RANK_PATH = config.ARTIFACTS_DIR / "report" / "feature_importance_rank.parquet"
SWEEP_PATH = config.ARTIFACTS_DIR / "report" / "feature_selection_sweep.parquet"


def _edge(df: pl.DataFrame, feature: str, pnl: str) -> tuple[float, float]:
    """(top−bottom quintile w-markout, monotone trend) of ``feature`` on ``df``."""
    cm = analysis.conditional_markout(df, feature, pnl, 5)
    d = dict(zip(cm["bucket"].to_list(), cm["wpnl"].to_list()))
    wp = np.array([d.get(str(i), np.nan) for i in range(5)])
    edge = d.get("4", np.nan) - d.get("0", np.nan)
    trend = float(np.corrcoef(np.arange(5), wp)[0, 1]) if np.isfinite(wp).all() else float("nan")
    return float(edge), trend


def feature_month_survival(panel: pl.DataFrame, feature: str, tau: int, ref_sign: float):
    """Count calendar months whose top−bottom edge keeps ``ref_sign`` (the train sign)."""
    pnl = f"pnl_{tau}"
    p = panel.with_columns(month=pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.strftime("%Y-%m"))
    months = sorted(p["month"].unique().to_list())
    consistent = 0
    for m in months:
        edge, _ = _edge(p.filter(pl.col("month") == m), feature, pnl)
        if np.isfinite(edge) and np.sign(edge) == ref_sign:
            consistent += 1
    return consistent, len(months)


def explain_cell(sym: str, tau: int, panel: pl.DataFrame, feats: list[str],
                 rank_df: pl.DataFrame, curated: bool) -> pl.DataFrame:
    """One row per explained feature with rank, train edge, and regime survival."""
    pnl = f"pnl_{tau}"
    tr = panel.filter(pl.col("split") == TRAIN)
    ranks = {r["feature"]: (r["rank"], r["importance"])
             for r in rank_df.filter((pl.col("sym") == sym) & (pl.col("tau") == tau)).iter_rows(named=True)}
    rows = []
    for f in feats:
        edge, trend = _edge(tr, f, pnl)
        sign = float(np.sign(edge)) if np.isfinite(edge) else 0.0
        cons, n_months = feature_month_survival(panel, f, tau, sign)
        rk, imp = ranks.get(f, (-1, float("nan")))
        rows.append({"sym": sym, "tau": tau, "feature": f, "curated": curated,
                     "imp_rank": rk, "importance": round(imp, 6),
                     "train_edge_bps": round(edge, 4), "edge_sign": sign,
                     "trend": round(trend, 3), "months_consistent": cons, "n_months": n_months})
    return pl.DataFrame(rows).sort("imp_rank")


def main() -> None:
    if not RANK_PATH.exists():
        raise SystemExit(f"missing {RANK_PATH} — run `python scripts/select_features.py` first")
    rank_df = pl.read_parquet(RANK_PATH)
    sweep = pl.read_parquet(SWEEP_PATH) if SWEEP_PATH.exists() else None

    panels, steps = {}, {}
    for sym in config.SYMBOLS:
        panels[sym], steps[sym] = analysis.load_panel(sym)
    all_feats = feature_columns(next(iter(panels.values())).columns)

    def chosen_set(sym: str, tau: int):
        """The leak-free chosen set: the sweep's chosen row (explained even when the cell
        ships all-73), falling back to top-K by importance rank if the sweep is absent."""
        if sweep is not None:
            ch = sweep.filter((pl.col("sym") == sym) & (pl.col("tau") == tau) & pl.col("chosen"))
            if ch.height:
                return list(ch["selected_features"][0])
        return (rank_df.filter((pl.col("sym") == sym) & (pl.col("tau") == tau))
                .sort("rank").head(TOP_K)["feature"].to_list()) or all_feats[:TOP_K]

    pl.Config.set_tbl_rows(60)
    frames = []
    for sym in config.SYMBOLS:
        for tau in config.TAUS:
            adopted = (sym, tau) in config.FEATURE_SETS
            feats = config.FEATURE_SETS[(sym, tau)] if adopted else chosen_set(sym, tau)
            tbl = explain_cell(sym, tau, panels[sym], feats, rank_df, adopted)
            frames.append(tbl)
            tag = (f"adopted curated N={len(feats)}" if adopted
                   else f"leak-free chosen N={len(feats)} (ships all-73)")
            print(f"\n=== {sym} τ{tau} — {tag} ===")
            print(tbl.select(["feature", "imp_rank", "train_edge_bps", "edge_sign",
                              "trend", "months_consistent", "n_months"]))

    outdir = config.ensure_artifacts() / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(outdir / "feature_explanations.parquet")
    print(f"\nwrote -> {outdir / 'feature_explanations.parquet'}")


if __name__ == "__main__":
    main()
