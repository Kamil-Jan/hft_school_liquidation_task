"""Results reporting: metrics tables + plots for the trained filter.

Renders a self-contained markdown report (``artifacts/report/report.md``) plus PNG
figures comparing, per symbol and horizon:

* the keep-all baseline ``PnL_all``,
* the model + expected-value rule (filter predicted markout < 0),
* the model + Score-maximising swept cutoff, and
* the previous single-feature "keep top 10%" rule (reference),

all on the validation split, with turnover and kept-fraction. Figures: Score-vs-
kept-fraction curves (the threshold trade-off), predicted-vs-realised markout by
decile (calibration), per-month Score stability (regime risk), and permutation
feature importance.

Pure rendering — it takes already-fitted models/thresholds and panels carrying a
``score_<tau>`` column; training lives in ``scripts/train_model.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import analysis, config
from .splits import TRAIN, VAL

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
SYM_COLOR = {"btc": "#f2a900", "eth": "#627eea"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _score_on(df: pl.DataFrame, pnl_col: str, f: np.ndarray, step: int):
    return analysis.score_split(df, pnl_col, f, step)


def evaluate(panels: dict[str, pl.DataFrame], steps: dict[str, int],
             thresholds: dict[int, float]) -> pl.DataFrame:
    """Per (symbol, tau, method) validation metrics as a tidy frame."""
    rows = []
    for sym, panel in panels.items():
        step = steps[sym]
        tr = panel.filter(pl.col("split") == TRAIN)
        va = panel.filter(pl.col("split") == VAL)
        for tau in config.TAUS:
            pnl = f"pnl_{tau}"; sc = f"score_{tau}"
            va_score = va[sc].to_numpy()
            methods = {
                "baseline_keep_all": np.zeros(va.height, np.int8),
                "model_expected_value": analysis.expected_value_threshold(va_score),
                "model_score_max": analysis.apply_threshold(va_score, thresholds[tau]),
            }
            # reference: previous single-feature keep-top-10% (fit on train ret_5s_signed)
            direction, thr = analysis.fit_keep_best(tr, "ret_5s_signed", pnl, 0.10)
            methods["ref_keep10pct_ret5s"] = analysis.apply_keep_best(va, "ret_5s_signed", direction, thr)

            for method, f in methods.items():
                r = _score_on(va, pnl, f, step)
                rows.append(dict(sym=sym, tau=tau, method=method,
                                 score=round(r.score, 4), pnl_kept=round(r.pnl_kept, 4),
                                 pnl_all=round(r.pnl_all, 4),
                                 keep_frac=round(1 - r.frac_filtered_n, 4),
                                 turnover_per_day=round(r.kept_turnover_per_day, 0),
                                 constraint_ok=r.constraint_ok))
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def fig_threshold_curves(panels, steps, thresholds, path: Path) -> None:
    fig, axes = plt.subplots(len(config.TAUS), len(panels), figsize=(5.2 * len(panels), 9), squeeze=False)
    for j, (sym, panel) in enumerate(panels.items()):
        va = panel.filter(pl.col("split") == VAL)
        for i, tau in enumerate(config.TAUS):
            ax = axes[i][j]
            sc = va[f"score_{tau}"].to_numpy()
            grid = np.quantile(sc[np.isfinite(sc)], np.linspace(0, 0.98, 50))
            keep_fracs, scores = [], []
            for thr in grid:
                f = analysis.apply_threshold(sc, thr)
                r = _score_on(va, f"pnl_{tau}", f, steps[sym])
                keep_fracs.append(1 - r.frac_filtered_n); scores.append(r.score)
            ax.plot(keep_fracs, scores, color=SYM_COLOR[sym], lw=1.5)
            # operating points
            f_ev = analysis.expected_value_threshold(sc)
            r_ev = _score_on(va, f"pnl_{tau}", f_ev, steps[sym])
            ax.scatter([1 - r_ev.frac_filtered_n], [r_ev.score], c="green", zorder=5, label="expected-value")
            f_sm = analysis.apply_threshold(sc, thresholds[tau])
            r_sm = _score_on(va, f"pnl_{tau}", f_sm, steps[sym])
            ax.scatter([1 - r_sm.frac_filtered_n], [r_sm.score], c="red", marker="*", s=120,
                       zorder=5, label="score-max")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{sym.upper()}  τ={tau}s")
            if j == 0:
                ax.set_ylabel("val Score (bps)")
            if i == len(config.TAUS) - 1:
                ax.set_xlabel("kept fraction")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Validation Score vs kept fraction (curve = sweep; markers = chosen cutoffs)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_pred_vs_realized(panels, path: Path, n_deciles: int = 10) -> None:
    fig, axes = plt.subplots(1, len(config.TAUS), figsize=(5 * len(config.TAUS), 4), squeeze=False)
    for i, tau in enumerate(config.TAUS):
        ax = axes[0][i]
        for sym, panel in panels.items():
            va = panel.filter((pl.col("split") == VAL) & pl.col(f"pnl_{tau}").is_finite()
                              & pl.col(f"score_{tau}").is_finite())
            d = va.with_columns(pl.col(f"score_{tau}").qcut(n_deciles, labels=[str(k) for k in range(n_deciles)],
                                                            allow_duplicates=True).alias("dec"))
            g = (d.group_by("dec").agg(
                    pred=pl.col(f"score_{tau}").mean(),
                    real=(pl.col(f"pnl_{tau}") * pl.col("w")).sum() / pl.col("w").sum()).sort("dec"))
            ax.plot(g["pred"], g["real"], "o-", color=SYM_COLOR[sym], ms=4, label=sym)
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"τ={tau}s"); ax.set_xlabel("predicted markout (bps)")
        if i == 0:
            ax.set_ylabel("realised w-mean markout (bps)"); ax.legend()
    fig.suptitle("Predicted vs realised markout by score decile (validation)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_monthly_stability(panels, steps, path: Path) -> None:
    """Per-month validation/train Score under the expected-value rule (regime check)."""
    fig, axes = plt.subplots(1, len(config.TAUS), figsize=(5 * len(config.TAUS), 4), squeeze=False)
    for i, tau in enumerate(config.TAUS):
        ax = axes[0][i]
        labels_all = []
        for sym, panel in panels.items():
            p = panel.with_columns(
                month=pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.strftime("%Y-%m"))
            months = sorted(p["month"].unique().to_list())
            vals = []
            for m in months:
                sub = p.filter(pl.col("month") == m)
                f = analysis.expected_value_threshold(sub[f"score_{tau}"].to_numpy())
                vals.append(_score_on(sub, f"pnl_{tau}", f, steps[sym]).score)
            x = np.arange(len(months)) + (0.0 if sym == "btc" else 0.4)
            ax.bar(x, vals, width=0.38, color=SYM_COLOR[sym], label=sym)
            labels_all = months
        ax.set_xticks(np.arange(len(labels_all)) + 0.2); ax.set_xticklabels(labels_all, rotation=45, fontsize=8)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"τ={tau}s")
        if i == 0:
            ax.set_ylabel("Score (bps), expected-value rule"); ax.legend()
    fig.suptitle("Per-month Score (train months Dec–Jan, val month Feb) — regime stability", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_feature_importance(model, panel: pl.DataFrame, features: list[str], tau: int,
                           path: Path, n_sample: int = 60_000, top: int = 18) -> None:
    from sklearn.inspection import permutation_importance
    va = panel.filter((pl.col("split") == VAL) & pl.col(f"pnl_{tau}").is_finite())
    va = va.sample(min(n_sample, va.height), seed=0)
    X = va.select(features).to_numpy().astype(np.float64)
    y = va[f"pnl_{tau}"].to_numpy()
    w = va["w"].to_numpy()
    imp = permutation_importance(model, X, y, sample_weight=w, n_repeats=4, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[-top:]
    fig, ax = plt.subplots(figsize=(7, 0.36 * top + 1))
    ax.barh([features[k] for k in order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#4c72b0")
    ax.set_title(f"Permutation feature importance (τ={tau}s, validation)")
    ax.set_xlabel("mean drop in R² when shuffled")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate(panels: dict[str, pl.DataFrame], steps: dict[str, int],
             models: dict[int, object], features: list[str],
             thresholds: dict[int, float], outdir: Path | None = None) -> Path:
    outdir = outdir or (config.ARTIFACTS_DIR / "report")
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(panels, steps, thresholds)
    metrics.write_parquet(outdir / "metrics.parquet")

    fig_threshold_curves(panels, steps, thresholds, figs / "threshold_curves.png")
    fig_pred_vs_realized(panels, figs / "pred_vs_realized.png")
    fig_monthly_stability(panels, steps, figs / "monthly_stability.png")
    importance_tau = 120 if 120 in models else config.TAUS[0]
    fig_feature_importance(models[importance_tau], next(iter(panels.values())),
                           features, importance_tau, figs / "feature_importance.png")

    _write_markdown(metrics, thresholds, importance_tau, outdir)
    return outdir / "report.md"


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.iter_rows():
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def _write_markdown(metrics: pl.DataFrame, thresholds, importance_tau, outdir: Path) -> None:
    lines = ["# Liquidation-filter results report", "",
             "Validation-split metrics per symbol & horizon (Score = PnL_kept − PnL_all; "
             "turnover floor 500k USD/day).", ""]
    pivot = (metrics.filter(pl.col("method") != "baseline_keep_all")
             .pivot(values="score", index=["sym", "tau"], on="method", aggregate_function="first")
             .sort(["sym", "tau"]))
    lines += ["## Score(τ) by method (validation, bps)", "", _md_table(pivot), ""]
    lines += ["## Full metrics", "", _md_table(metrics.sort(["sym", "tau", "method"])), ""]
    lines += ["## Fitted Score-maximising thresholds (predicted markout, bps)", "",
              _md_table(pl.DataFrame({"tau": list(thresholds), "threshold_bps": list(thresholds.values())})), ""]
    lines += ["## Figures", "",
              "![Score vs kept fraction](figs/threshold_curves.png)", "",
              "![Predicted vs realised markout](figs/pred_vs_realized.png)", "",
              "![Per-month stability](figs/monthly_stability.png)", "",
              f"![Feature importance τ={importance_tau}s](figs/feature_importance.png)", ""]
    (outdir / "report.md").write_text("\n".join(lines))
