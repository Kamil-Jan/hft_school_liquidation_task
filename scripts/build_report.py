#!/usr/bin/env python
"""Regenerate artifacts/report/ from the already-trained per-(symbol, tau) models.

Loads the persisted models/thresholds (no refitting), recomputes the panels' ``score_<tau>``
columns, and re-renders the report — used to refresh figures/markdown after a `report.py`
change without paying for a full `make train`.
"""
from __future__ import annotations

import os
import warnings

# Silence LightGBM's nameless-array warning here and in joblib/loky workers (see train_model.py).
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import polars as pl

from liqsignal import analysis, config, model, report


def main() -> None:
    panels, steps, models, feats_by, thresholds = {}, {}, {}, {}, {}
    for sym in config.SYMBOLS:
        panels[sym], steps[sym] = analysis.load_panel(sym)
    for sym in config.SYMBOLS:
        for tau in config.TAUS:
            mdl, feats = model.load(tau, sym)
            if mdl is None:
                raise SystemExit(f"no trained model for ({sym}, {tau}) — run `make train` first")
            models[(sym, tau)], feats_by[(sym, tau)] = mdl, feats
            thresholds[(sym, tau)] = model.load_threshold(tau, sym)
            panels[sym] = panels[sym].with_columns(
                pl.Series(f"score_{tau}", model.predict_markout(mdl, panels[sym], feats)))

    out = report.generate(panels, steps, models, feats_by, thresholds)
    print(f"wrote report -> {out}")


if __name__ == "__main__":
    main()
