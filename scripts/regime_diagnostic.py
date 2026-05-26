#!/usr/bin/env python
"""Per-month regime diagnostic: how the liquidation edge, markout and vol shift across months.

Standalone runner for `report.regime_by_month` / `report.fig_regime_by_month` so the
regime picture can be inspected without retraining (it needs only the panels, no model).
`make train` regenerates the same artifacts inside the full report.

Writes `artifacts/report/regime_by_month.parquet` and `figs/regime_by_month.png`.
"""
from __future__ import annotations

import polars as pl

from liqsignal import analysis, config, report


def main() -> None:
    panels, steps = {}, {}
    for sym in config.SYMBOLS:
        panels[sym], steps[sym] = analysis.load_panel(sym)

    outdir = config.ensure_artifacts() / "report"
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    table = report.regime_by_month(panels, steps)
    table.write_parquet(outdir / "regime_by_month.parquet")
    report.fig_regime_by_month(table, outdir / "figs" / "regime_by_month.png")

    pl.Config.set_tbl_rows(50)
    print(table.sort(["sym", "month"]))
    print(f"\nwrote -> {outdir / 'regime_by_month.parquet'} and figs/regime_by_month.png")


if __name__ == "__main__":
    main()
