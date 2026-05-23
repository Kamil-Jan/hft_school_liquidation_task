# Workflows & gotchas

## Environment
- **System Python 3.9** in `.venv` (no `uv`, no homebrew on this machine — don't assume them).
  Polars 1.36, PyArrow 21, pandas 2.3, numpy 2.0, scikit-learn 1.6.1, matplotlib 3.9.
- Install: `make install` (= `pip install -e ".[dev,notebook]"`). Editable; `import liqsignal` works anywhere.
- A Jupyter kernel named `python3` is registered into the venv (for notebook execution).

## Common workflows
```bash
make test                          # fast unit tests (spec math + features + thresholds)
make panel                         # rebuild artifacts/panel_<sym>.parquet after a feature change
make train                         # refit models + thresholds + report after a panel/model change
make study                         # single-feature conditional-markout study (reads panels)
make baselines                     # full-data PnL_all + turnover (reference; ~4 min)
make eda                           # rebuild + execute notebooks/01_exploration.ipynb
.venv/bin/python -m pytest         # tests directly
```
**Order from scratch:** `make install → make panel → make train`. `study`/`baselines`/`eda` are independent.

## Memory & performance (16 GB RAM — this matters)
- Trade files are 400–700M rows; **never `read_parquet` them whole**. Patterns that work:
  - `io.iter_trade_batches(sym, batch_size=20M)` — PyArrow batches → numpy (used by `baselines`).
  - `io.sample_trades(sym, target_rows)` — deterministic every-k-th-row sample (used by panels).
  - `signal._model_signal` — chunks trades, feature matrix is per-batch, output is 1 byte/trade.
- **Polars `join_asof` is not truly streaming and OOMs on ETH (706M).** That's why baselines
  use chunked numpy `searchsorted` forward-fill (`markout.forward_fill_mid`). Don't "simplify"
  it back to a single `join_asof`.
- Loading a full BBO (`io.load_book_top`) is ~2 GB and a few seconds — fine. Loading full BBO
  for both symbols at once is borderline; process one symbol at a time.
- Markout / windowed-liq / book-state are all vectorised `searchsorted` over sorted arrays —
  cheap. Per-minute / 1s grids via Polars group-by are seconds even on the big files.

## Reusable tricks
- **Forward-fill at arbitrary query times:** `markout.forward_fill_mid(bbo_ts, bbo_mid, query)`
  (or `last_index_at`) — `searchsorted(side='right')-1`, invalid before first / after last tick.
- **Windowed sums (liq pressure, trade flow):** prefix-sum + two `searchsorted` — see
  `features.windowed_liq`. This is how to add tape-flow features cheaply via a 1s grid.
- **Turnover on a sample:** pass `turnover_scale = step` to `scoring.evaluate_filter`
  (the sample is every-k-th row; sums must be rescaled, ratios are unbiased as-is).

## Gotchas / pitfalls
- **`NON_FEATURE_COLUMNS`** in `features.py` defines what the model treats as meta vs feature.
  A new panel column is auto-used as a feature unless added there. Keep label/meta columns listed.
- **`seconds_since_last` default is NaN** (not a giant sentinel) so means/plots aren't polluted;
  HistGBR treats NaN as missing — fine. Don't reintroduce 1e9.
- **`signal()` default loads models from `artifacts/`**; if absent it returns keep-all with a
  warning. Run `make train` first, or pass a `filter_fn`.
- **`fit_score_threshold` uses the passed `turnover_floor`** (a bug where it used the hardcoded
  config floor was fixed + tested — keep it that way).
- **Plots:** `report.py` forces the `Agg` backend; figures land in `artifacts/report/figs/`.
- **Don't commit `artifacts/` or `data/`** (gitignored) — they're large and reproducible.
- The repo is **not a git repo** (`git init` if you need version control).
- `description.md` is in Russian; the spec is mirrored in English in `config.py` docstrings and
  `.claude/docs/data-and-conventions.md`.
