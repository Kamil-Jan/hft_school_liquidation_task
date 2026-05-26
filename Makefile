# liqsignal — common workflows. Assumes a local .venv (see `make venv`).
PY := .venv/bin/python

# Data directory the loaders read from (override for held-out test data):
#   make evaluate DATA_DIR=data_test
DATA_DIR ?= data

# Spec set for the walk-forward judge: baseline | regime | objective | features | shipped
#   make walkforward WF_SPECS=features
WF_SPECS ?= baseline

.PHONY: help venv install test baselines flowgrid panel train walkforward regime report eda evaluate feature-select feature-explain feature-nb clean

help:
	@echo "Setup"
	@echo "  venv        create .venv"
	@echo "  install     editable install of liqsignal + dev/notebook extras (+ patch_lightgbm)"
	@echo "  test        run unit tests"
	@echo ""
	@echo "Data"
	@echo "  baselines   full-data PnL_all + turnover/day  -> artifacts/baselines.parquet"
	@echo "  flowgrid    1s trade-flow grid                -> artifacts/flow_grid_<sym>.parquet"
	@echo "  panel       sampled feature panels (+flowgrid) -> artifacts/panel_<sym>.parquet"
	@echo ""
	@echo "Model"
	@echo "  train       fit per-(sym,tau) models + thresholds + report  (N_FEATURES=N to prune)"
	@echo "  walkforward expanding-window OOS backtest  (WF_SPECS=baseline|regime|objective|features|shipped)"
	@echo "  report      regenerate artifacts/report/ from trained models (no refit)"
	@echo ""
	@echo "Diagnostics"
	@echo "  regime          per-month edge/markout/vol -> artifacts/report/regime_by_month.parquet"
	@echo "  feature-select  leak-free N-sweep -> feature_selection_sweep.parquet + FEATURE_SETS dict"
	@echo "  feature-explain why the chosen features help -> feature_explanations.parquet"
	@echo "  eda             EDA precompute + build/execute notebooks/01_exploration.ipynb"
	@echo "  feature-nb      build/execute notebooks/02_feature_selection.ipynb (needs panels)"
	@echo ""
	@echo "Review"
	@echo "  evaluate    run signal() on DATA_DIR, report Score/turnover per (sym,tau)  (DATA_DIR=data_test)"
	@echo "  clean       remove build caches"

venv:
	python3 -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook]"
	$(PY) scripts/patch_lightgbm.py   # macOS: point lightgbm at sklearn's vendored libomp

test:
	$(PY) -m pytest

baselines:
	$(PY) scripts/compute_baselines.py

flowgrid:
	$(PY) scripts/build_flow_grid.py

panel: flowgrid
	$(PY) scripts/build_panel.py

train:
	$(PY) scripts/train_model.py $(if $(N_FEATURES),--n-features $(N_FEATURES))

walkforward:
	$(PY) scripts/walk_forward.py --specs $(WF_SPECS)

feature-select:
	$(PY) scripts/select_features.py

feature-explain:
	$(PY) scripts/analyze_feature_sets.py

regime:
	$(PY) scripts/regime_diagnostic.py

report:
	$(PY) scripts/build_report.py

evaluate:
	LIQSIGNAL_DATA_DIR=$(DATA_DIR) $(PY) scripts/evaluate.py

feature-nb:
	$(PY) scripts/build_feature_selection_nb.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 notebooks/02_feature_selection.ipynb

eda:
	$(PY) scripts/eda/precompute_aggregates.py
	$(PY) scripts/eda/precompute_events.py
	$(PY) scripts/eda/precompute_cascades.py
	$(PY) scripts/eda/build_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 notebooks/01_exploration.ipynb

clean:
	rm -rf build *.egg-info src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
