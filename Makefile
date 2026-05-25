# liqsignal — common workflows. Assumes a local .venv (see `make venv`).
PY := .venv/bin/python

# Data directory the loaders read from (override for held-out test data):
#   make evaluate DATA_DIR=data_test
DATA_DIR ?= data

.PHONY: help venv install test baselines flowgrid panel study train eda evaluate feature-selection notebook clean

help:
	@echo "venv      create .venv"
	@echo "install   editable install of liqsignal + dev/notebook extras"
	@echo "test      run unit tests (spec math + features + thresholding)"
	@echo "baselines full-data PnL_all + turnover  -> artifacts/baselines.parquet"
	@echo "flowgrid  1s trade-flow grid            -> artifacts/flow_grid_<sym>.parquet"
	@echo "panel     sampled feature panels        -> artifacts/panel_<sym>.parquet"
	@echo "study     conditional-markout study + filter sweep (reads panels)"
	@echo "train     fit per-(sym,tau) models + thresholds, write report -> artifacts/report/"
	@echo "          (N_FEATURES=N prunes to top-N features per model)"
	@echo "eda       EDA precompute (aggregates + event studies) + build notebook"
	@echo "feature-selection  build+execute notebooks/02_feature_selection.ipynb (needs panels)"
	@echo "evaluate  run signal() on DATA_DIR and report Score/turnover per (sym,tau)"
	@echo "          (e.g. make evaluate DATA_DIR=data_test)"
	@echo "clean     remove build caches"

venv:
	python3 -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook]"

test:
	$(PY) -m pytest

baselines:
	$(PY) scripts/compute_baselines.py

flowgrid:
	$(PY) scripts/build_flow_grid.py

panel: flowgrid
	$(PY) scripts/build_panel.py

study:
	$(PY) scripts/run_study.py

train:
	$(PY) scripts/train_model.py $(if $(N_FEATURES),--n-features $(N_FEATURES))

evaluate:
	LIQSIGNAL_DATA_DIR=$(DATA_DIR) $(PY) scripts/evaluate.py

feature-selection:
	$(PY) scripts/build_feature_selection_nb.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 notebooks/02_feature_selection.ipynb

eda:
	$(PY) scripts/eda/precompute_aggregates.py
	$(PY) scripts/eda/precompute_events.py
	$(PY) scripts/eda/build_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 notebooks/01_exploration.ipynb

clean:
	rm -rf build *.egg-info src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
