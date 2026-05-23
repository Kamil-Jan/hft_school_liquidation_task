# liqsignal — common workflows. Assumes a local .venv (see `make venv`).
PY := .venv/bin/python

# Data directory the loaders read from (override for held-out test data):
#   make evaluate DATA_DIR=data_test
DATA_DIR ?= data

.PHONY: help venv install test baselines panel study train eda evaluate notebook clean

help:
	@echo "venv      create .venv"
	@echo "install   editable install of liqsignal + dev/notebook extras"
	@echo "test      run unit tests (spec math + features + thresholding)"
	@echo "baselines full-data PnL_all + turnover  -> artifacts/baselines.parquet"
	@echo "panel     sampled feature panels        -> artifacts/panel_<sym>.parquet"
	@echo "study     conditional-markout study + filter sweep (reads panels)"
	@echo "train     fit per-tau models + thresholds, write report -> artifacts/report/"
	@echo "eda       EDA precompute (aggregates + event studies) + build notebook"
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

panel:
	$(PY) scripts/build_panel.py

study:
	$(PY) scripts/run_study.py

train:
	$(PY) scripts/train_model.py

evaluate:
	LIQSIGNAL_DATA_DIR=$(DATA_DIR) $(PY) scripts/evaluate.py

eda:
	$(PY) scripts/eda/precompute_aggregates.py
	$(PY) scripts/eda/precompute_events.py
	$(PY) scripts/eda/build_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 notebooks/01_exploration.ipynb

clean:
	rm -rf build *.egg-info src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
