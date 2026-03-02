# SecureRelayFL — reproducible workflow
# ======================================
# Quick start:
#   make env          # create venv + install deps
#   make data         # generate synthetic fault waveforms
#   make baseline     # train centralized baseline
#   make experiments  # run all FL experiments
#   make figures      # generate publication figures
#   make all          # everything end-to-end

SHELL   := /bin/bash
PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
SEED    := 42
N_SAMPLES := 1000

# ---- environment ----

.PHONY: env env-exact env-dev clean

env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo ""
	@echo "✓ Environment ready.  Activate with:  source $(VENV)/bin/activate"

env-exact:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.lock
	$(PIP) install -e .
	@echo ""
	@echo "✓ Environment ready (exact pins).  Activate with:  source $(VENV)/bin/activate"

env-dev: env
	$(PIP) install -e ".[dev]"
	@echo "✓ Dev dependencies installed."

# ---- data ----

.PHONY: data data-clean

data:
	$(PY) data/generator/generate.py --n-samples $(N_SAMPLES) --seed $(SEED)
	@echo "✓ Generated $(N_SAMPLES) samples/facility in data/generated/"

data-clean:
	rm -rf data/generated/
	@echo "✓ Cleaned generated data."

# ---- training (placeholders — will be filled as modules are built) ----

.PHONY: baseline experiments figures

baseline:
	$(PY) -m models.train_centralized --seed $(SEED)

experiments:
	$(PY) -m experiments.run_all --seed $(SEED)

figures:
	$(PY) -m analysis.generate_figures

# ---- full pipeline ----

.PHONY: all

all: env data baseline experiments figures
	@echo "✓ Full pipeline complete."

# ---- housekeeping ----

.PHONY: clean clean-all lint test

clean:
	rm -rf data/generated/ results/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf $(VENV) *.egg-info

lint:
	$(VENV)/bin/ruff check .

test:
	$(VENV)/bin/pytest tests/
