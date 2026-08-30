.PHONY: help install dev test test-all lint fmt cov serve model train eval docker clean

VENV ?= ../.venv/bin
FULL_CKPT ?= ../vit_food_detection_model.pth
PY   ?= $(VENV)/python
# Dataset lives in the companion repo: github.com/Krishna-web-hub/PD-lab
# Clone it alongside this one, or point DATA at your copy.
DATA ?= ../PD-lab/dataset

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	$(PY) -m pip install -e .

dev:  ## Install with dev + llm extras
	$(PY) -m pip install -e ".[dev,llm]"

test:  ## Fast suite (no weights, no network)
	$(PY) -m pytest -m "not slow"

test-all:  ## Everything, including the real model and a short training run
	$(PY) -m pytest

cov:  ## Fast suite with a coverage report
	$(PY) -m pytest -m "not slow" --cov=freshsense --cov-report=term-missing

lint:  ## Ruff check
	$(PY) -m ruff check src tests

fmt:  ## Ruff format
	$(PY) -m ruff format src tests

serve:  ## Run the API on :8000
	$(PY) -m freshsense.cli serve --reload

model:  ## Export a head checkpoint from the notebook's full state dict
	$(PY) scripts/export_head.py $(FULL_CKPT) artifacts/vit_head.pt

train:  ## Fine-tune the head on the PD-lab dataset
	$(PY) -m freshsense.cli train \
		--train-dir $(DATA)/train/spoilage_detection \
		--val-dir $(DATA)/val/spoilage_detection

eval:  ## Evaluate on the held-out test split
	$(PY) scripts/evaluate.py --data-dir $(DATA)/test/spoilage_detection

docker:  ## Build the container image
	docker build -t freshsense:0.1.0 .

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
