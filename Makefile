PYTHON := uv run python
SRC := src

.PHONY: help setup preprocess analyze train eval report clean

help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies (uv sync)"
	@echo "  make preprocess   - Data preprocessing"
	@echo "  make analyze      - Analyze model parameters & size"
	@echo "  make train        - Train model (with compression)"
	@echo "  make eval         - Evaluate accuracy / precision"
	@echo "  make report       - Generate comparison report"
	@echo "  make clean        - Clean outputs & logs"

# Environment
setup:
	uv sync

# Pipeline
preprocess:
	$(PYTHON) $(SRC)/data/preprocess.py --config-name=config

analyze:
	$(PYTHON) $(SRC)/compression/analyze.py --config-name=config

train:
	$(PYTHON) $(SRC)/models/train.py --config-name=config

eval:
	$(PYTHON) $(SRC)/evaluation/evaluate.py --config-name=config

report:
	$(PYTHON) $(SRC)/compression/report.py --config-name=config

# Utils
clean:
	rm -rf outputs mlruns __pycache__
