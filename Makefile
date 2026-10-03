.PHONY: install dev-install init-db ingest gen-clients replay run-daily backtest what-if report test lint clean

PYTHON := python
CLI := collateraliq
SEED := 42
START_DATE := 2022-01-01
END_DATE := 2026-09-30
DAILY_DATE := 2026-09-30

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

init-db:
	$(CLI) init-db

ingest:
	$(CLI) ingest

gen-clients:
	$(CLI) gen-clients --seed $(SEED)

replay:
	$(CLI) replay --from $(START_DATE) --to $(END_DATE)

run-daily:
	$(CLI) run-daily --date $(DAILY_DATE)

backtest:
	$(CLI) backtest-im

what-if:
	$(CLI) what-if --scenario config/im_lookback_change.yaml

report:
	$(CLI) report --date $(DAILY_DATE)

full-pipeline: init-db ingest gen-clients replay run-daily backtest report

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/unit/ -v --tb=short

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/

streamlit:
	streamlit run app/streamlit_app.py

api:
	uvicorn collateraliq.api.pretrade_quote:app --reload --port 8001

help:
	@echo "CollateralIQ Makefile targets:"
	@echo "  install       Install package"
	@echo "  dev-install   Install with dev dependencies"
	@echo "  full-pipeline Run complete pipeline"
	@echo "  test          Run all tests"
	@echo "  streamlit     Launch Streamlit dashboard"
	@echo "  api           Launch FastAPI pre-trade quote service"
