.PHONY: install lint format test gate check clean

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	python -m pytest

gate:
	python tools/check_release.py --ci

# Everything CI runs, in the same order.
check: lint gate test

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
