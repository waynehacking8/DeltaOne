.PHONY: env test select apply convert clean

env:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install -e .[dev] && \
	pre-commit install

test:
	pytest -q

select:
	python -m deltaone.cli.d1_select $(ARGS)

apply:
	python -m deltaone.cli.d1_apply $(ARGS)

convert:
	python -m deltaone.cli.d1_convert $(ARGS)

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
