install:
	python -m pip install -e .[dev]
data:
	python -m src.cli fetch --acknowledge-local-rights
train:
	python -m src.cli index
test:
	python -m pytest -q
evaluate:
	python -m src.cli evaluate
