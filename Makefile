# `make figures` regenerates all paper figures/tables; `make quick` is a smoke test; `make paper` rebuilds the PDF;
# `make lint` runs ruff (needs: pip install -e .[lint]).
PYTHON ?= python
.PHONY: install figures quick retrain paper test lint
install:
	$(PYTHON) -m pip install -e .[test]
figures:
	$(PYTHON) reproduce.py
quick:
	$(PYTHON) reproduce.py --quick
retrain:
	$(PYTHON) reproduce.py --retrain
paper: figures
	$(PYTHON) docs/paper/build_paper.py
test:
	$(PYTHON) -m pytest tests -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
