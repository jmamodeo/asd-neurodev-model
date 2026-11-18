# `make figures` regenerates all paper figures/tables; `make quick` is a smoke test; `make paper` rebuilds the PDF.
PYTHON ?= python
.PHONY: install figures quick retrain paper test
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
