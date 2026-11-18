#!/usr/bin/env python
"""Shortcut for `python -m asd_model.reproduce`: re-creates every figure and table of the paper (see --help).

    python reproduce.py [--only fig3 fig5 ...] [--quick] [--retrain] ...

Needs the package installed once from the repository root: pip install -e .
"""
try:
    from asd_model.reproduce import main
except ModuleNotFoundError as error:
    if error.name != "asd_model":
        raise
    raise SystemExit("asd_model is not installed. From the repository root run: pip install -e .")

if __name__ == "__main__":
    main()
