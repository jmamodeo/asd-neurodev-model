#!/usr/bin/env python
"""Trains and tests one network (the one way to run the model).

    python main.py [--learning-rate X] [--sessions N] [--wnt-delta X] [--ccl5-delta X]

Needs the package installed once from the repository root: pip install -e .[test]
"""

import runpy

try:
    import asd_model.main  # noqa: F401  (checks that the package is installed)
except ModuleNotFoundError as error:
    if error.name != "asd_model":
        raise
    raise SystemExit("asd_model is not installed. From the repository root run: pip install -e .[test]") from error

if __name__ == "__main__":
    runpy.run_module("asd_model.main", run_name="__main__")
