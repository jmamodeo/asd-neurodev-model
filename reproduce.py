#!/usr/bin/env python
"""Re-creates every figure and table of the paper (the one way to reproduce them; see --help).

    python reproduce.py [--quick] [--retrain] [--workers N] [--seeds N] [--sessions N]
                        [--original-images DIR] [--no-fetch]

One figure at a time: python scripts/figures/<name>.py (see the README).
Needs the package installed once from the repository root: pip install -e .[test]
"""

try:
    from asd_model.reproduce import main
except ModuleNotFoundError as error:
    if error.name != "asd_model":
        raise
    raise SystemExit("asd_model is not installed. From the repository root run: pip install -e .[test]") from error

if __name__ == "__main__":
    main()
