#!/usr/bin/env python
"""All statistics and summary tables for both learning rates (tables/statistics_*.json, summary_*.md, per_run_*.csv).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR
(see `python reproduce.py --help`).

    python scripts/figures/tables.py
"""

from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["tables"])
