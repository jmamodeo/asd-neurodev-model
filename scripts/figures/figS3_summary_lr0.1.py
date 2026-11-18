#!/usr/bin/env python
"""Supplementary summary panels at learning rate 0.1 (figures/summary_lr0.1.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/figS3_summary_lr0.1.py
"""
from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["figS3"])
