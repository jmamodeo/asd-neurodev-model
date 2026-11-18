#!/usr/bin/env python
"""Supplementary Figure S2: observed vs additive effect of Wnt and CCL5, learning rate 0.1.

Writes figures/observed_vs_additive_lr0.1.png.

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR
(see `python reproduce.py --help`).

    python scripts/figures/figs2_observed_vs_additive_lr0_1.py
"""

from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["figs2"])
