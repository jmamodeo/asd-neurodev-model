#!/usr/bin/env python
"""Supplementary Figure S1: loss curves at learning rate 0.1 (figures/loss_curves_lr0.1.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR
(see `python reproduce.py --help`).

    python scripts/figures/figs1_loss_curves_lr0_1.py
"""

from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["figs1"])
