#!/usr/bin/env python
"""Figure 1: training and held-out loss curves with 95% CI bands, learning rate 0.03 (figures/loss_curves_lr0.03.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig1_loss_curves.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import reproduce  # noqa: E402

if __name__ == "__main__":
    reproduce.main(only=["fig1"])
