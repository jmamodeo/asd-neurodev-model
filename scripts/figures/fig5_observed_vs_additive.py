#!/usr/bin/env python
"""Figure 5: observed vs additive effect of Wnt and CCL5, learning rate 0.03 (figures/observed_vs_additive_lr0.03.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig5_observed_vs_additive.py
"""
from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["fig5"])
