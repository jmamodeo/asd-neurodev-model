#!/usr/bin/env python
"""Figure 3: training and held-out loss curves, learning rate 0.03 (figures/loss_curves_lr0.03.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR
(see `python reproduce.py --help`).

    python scripts/figures/fig3_loss_curves.py
"""

from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["fig3"])
