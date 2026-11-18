#!/usr/bin/env python
"""Figure 6: final loss, sessions to chance-level loss and held-out accuracy, learning rate 0.03.

Writes figures/summary_lr0.03.png.

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR
(see `python reproduce.py --help`).

    python scripts/figures/fig6_summary.py
"""

from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["fig6"])
