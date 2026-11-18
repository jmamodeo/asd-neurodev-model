#!/usr/bin/env python
"""Figure 4: cumulative fraction of networks that reached the chance-level loss, learning rate 0.03 (figures/sessions_to_criterion_lr0.03.png).

Runs only the training runs this output needs (shared cache: run_cache/ and the shipped reproduce_data/), then plots.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig4_sessions_to_criterion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import reproduce  # noqa: E402

if __name__ == "__main__":
    reproduce.main(only=["fig4"])
