#!/usr/bin/env python
"""Figure 2: release-map (synaptic gain) distributions, control vs CCL5 (figures/release_map_distributions.png).

Needs no training runs.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig2_release_map_distributions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import reproduce  # noqa: E402

if __name__ == "__main__":
    reproduce.main(only=["fig2"])
