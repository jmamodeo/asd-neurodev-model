#!/usr/bin/env python
"""Figure 2: release-map (synaptic gain) distributions, control vs CCL5 (figures/release_map_distributions.png).

Needs no training runs.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig2_release_map_distributions.py
"""
from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["fig2"])
