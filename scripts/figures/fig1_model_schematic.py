#!/usr/bin/env python
"""Figure 1: schematic of the network, the Wnt layer widths and the CCL5 release maps (figures/model_schematic.png).

Needs no training runs.
Options: --quick, --retrain, --workers N, --seeds N, --sessions N, --original-images DIR (see `python reproduce.py --help`).

    python scripts/figures/fig1_model_schematic.py
"""
from asd_model import reproduce

if __name__ == "__main__":
    reproduce.main(only=["fig1"])
