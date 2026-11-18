#!/usr/bin/env bash
# One command to regenerate every figure and table of the paper (see reproduce.py --help; needs `pip install -e .` once).
set -euo pipefail
cd "$(dirname "$0")"
exec python reproduce.py "$@"
