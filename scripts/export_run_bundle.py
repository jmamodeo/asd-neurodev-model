"""Developer helper: pack a directory of per-run JSON files into reproduce_data/runs_lr<lr>.json.gz.

python scripts/export_run_bundle.py run_cache/lr0.03 0.03
"""

import glob
import gzip
import json
import os
import sys

run_dir, learning_rate = sys.argv[1], float(sys.argv[2])
runs = []
for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    with open(path) as f:
        run = json.load(f)
    run["lr"] = learning_rate
    runs.append(run)
repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(repository_root, "reproduce_data", f"runs_lr{learning_rate:g}.json.gz")
os.makedirs(os.path.dirname(out), exist_ok=True)
with gzip.GzipFile(out, "wb", mtime=0) as raw:  # mtime=0: byte-reproducible archive
    raw.write(json.dumps(runs, separators=(",", ":")).encode())
print(len(runs), "runs ->", out, os.path.getsize(out) // 1024, "KiB")
