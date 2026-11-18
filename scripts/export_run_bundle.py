"""Developer helper: pack a directory of per-run JSON files into reproduce_data/runs_lr<lr>.json.gz.

    python scripts/export_run_bundle.py run_cache/lr0.03 0.03
"""
import glob, gzip, json, os, sys

run_dir, lr = sys.argv[1], float(sys.argv[2])
runs = []
for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    d = json.load(open(path)); d["lr"] = lr; runs.append(d)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reproduce_data", f"runs_lr{lr:g}.json.gz")
os.makedirs(os.path.dirname(out), exist_ok=True)
with gzip.GzipFile(out, "wb", mtime=0) as raw:   # mtime=0: byte-reproducible archive
    raw.write(json.dumps(runs, separators=(",", ":")).encode())
print(len(runs), "runs ->", out, os.path.getsize(out) // 1024, "KiB")
