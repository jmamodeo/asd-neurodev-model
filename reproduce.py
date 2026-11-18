#!/usr/bin/env python
"""Reproduce every figure and table of the revised paper with one command:

    python reproduce.py            # (or ./reproduce.sh, or `make figures`)

What it does
  1. For each learning-rate configuration (main = 0.03, supplementary = 0.1) it needs 4 conditions x 20 seeds per dataset
     (CIFAR-10 subset; plus the original image set if you supply it, see --original-images). Finished runs are cached
     in run_cache/ and the distributed cache reproduce_data/runs_lr*.json.gz is used first, so by default
     NOTHING is retrained and the command just re-analyzes and re-plots (about 1-2 minutes).
  2. Missing runs are trained (CIFAR-10 is downloaded automatically with scripts/fetch_cifar10.py), in parallel (--workers).
  3. Statistics, tables and figures are written to figures/ and tables/.

Useful options
  --retrain                ignore the distributed cache and retrain everything from scratch (hours; see README)
  --quick                  smoke test: 3 seeds, 20 sessions, CIFAR-10 only, outputs in figures_quick/ and tables_quick/
  --workers N              parallel training processes (default: number of CPU cores)
  --lr 0.03 0.1            learning-rate configurations to run (0.03 = main, 0.1 = original value; default both)
  --original-images DIR    folder with train_subset/ and test_subset/ (trucks, dogs, cats, cars) for the original image set
  --seeds N / --sessions N override the number of seeds (default 20) / training sessions (default 100)

Seeds are fixed (seed s always gives the same initial weights, shuffle order and release-map stream), so reruns are
deterministic on the same machine/library versions.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import experiment as ex          # noqa: E402
import analysis as an            # noqa: E402

CACHE = os.path.join(ROOT, "run_cache")
BUNDLES = os.path.join(ROOT, "reproduce_data")
CIFAR_DIR = os.path.join(ROOT, "image_set_cifar3")
CONFIG_NAME = {0.03: "main configuration (lr 0.03)", 0.1: "original learning rate (lr 0.1)"}
# figure file stem used in the paper for each configuration
PAPER_FIGURES = {0.03: "lr0.03", 0.1: "lr0.1"}


def tag_for(lr, quick):
    return ("quick_" if quick else "") + f"lr{lr:g}"


def load_bundle(lr):
    path = os.path.join(BUNDLES, f"runs_lr{lr:g}.json.gz")
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt") as f:
        return json.load(f)


def seed_cache_from_bundle(lr, run_dir, sessions, seeds):
    n = 0
    for d in load_bundle(lr):
        if d["sessions"] != sessions or d["seed"] >= seeds:
            continue
        path = ex.run_path(run_dir, d["data"], d["cond"], d["seed"])
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(d, f)
            n += 1
    return n


def ensure_cifar():
    if os.path.isdir(os.path.join(CIFAR_DIR, "train_subset")) and os.path.isdir(os.path.join(CIFAR_DIR, "test_subset")):
        return
    print("Fetching CIFAR-10 (about 163 MB, cached in .cache/) ...", flush=True)
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "fetch_cifar10.py"), "--train-per-class", "3",
                           "--test-per-class", "3", "--out", CIFAR_DIR])


def collect_runs(run_dir, datasets, seeds, sessions, lr):
    runs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if d["data"] in datasets and d["seed"] < seeds and d["sessions"] == sessions and abs(d.get("lr", lr) - lr) < 1e-12:
            runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--lr", type=float, nargs="+", default=[0.03, 0.1])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--original-images", default=None)
    ap.add_argument("--no-fetch", action="store_true", help="do not download CIFAR-10 (fail if training needs it)")
    args = ap.parse_args()

    seeds = args.seeds or (3 if args.quick else 20)
    sessions = args.sessions or (20 if args.quick else 100)
    fig_dir = os.path.join(ROOT, "figures_quick" if args.quick else "figures")
    tab_dir = os.path.join(ROOT, "tables_quick" if args.quick else "tables")
    data_dirs = {"cifar3": CIFAR_DIR}
    if args.original_images and not args.quick:
        data_dirs["original"] = os.path.abspath(args.original_images)
    t_start = time.time()

    for lr in args.lr:
        tag = tag_for(lr, args.quick)
        run_dir = os.path.join(CACHE, tag + (f"_s{sessions}_n{seeds}" if (args.quick or sessions != 100 or seeds != 20) else ""))
        os.makedirs(run_dir, exist_ok=True)
        if not args.quick and not args.retrain:
            n = seed_cache_from_bundle(lr, run_dir, sessions, seeds)
            if n:
                print(f"[{tag}] loaded {n} cached runs from reproduce_data/", flush=True)
        jobs = [(name, d, cond, s, lr, sessions, run_dir) for cond in ("wnt", "both", "control", "ccl5")
                for s in range(seeds) for name, d in data_dirs.items()
                if not os.path.exists(ex.run_path(run_dir, name, cond, s))]
        if jobs:
            if not args.no_fetch and "cifar3" in {j[0] for j in jobs}:
                ensure_cifar()
            print(f"[{tag}] training {len(jobs)} runs on {args.workers} workers ...", flush=True)
            t0 = time.time()
            with mp.get_context("fork").Pool(args.workers) as pool:
                for k, path in enumerate(pool.imap_unordered(ex._job, jobs), 1):
                    print(f"  [{time.time() - t0:7.0f}s] {k}/{len(jobs)} {os.path.basename(path)}", flush=True)
        runs = collect_runs(run_dir, {"cifar3", "original"}, seeds, sessions, lr)
        if not runs:
            sys.exit(f"[{tag}] no runs available")
        title = f"Corrected network, {CONFIG_NAME.get(lr, f'lr {lr:g}')}"
        an.analyze(runs, tag, fig_dir, tab_dir, title)
        print(f"[{tag}] {len(runs)} runs analyzed -> {os.path.relpath(fig_dir, ROOT)}/ , {os.path.relpath(tab_dir, ROOT)}/", flush=True)

    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
