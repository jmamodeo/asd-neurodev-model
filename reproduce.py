#!/usr/bin/env python
"""Reproduce every figure and table of the paper with one command:

    python reproduce.py            # (or ./reproduce.sh, or `make figures`)

What it does
  1. For each learning-rate configuration (main = 0.03, supplementary = 0.1) it needs 4 conditions x 20 seeds per dataset
     (CIFAR-10 subset; plus the authors' image set if you supply it, see --original-images). Finished runs are cached
     in run_cache/ and the distributed cache reproduce_data/runs_lr*.json.gz is used first, so by default
     NOTHING is retrained and the command just re-analyzes and re-plots (about 1-2 minutes).
  2. Missing runs are trained (CIFAR-10 is downloaded automatically with scripts/fetch_cifar10.py), in parallel (--workers).
  3. Statistics, tables and figures are written to figures/ and tables/.

Individual figures: scripts/figures/fig1_model_schematic.py ... fig7_summary.py, figS1_..., figS2_..., figS3_... and tables.py each
build just that figure (training only the runs it needs, sharing the run cache); the same is available here as
python reproduce.py --only fig3 fig4 ...  (fig1 and fig2 need no training runs).

Useful options
  --only NAME [NAME ...]   build only these outputs: fig1 ... fig7 figS1 figS2 figS3 tables (default: everything)
  --retrain                ignore the distributed cache and retrain everything from scratch (hours; see README)
  --quick                  smoke test: 3 seeds, 20 sessions, CIFAR-10 only, outputs in figures_quick/ and tables_quick/
  --workers N              parallel training processes (default: number of CPU cores)
  --original-images DIR    folder with train_subset/ and test_subset/ (trucks, dogs, cats, cars) for the authors' image set
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
import figures_model as fm       # noqa: E402

CACHE = os.path.join(ROOT, "run_cache")
BUNDLES = os.path.join(ROOT, "reproduce_data")
CIFAR_DIR = os.path.join(ROOT, "image_set_cifar3")
CONFIG_NAME = {0.03: "Learning rate 0.03", 0.1: "Learning rate 0.1 (supplementary)"}
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


FIGURES = {   # name: (learning rate or None, kind); the file written is figures/<kind>_lr<lr>.png (figures/<kind>.png without runs)
    "fig1": (None, "model_schematic"), "fig2": (None, "release_map_distributions"),
    "fig3": (0.03, "loss_curves"), "fig4": (0.03, "sessions_to_criterion"), "fig5": (0.03, "auc_interaction"),
    "fig6": (0.03, "observed_vs_additive"), "fig7": (0.03, "summary"),
    "figS1": (0.1, "loss_curves"), "figS2": (0.1, "auc_interaction"), "figS3": (0.1, "summary"),
}
ALL_OUTPUTS = list(FIGURES) + ["tables"]


def get_runs(lr, opts):
    """Return the runs of one learning-rate configuration, training whatever is missing (cache first)."""
    tag = tag_for(lr, opts.quick)
    run_dir = os.path.join(CACHE, tag + (f"_s{opts.sessions}_n{opts.seeds}" if (opts.quick or opts.sessions != 100 or opts.seeds != 20) else ""))
    os.makedirs(run_dir, exist_ok=True)
    data_dirs = {"cifar3": CIFAR_DIR}
    if opts.original_images and not opts.quick:
        data_dirs["original"] = os.path.abspath(opts.original_images)
    if not opts.quick and not opts.retrain:
        n = seed_cache_from_bundle(lr, run_dir, opts.sessions, opts.seeds)
        if n:
            print(f"[{tag}] loaded {n} cached runs from reproduce_data/", flush=True)
    jobs = [(name, d, cond, s, lr, opts.sessions, run_dir) for cond in ("wnt", "both", "control", "ccl5")
            for s in range(opts.seeds) for name, d in data_dirs.items()
            if not os.path.exists(ex.run_path(run_dir, name, cond, s))]
    if jobs:
        if not opts.no_fetch and "cifar3" in {j[0] for j in jobs}:
            ensure_cifar()
        print(f"[{tag}] training {len(jobs)} runs on {opts.workers} workers ...", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(opts.workers) as pool:
            for k, path in enumerate(pool.imap_unordered(ex._job, jobs), 1):
                print(f"  [{time.time() - t0:7.0f}s] {k}/{len(jobs)} {os.path.basename(path)}", flush=True)
    runs = collect_runs(run_dir, {"cifar3", "original"}, opts.seeds, opts.sessions, lr)
    if not runs:
        sys.exit(f"[{tag}] no runs available")
    return runs


def build_parser(prog_doc):
    ap = argparse.ArgumentParser(description=prog_doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--original-images", default=None)
    ap.add_argument("--no-fetch", action="store_true", help="do not download CIFAR-10 (fail if training needs it)")
    return ap


def main(argv=None, only=None):
    """only: list of output names fixed by a per-figure script; otherwise --only (default: everything)."""
    ap = build_parser(__doc__)
    if only is None:
        ap.add_argument("--only", nargs="+", choices=ALL_OUTPUTS, default=ALL_OUTPUTS)
    opts = ap.parse_args(argv)
    wanted = only if only is not None else opts.only
    opts.seeds = opts.seeds or (3 if opts.quick else 20)
    opts.sessions = opts.sessions or (20 if opts.quick else 100)
    fig_dir = os.path.join(ROOT, "figures_quick" if opts.quick else "figures")
    tab_dir = os.path.join(ROOT, "tables_quick" if opts.quick else "tables")
    t_start = time.time()
    lrs = sorted({FIGURES[w][0] for w in wanted if w in FIGURES and FIGURES[w][0] is not None} | ({0.03, 0.1} if "tables" in wanted else set()), reverse=True)
    runs_by_lr = {lr: get_runs(lr, opts) for lr in lrs}       # only the configurations the requested outputs need
    for name in wanted:
        if name == "tables":
            for lr in lrs:
                an.write_tables(runs_by_lr[lr], tag_for(lr, opts.quick), tab_dir)
                print(f"[tables] {tag_for(lr, opts.quick)} -> {os.path.relpath(tab_dir, ROOT)}/", flush=True)
            continue
        lr, kind = FIGURES[name]
        if lr is None:                                       # model figures: no training runs needed
            os.makedirs(fig_dir, exist_ok=True)
            path = os.path.join(fig_dir, f"{kind}.png")
            {"model_schematic": fm.figure_model_schematic, "release_map_distributions": fm.figure_release_maps}[kind](path)
            print(f"[{name}] wrote {os.path.relpath(path, ROOT)}", flush=True)
            continue
        title = f"{CONFIG_NAME.get(lr, f'lr {lr:g}')}"
        path = an.make_figure(kind, runs_by_lr[lr], tag_for(lr, opts.quick), fig_dir, title)
        print(f"[{name}] wrote {os.path.relpath(path, ROOT)}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
