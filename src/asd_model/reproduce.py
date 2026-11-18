"""Reproduce every figure and table of the paper with one command.

    python reproduce.py   # in the repository root

What it does
  1. For each learning-rate configuration (main = 0.03, supplementary = 0.1) it needs 4 conditions x 20 seeds per
     dataset (CIFAR-10 subset; plus the authors' image set if you supply it, see --original-images). Finished
     runs are cached in run_cache/ and the distributed cache reproduce_data/runs_lr*.json.gz is used first, so by
     default NOTHING is retrained and the command just re-analyzes and re-plots (about 1-2 minutes).
  2. Missing runs are trained (CIFAR-10 is downloaded automatically with scripts/fetch_cifar10.py), in parallel
     (--workers).
  3. Statistics, tables and figures are written to figures/ and tables/.

One figure at a time: python scripts/figures/<name>.py (fig1_model_schematic.py ... fig6_summary.py, figs1_...,
figs2_..., figs3_... and tables.py). Each builds just that output, training only the runs it needs and sharing the run
cache; they call main(only=[...]) below and accept the options listed here (fig1 and fig2 need no training runs).

Useful options
  --retrain                ignore the distributed cache and retrain everything from scratch (hours; see README)
  --quick                  smoke test: 3 seeds, 20 sessions, CIFAR-10 only, outputs in figures_quick/ and tables_quick/
  --workers N              parallel training processes (default: number of CPU cores)
  --original-images DIR    folder with train_subset/ and test_subset/ (trucks, dogs, cats, cars) of the authors' images
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
import multiprocessing
import subprocess
import sys
import time

from asd_model import experiment, model_figures, paper_analysis
from asd_model.neural_network import DEFAULT_LEARNING_RATE, DEFAULT_SESSIONS, SUPPLEMENT_LEARNING_RATE

# repository root (src/asd_model/ is two levels down)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = os.path.join(ROOT, "run_cache")
BUNDLES = os.path.join(ROOT, "reproduce_data")
CIFAR_DIR = os.path.join(ROOT, "image_set_cifar3")
CONFIG_NAME = {
    DEFAULT_LEARNING_RATE: "Learning rate 0.03",
    SUPPLEMENT_LEARNING_RATE: "Learning rate 0.1 (supplementary)",
}
DEFAULT_SEEDS = 20
QUICK_SEEDS = 3
QUICK_SESSIONS = 20
CONDITION_TRAINING_ORDER = ("wnt", "both", "control", "ccl5")  # the slowest (widest) networks start first
IMAGE_SET_KEYS = {"cifar3", "original"}
CIFAR_TRAIN_PER_CLASS = 3
CIFAR_TEST_PER_CLASS = 3

# name: (learning rate or None, kind); the file written is figures/<kind>_lr<learning rate>.png
# (figures/<kind>.png without runs)
FIGURES = {
    "fig1": (None, "model_schematic"),
    "fig2": (None, "release_map_distributions"),
    "fig3": (DEFAULT_LEARNING_RATE, "loss_curves"),
    "fig4": (DEFAULT_LEARNING_RATE, "sessions_to_criterion"),
    "fig5": (DEFAULT_LEARNING_RATE, "observed_vs_additive"),
    "fig6": (DEFAULT_LEARNING_RATE, "summary"),
    "figs1": (SUPPLEMENT_LEARNING_RATE, "loss_curves"),
    "figs2": (SUPPLEMENT_LEARNING_RATE, "observed_vs_additive"),
    "figs3": (SUPPLEMENT_LEARNING_RATE, "summary"),
}
ALL_OUTPUTS = [*FIGURES, "tables"]
MODEL_FIGURES = {
    "model_schematic": model_figures.figure_model_schematic,
    "release_map_distributions": model_figures.figure_release_maps,
}


def make_tag(learning_rate, quick):
    """Return the tag used in file and folder names, e.g. ``lr0.03`` (``quick_lr0.03`` in quick mode)."""
    return ("quick_" if quick else "") + f"lr{learning_rate:g}"


def load_bundle(learning_rate):
    """Return the runs shipped in ``reproduce_data/`` for a learning rate (an empty list if there is no bundle)."""
    path = os.path.join(BUNDLES, f"runs_lr{learning_rate:g}.json.gz")
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt") as f:
        return json.load(f)


def seed_cache_from_bundle(learning_rate, run_dir, sessions, seeds):
    """Copy the shipped runs that match the requested sessions and seeds into the run cache; return how many."""
    n_copied = 0
    for run in load_bundle(learning_rate):
        if run["sessions"] != sessions or run["seed"] >= seeds:
            continue
        path = experiment.run_path(run_dir, run["data"], run["cond"], run["seed"])
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(run, f)
            n_copied += 1
    return n_copied


def ensure_cifar():
    """Download the CIFAR-10 image subset (once) if it is not there yet."""
    if os.path.isdir(os.path.join(CIFAR_DIR, "train_subset")) and os.path.isdir(os.path.join(CIFAR_DIR, "test_subset")):
        return
    print("Fetching CIFAR-10 (about 163 MB, cached in .cache/) ...", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "fetch_cifar10.py"),
            "--train-per-class",
            str(CIFAR_TRAIN_PER_CLASS),
            "--test-per-class",
            str(CIFAR_TEST_PER_CLASS),
            "--out",
            CIFAR_DIR,
        ]
    )


def collect_runs(run_dir, image_sets, seeds, sessions, learning_rate):
    """Load the cached runs of one configuration (the requested image sets, seeds, sessions and learning rate)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        if (
            run["data"] in image_sets
            and run["seed"] < seeds
            and run["sessions"] == sessions
            and abs(run.get("lr", learning_rate) - learning_rate) < 1e-12
        ):
            runs.append(run)
    return runs


def get_runs(learning_rate, options):
    """Return the runs of one learning-rate configuration, training whatever is missing (cache first)."""
    tag = make_tag(learning_rate, options.quick)
    custom_size = options.quick or options.sessions != DEFAULT_SESSIONS or options.seeds != DEFAULT_SEEDS
    run_dir = os.path.join(CACHE, tag + (f"_s{options.sessions}_n{options.seeds}" if custom_size else ""))
    os.makedirs(run_dir, exist_ok=True)
    data_dirs = {"cifar3": CIFAR_DIR}
    if options.original_images and not options.quick:
        data_dirs["original"] = os.path.abspath(options.original_images)
    if not options.quick and not options.retrain:
        n_loaded = seed_cache_from_bundle(learning_rate, run_dir, options.sessions, options.seeds)
        if n_loaded:
            print(f"[{tag}] loaded {n_loaded} cached runs from reproduce_data/", flush=True)
    jobs = [
        (image_set, image_dir, condition, seed, learning_rate, options.sessions, run_dir)
        for condition in CONDITION_TRAINING_ORDER
        for seed in range(options.seeds)
        for image_set, image_dir in data_dirs.items()
        if not os.path.exists(experiment.run_path(run_dir, image_set, condition, seed))
    ]
    if jobs:
        if not options.no_fetch and "cifar3" in {job[0] for job in jobs}:
            ensure_cifar()
        print(f"[{tag}] training {len(jobs)} runs on {options.workers} workers ...", flush=True)
        start_time = time.time()
        with multiprocessing.get_context("fork").Pool(options.workers) as pool:
            for k, path in enumerate(pool.imap_unordered(experiment.run_job, jobs), 1):
                print(f"  [{time.time() - start_time:7.0f}s] {k}/{len(jobs)} {os.path.basename(path)}", flush=True)
    runs = collect_runs(run_dir, IMAGE_SET_KEYS, options.seeds, options.sessions, learning_rate)
    if not runs:
        sys.exit(f"[{tag}] no runs available")
    return runs


def build_parser(description):
    """Return the command-line parser (shared by ``reproduce.py`` and the per-figure scripts)."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--original-images", default=None)
    parser.add_argument("--no-fetch", action="store_true", help="do not download CIFAR-10 (fail if training needs it)")
    return parser


def main(argv=None, only=None):
    """Re-create the requested figures and tables (everything by default).

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (default: ``sys.argv``).
    only : list of str, optional
        Output names fixed by a per-figure script (keys of ``FIGURES`` or ``"tables"``); if None, everything.
    """
    parser = build_parser(__doc__)
    options = parser.parse_args(argv)
    wanted = ALL_OUTPUTS if only is None else only
    options.seeds = options.seeds or (QUICK_SEEDS if options.quick else DEFAULT_SEEDS)
    options.sessions = options.sessions or (QUICK_SESSIONS if options.quick else DEFAULT_SESSIONS)
    figure_dir = os.path.join(ROOT, "figures_quick" if options.quick else "figures")
    table_dir = os.path.join(ROOT, "tables_quick" if options.quick else "tables")
    start_time = time.time()
    # only the configurations the requested outputs need
    learning_rates = {FIGURES[name][0] for name in wanted if name in FIGURES and FIGURES[name][0] is not None}
    if "tables" in wanted:
        learning_rates |= {DEFAULT_LEARNING_RATE, SUPPLEMENT_LEARNING_RATE}
    learning_rates = sorted(learning_rates, reverse=True)
    runs_by_learning_rate = {rate: get_runs(rate, options) for rate in learning_rates}
    for name in wanted:
        if name == "tables":
            for rate in learning_rates:
                tag = make_tag(rate, options.quick)
                paper_analysis.write_tables(runs_by_learning_rate[rate], tag, table_dir)
                print(f"[tables] {tag} -> {os.path.relpath(table_dir, ROOT)}/", flush=True)
            continue
        learning_rate, kind = FIGURES[name]
        if learning_rate is None:  # model figures: no training runs needed
            os.makedirs(figure_dir, exist_ok=True)
            path = os.path.join(figure_dir, f"{kind}.png")
            MODEL_FIGURES[kind](path)
            print(f"[{name}] wrote {os.path.relpath(path, ROOT)}", flush=True)
            continue
        title = f"{CONFIG_NAME.get(learning_rate, f'lr {learning_rate:g}')}"
        is_supplement = learning_rate == SUPPLEMENT_LEARNING_RATE
        height = paper_analysis.SUPPLEMENT_FIGURE_HEIGHTS.get(kind) if is_supplement else None
        path = paper_analysis.make_figure(
            kind,
            runs_by_learning_rate[learning_rate],
            make_tag(learning_rate, options.quick),
            figure_dir,
            title,
            height=height,
        )
        print(f"[{name}] wrote {os.path.relpath(path, ROOT)}", flush=True)
    print(f"done in {time.time() - start_time:.0f}s")
