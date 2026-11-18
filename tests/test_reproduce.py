"""Tests for the reproduction pipeline: defaults, determinism, and that the shipped runs reproduce the paper's numbers."""
import gzip
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_default_learning_rate_is_0_03_and_original_selectable():
    import inspect
    import NeuralNetwork as nn
    assert nn.DEFAULT_LEARNING_RATE == 0.03 and nn.ORIGINAL_LEARNING_RATE == 0.1
    assert inspect.signature(nn.NeuralNetwork.train).parameters["learn_r"].default == 0.03
    src = open(os.path.join(ROOT, "main.py")).read()
    assert "--lr" in src and "DEFAULT_LEARNING_RATE" in src


def _write_images(root, per_class=2):
    import cv2
    rng = np.random.default_rng(0)
    for split in ("train_subset", "test_subset"):
        for cls in ("trucks", "dogs", "cats", "cars"):
            os.makedirs(os.path.join(root, split, cls), exist_ok=True)
            for i in range(per_class):
                cv2.imwrite(os.path.join(root, split, cls, f"{i}.png"), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


def test_experiment_is_deterministic_for_a_seed_and_shuffles_identically_across_conditions():
    import experiment as ex
    with tempfile.TemporaryDirectory() as tmp:
        _write_images(tmp)
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        os.makedirs(a); os.makedirs(b)
        ex.IMAGE_VECTOR_SIZE = 100
        ex.POOL_SIZE = 200_000
        pa = ex.run_one("t", tmp, "control", 3, 0.03, 3, a)
        pb = ex.run_one("t", tmp, "control", 3, 0.03, 3, b)
        ra, rb = json.load(open(pa)), json.load(open(pb))
        assert ra["train_loss"] == rb["train_loss"] and ra["val_acc"] == rb["val_acc"]
        pc = ex.run_one("t", tmp, "ccl5", 3, 0.03, 3, a)
        assert json.load(open(pc))["train_loss"] != ra["train_loss"]       # the perturbation changes the run
        assert ra["layer_sizes"] == [100, 100, 100, 100, 100, 4]


def test_shipped_runs_reproduce_paper_numbers():
    import analysis as an
    with gzip.open(os.path.join(ROOT, "reproduce_data", "runs_lr0.03.json.gz"), "rt") as f:
        runs = json.load(f)
    assert len(runs) == 160 and {r["seed"] for r in runs} == set(range(20))
    stat = an.compute_statistics(an.per_run_table(runs), rng=np.random.default_rng(0))
    o = stat["original"]
    assert abs(o["summary"]["control"]["auc"]["mean"] - 0.86) < 0.01
    assert abs(o["summary"]["both"]["auc"]["mean"] - 4.42) < 0.01
    assert o["anova"]["auc"]["Wnt x CCL5"]["p"] < 1e-9
    assert all(o["summary"][c]["auc"]["mean"] > o["summary"]["control"]["auc"]["mean"] for c in ("ccl5", "wnt", "both"))
