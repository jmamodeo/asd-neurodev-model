"""Tests for the reproduction pipeline: defaults, determinism, and that the shipped runs reproduce the paper's numbers."""
import gzip
import json
import os
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_learning_rate_is_0_03_and_original_selectable():
    import inspect
    from asd_model import neural_network
    assert neural_network.DEFAULT_LEARNING_RATE == 0.03 and neural_network.ORIGINAL_LEARNING_RATE == 0.1
    assert inspect.signature(neural_network.NeuralNetwork.train).parameters["learn_r"].default == 0.03
    src = open(os.path.join(ROOT, "src", "asd_model", "main.py")).read()
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
    from asd_model import experiment
    saved = experiment.IMAGE_VECTOR_SIZE, experiment.POOL_SIZE
    try:
        _deterministic_check(experiment)
    finally:
        experiment.IMAGE_VECTOR_SIZE, experiment.POOL_SIZE = saved


def _deterministic_check(experiment):
    with tempfile.TemporaryDirectory() as tmp:
        _write_images(tmp)
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        os.makedirs(a); os.makedirs(b)
        experiment.IMAGE_VECTOR_SIZE = 100
        experiment.POOL_SIZE = 200_000
        pa = experiment.run_one("t", tmp, "control", 3, 0.03, 3, a)
        pb = experiment.run_one("t", tmp, "control", 3, 0.03, 3, b)
        ra, rb = json.load(open(pa)), json.load(open(pb))
        assert ra["train_loss"] == rb["train_loss"] and ra["val_acc"] == rb["val_acc"]
        pc = experiment.run_one("t", tmp, "ccl5", 3, 0.03, 3, a)
        assert json.load(open(pc))["train_loss"] != ra["train_loss"]       # the perturbation changes the run
        assert ra["layer_sizes"] == [100, 100, 100, 100, 100, 4]


def test_shipped_runs_reproduce_paper_numbers():
    from asd_model import paper_analysis
    with gzip.open(os.path.join(ROOT, "reproduce_data", "runs_lr0.03.json.gz"), "rt") as f:
        runs = json.load(f)
    assert len(runs) == 160 and {r["seed"] for r in runs} == set(range(20))
    stat = paper_analysis.compute_statistics(paper_analysis.per_run_table(runs), rng=np.random.default_rng(0))
    o = stat["original"]
    assert abs(o["summary"]["control"]["auc"]["mean"] - 0.86) < 0.01
    assert abs(o["summary"]["both"]["auc"]["mean"] - 4.42) < 0.01
    assert o["anova"]["auc"]["Wnt x CCL5"]["p"] < 1e-9
    assert all(o["summary"][c]["auc"]["mean"] > o["summary"]["control"]["auc"]["mean"] for c in ("ccl5", "wnt", "both"))


def test_every_figure_has_a_standalone_script():
    import os
    from asd_model import reproduce
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = os.listdir(os.path.join(root, "scripts", "figures"))
    for key in reproduce.FIGURES:
        assert any(f.startswith(key.lower() + "_") for f in scripts), key
    assert "tables.py" in scripts


def test_schematic_layer_sizes_match_shipped_runs():
    from asd_model import reproduce
    from asd_model import model_figures
    seen = {r["cond"]: r["layer_sizes"] for r in reproduce.load_bundle(0.03)}
    assert model_figures.layer_sizes(0.0) == seen["control"] == seen["ccl5"]
    assert model_figures.layer_sizes(model_figures.WNT_DELTA) == seen["wnt"] == seen["both"] == [900, 1276, 1810, 2567, 3641, 4]


def test_release_map_figure_statistics():
    import numpy as np
    from asd_model import model_figures
    for delta in (0.0, model_figures.CCL5_DELTA):
        v = model_figures.sample_release_values(delta, n=1_000_000)
        mean = 1 + delta
        assert abs(v.mean() - mean) < 0.002
        assert abs(v.std() - np.sqrt(100 * mean) / 100) < 0.002


def test_observed_vs_additive_estimates():
    import numpy as np
    from asd_model import reproduce
    from asd_model import paper_analysis
    df = paper_analysis.per_run_table([r for r in reproduce.load_bundle(0.03) if r["data"] == "cifar3"])
    E = paper_analysis.observed_vs_additive(df, "auc")
    m = df.groupby("cond").auc.mean()
    assert np.isclose(E["additive"][0], m["ccl5"] + m["wnt"] - m["control"])
    assert np.isclose(E["interaction"][0], m["both"] - E["additive"][0])
    for k, (p, lo, hi) in E.items():
        assert lo <= p <= hi
    assert E["interaction"][1] > 0          # the interval excludes zero (see paper)
