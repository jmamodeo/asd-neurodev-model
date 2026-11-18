"""Tests for the reproduction pipeline: defaults, determinism, and the paper's numbers from the shipped runs."""

import gzip
import inspect
import json
import os
import pathlib

import numpy as np
import pytest

from asd_model import experiment, model_figures, neural_network, paper_analysis, reproduce

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = pathlib.Path(ROOT)


def test_default_learning_rate_is_0_03_and_supplement_selectable():
    assert neural_network.DEFAULT_LEARNING_RATE == 0.03
    assert neural_network.SUPPLEMENT_LEARNING_RATE == 0.1
    assert inspect.signature(neural_network.NeuralNetwork.train).parameters["learning_rate"].default == 0.03
    with open(os.path.join(ROOT, "src", "asd_model", "main.py")) as f:
        main_source = f.read()
    assert "--learning-rate" in main_source and "DEFAULT_LEARNING_RATE" in main_source


def test_experiment_is_deterministic_for_a_seed_and_shuffles_identically_across_conditions(
    tmp_path, write_class_images, monkeypatch
):
    monkeypatch.setattr(experiment, "IMAGE_VECTOR_SIZE", 100)
    monkeypatch.setattr(experiment, "POOL_SIZE", 200_000)
    for split in ("train_subset", "test_subset"):
        write_class_images(tmp_path / split)
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    first_path = experiment.run_experiment("t", str(tmp_path), "control", 3, 0.03, 3, str(first_dir))
    second_path = experiment.run_experiment("t", str(tmp_path), "control", 3, 0.03, 3, str(second_dir))
    with open(first_path) as f:
        first_run = json.load(f)
    with open(second_path) as f:
        second_run = json.load(f)
    assert first_run["train_loss"] == second_run["train_loss"] and first_run["val_acc"] == second_run["val_acc"]

    ccl5_path = experiment.run_experiment("t", str(tmp_path), "ccl5", 3, 0.03, 3, str(first_dir))
    with open(ccl5_path) as f:
        ccl5_run = json.load(f)
    assert ccl5_run["train_loss"] != first_run["train_loss"]  # the perturbation changes the run
    assert first_run["layer_sizes"] == [100, 100, 100, 100, 100, 4]


def test_shipped_runs_reproduce_paper_numbers():
    with gzip.open(os.path.join(ROOT, "reproduce_data", "runs_lr0.03.json.gz"), "rt") as f:
        runs = json.load(f)
    assert len(runs) == 160 and {run["seed"] for run in runs} == set(range(20))
    statistics = paper_analysis.compute_statistics(paper_analysis.per_run_table(runs), rng=np.random.default_rng(0))
    original = statistics["original"]
    assert abs(original["summary"]["control"]["auc"]["mean"] - 0.86) < 0.01
    assert abs(original["summary"]["both"]["auc"]["mean"] - 4.42) < 0.01
    assert original["anova"]["auc"]["Wnt x CCL5"]["p"] < 1e-9
    control_auc = original["summary"]["control"]["auc"]["mean"]
    assert all(original["summary"][condition]["auc"]["mean"] > control_auc for condition in ("ccl5", "wnt", "both"))


def test_every_figure_has_a_standalone_script():
    scripts = os.listdir(os.path.join(ROOT, "scripts", "figures"))
    for key in reproduce.FIGURES:
        assert any(script.startswith(key.lower() + "_") for script in scripts), key
    assert "tables.py" in scripts


def test_schematic_layer_sizes_match_shipped_runs():
    seen = {run["cond"]: run["layer_sizes"] for run in reproduce.load_bundle(0.03)}
    assert model_figures.layer_sizes(0.0) == seen["control"] == seen["ccl5"]
    wnt_sizes = model_figures.layer_sizes(neural_network.ASD_WNT_DELTA)
    assert wnt_sizes == seen["wnt"] == seen["both"] == [900, 1276, 1810, 2567, 3641, 4]


def test_release_map_figure_statistics():
    for delta in (0.0, neural_network.ASD_CCL5_DELTA):
        values = model_figures.sample_release_values(delta, n=1_000_000)
        mean = 1 + delta
        assert abs(values.mean() - mean) < 0.002
        assert abs(values.std() - np.sqrt(100 * mean) / 100) < 0.002


def test_observed_vs_additive_estimates():
    runs = [run for run in reproduce.load_bundle(0.03) if run["data"] == "cifar3"]
    per_run = paper_analysis.per_run_table(runs)
    effects = paper_analysis.observed_vs_additive(per_run, "auc")
    means = per_run.groupby("cond").auc.mean()
    assert np.isclose(effects["additive"][0], means["ccl5"] + means["wnt"] - means["control"])
    assert np.isclose(effects["interaction"][0], means["both"] - effects["additive"][0])
    for estimate, lower, upper in effects.values():
        assert lower <= estimate <= upper
    assert effects["interaction"][1] > 0  # the interval excludes zero (see paper)


def test_reproduce_has_one_command_line_without_only_or_abbreviations():
    """`--only` and abbreviated flags are gone; a single figure is built by its script in scripts/figures/."""
    for flag in ("--only", "--quic", "--retr"):
        with pytest.raises(SystemExit) as error:
            reproduce.main([flag])
        assert error.value.code == 2, flag


def test_one_way_to_run_each_task():
    """The only entry points are the root `main.py`, the root `reproduce.py` and the per-figure scripts."""
    assert (ROOT_PATH / "main.py").is_file() and (ROOT_PATH / "reproduce.py").is_file()
    for removed in ("Makefile", "reproduce.sh", "requirements.txt", "src/asd_model/__main__.py"):
        assert not (ROOT_PATH / removed).exists(), removed
    assert "[project.scripts]" not in (ROOT_PATH / "pyproject.toml").read_text()
    assert "__main__" not in inspect.getsource(reproduce)
