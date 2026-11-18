"""Tests for the main.py outputs (training-loss plot and Excel table); also guards against seaborn/pandas changes."""

import subprocess
import sys

import matplotlib
import numpy as np

from asd_model.loss_outputs import export_loss_table, make_results_directory, plot_loss_over_time
from asd_model.neural_network import ASD_CCL5_DELTA, ASD_WNT_DELTA, NeuralNetwork, TrainingHistory
from asd_model.visual_input import encode_image_labels, image_to_matrix

matplotlib.use("Agg")

CLASS_LIST = ["trucks", "dogs", "cats", "cars"]


def _train_short_run(tmp_path, write_class_images):
    write_class_images(tmp_path / "images")
    images = image_to_matrix(str(tmp_path / "images"), CLASS_LIST, 100)
    labels = encode_image_labels(images)
    np.random.seed(0)
    net = NeuralNetwork(CLASS_LIST, 100, hidden_layer_number=2, ccl5_delta=ASD_CCL5_DELTA, wnt_delta=ASD_WNT_DELTA)
    return net.train(images, labels, learning_rate=0.03, sessions=3)


def test_training_returns_history_and_writes_nothing(tmp_path, monkeypatch, write_class_images):
    """train() only returns the losses; the plot and the table are written by the caller (main.py)."""
    monkeypatch.chdir(tmp_path)
    history = _train_short_run(tmp_path, write_class_images)
    assert isinstance(history, TrainingHistory)
    assert len(history.loss_table_list) == 3
    assert sorted(set(history.session_list)) == [1, 2, 3]
    assert len(history.loss_over_time) == len(history.session_list)
    assert not (tmp_path / "results").exists()


def test_loss_plot_and_table_are_written_to_one_results_folder(tmp_path, monkeypatch, write_class_images):
    """A short run of the model, as main.py does it, writes results/<date> results/ with the plot and the table."""
    monkeypatch.chdir(tmp_path)  # the plot and table are written to ./results/ (created if it is missing)
    history = _train_short_run(tmp_path, write_class_images)
    results_directory = make_results_directory()
    plot_loss_over_time(history.loss_over_time, history.session_list, results_directory)
    export_loss_table(history.loss_table_list, history.session_list, results_directory)

    out = list((tmp_path / "results").iterdir())
    assert len(out) == 1
    assert sorted(p.name for p in out[0].iterdir()) == ["training_loss_plot.png", "training_loss_table.xlsx"]


def test_module_entry_point_runs():
    """`python -m asd_model --help` starts the main.py command line."""
    result = subprocess.run([sys.executable, "-m", "asd_model", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--wnt-delta" in result.stdout
    assert "--learning-rate" in result.stdout and "--lr " not in result.stdout


def test_learning_rate_has_no_short_alias():
    """`--learning-rate` is the only spelling; the old `--lr` alias is rejected."""
    result = subprocess.run([sys.executable, "-m", "asd_model", "--lr", "0.03"], capture_output=True, text=True)
    assert result.returncode != 0 and "unrecognized arguments: --lr" in result.stderr
