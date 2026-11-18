"""Tests for the main.py outputs (training-loss plot and Excel table), which also guard against seaborn/pandas API changes."""
import os
import sys

import numpy as np


def _write_images(root, per_class=2):
    import cv2
    rng = np.random.default_rng(0)
    for cls in ("trucks", "dogs", "cats", "cars"):
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for i in range(per_class):
            cv2.imwrite(os.path.join(root, cls, f"{i}.png"), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


def test_training_writes_loss_plot_and_table(tmp_path, monkeypatch):
    """A short run of the model, as main.py does it, writes results/<date> results/ with the plot and the table."""
    import matplotlib
    matplotlib.use("Agg")
    from asd_model.neural_network import NeuralNetwork
    from asd_model.visual_input import encode_image_labels, image_to_matrix

    class_list = ['trucks', 'dogs', 'cats', 'cars']
    _write_images(str(tmp_path / "images"))
    images = image_to_matrix(str(tmp_path / "images"), class_list, 100)
    labels = encode_image_labels(images)

    monkeypatch.chdir(tmp_path)          # the plot and table are written to ./results/ (created if it is missing)
    np.random.seed(0)
    net = NeuralNetwork(class_list, 100, hidden_layer_number=2, ccl5_delta=0.40467, wnt_delta=0.41859)
    net.train(images, labels, learning_rate=0.03, sessions=3)

    out = list((tmp_path / "results").iterdir())
    assert len(out) == 1
    assert sorted(p.name for p in out[0].iterdir()) == ["training_loss_plot.png", "training_loss_table.xlsx"]


def test_module_entry_point_runs():
    """`python -m asd_model --help` starts the main.py command line."""
    import subprocess
    result = subprocess.run([sys.executable, "-m", "asd_model", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--wnt-delta" in result.stdout
