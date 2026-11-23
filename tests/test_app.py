import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from asd_model.app import (
    DEFAULT_PRESET,
    DEMO_CLASS_NAMES,
    DEMO_IMAGE_VECTOR_SIZE,
    DIAGRAM_SHAPE_CAPTION,
    PRESETS,
    area_under_training_loss,
    demo_images_available,
    demo_layer_sizes,
    load_demo_images,
    make_demo_release_sampler,
    plot_network_diagram,
    run_live_condition,
)
from asd_model.neural_network import ASD_CCL5_DELTA, ASD_WNT_DELTA

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
needs_images = pytest.mark.skipif(not demo_images_available(), reason="demo_images/ has no PNG files")


def new_app():
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    return app


def listed_sizes(app):
    line = next(m.value for m in app.markdown if m.value.startswith("Layer sizes:"))
    return [int(part) for part in line.split(":")[1].split(",")]


def test_demo_layer_sizes_follow_growth_rule():
    assert demo_layer_sizes(0) == [100, 100, 100, 100, 100, 4]
    assert demo_layer_sizes(ASD_WNT_DELTA) == [100, 141, 200, 283, 401, 4]
    assert demo_layer_sizes(0.5, image_vector_size=10, hidden_layer_number=2, n_classes=3) == [10, 15, 22, 3]


def test_network_diagram_returns_figure_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = plot_network_diagram(ASD_WNT_DELTA, ASD_CCL5_DELTA)
    assert isinstance(fig, Figure)
    assert list(tmp_path.iterdir()) == []


def test_area_under_training_loss():
    assert area_under_training_loss([1.0, 1.0, 1.0]) == pytest.approx(2.0)
    assert area_under_training_loss([0.0, 2.0]) == pytest.approx(1.0)


def test_release_sampler_shapes_and_float32():
    sampler = make_demo_release_sampler(ASD_CCL5_DELTA, seed=1, pool_size=5000)
    weights = [np.zeros((10, 20), dtype=np.float32), np.zeros((4, 10), dtype=np.float32)]
    maps = sampler(weights, ASD_CCL5_DELTA)
    assert [m.shape for m in maps] == [(10, 20), (4, 10)]
    assert all(m.dtype == np.float32 for m in maps)


def tiny_images():
    rng = np.random.default_rng(0)
    images = {
        name: [rng.random((DEMO_IMAGE_VECTOR_SIZE, 1)).astype(np.float32) for _ in range(3)]
        for name in DEMO_CLASS_NAMES
    }
    labels = {name: np.eye(4, dtype=np.float32)[[i]].T for i, name in enumerate(DEMO_CLASS_NAMES)}
    return images, labels


def test_run_live_condition_keys_and_repeatability():
    if demo_images_available():
        train, test, labels = load_demo_images()
    else:
        train, labels = tiny_images()
        test = train
    first = run_live_condition(ASD_CCL5_DELTA, ASD_WNT_DELTA, 1, 0.03, 3, train, test, labels)
    again = run_live_condition(ASD_CCL5_DELTA, ASD_WNT_DELTA, 1, 0.03, 3, train, test, labels)
    expected = {
        "train_loss",
        "val_loss",
        "val_acc",
        "layer_sizes",
        "final_train_loss",
        "final_val_loss",
        "final_val_acc",
        "seconds",
    }
    assert expected <= set(first)
    assert len(first["train_loss"]) == 1
    assert first["layer_sizes"] == demo_layer_sizes(ASD_WNT_DELTA)
    assert first["train_loss"] == again["train_loss"]


def test_page_opens_with_mechanism_pane():
    app = new_app()
    assert not app.exception
    assert app.title[0].value
    assert len(app.get("image")) >= 2
    assert listed_sizes(app) == demo_layer_sizes(0)


def test_default_preset_is_control():
    assert DEFAULT_PRESET == "Control"
    assert PRESETS[DEFAULT_PRESET] == (0.0, 0.0)
    app = new_app()
    assert not app.exception
    assert app.radio(key="preset").value == "Control"
    assert app.slider(key="wnt_delta").value == 0
    assert app.slider(key="ccl5_delta").value == 0
    assert listed_sizes(app) == [100, 100, 100, 100, 100, 4]
    assert listed_sizes(app) == demo_layer_sizes(0)
    captions = [item.value for item in app.caption]
    assert DIAGRAM_SHAPE_CAPTION in captions


def test_presets_set_sliders():
    app = new_app()
    assert not app.exception
    app.radio(key="preset").set_value("Wnt + CCL5").run()
    assert not app.exception
    assert app.slider(key="wnt_delta").value == pytest.approx(ASD_WNT_DELTA)
    assert app.slider(key="ccl5_delta").value == pytest.approx(ASD_CCL5_DELTA)
    assert listed_sizes(app) == demo_layer_sizes(ASD_WNT_DELTA)
    app.radio(key="preset").set_value("Control").run()
    assert app.slider(key="wnt_delta").value == 0
    assert app.slider(key="ccl5_delta").value == 0


def test_wnt_slider_grows_hidden_layers():
    app = new_app()
    assert not app.exception
    before = listed_sizes(app)
    app.slider(key="wnt_delta").set_value(ASD_WNT_DELTA).run()
    assert not app.exception
    after = listed_sizes(app)
    assert after == demo_layer_sizes(ASD_WNT_DELTA)
    assert all(a > b for a, b in zip(after[1:-1], before[1:-1]))


@needs_images
def test_tiny_live_run():
    app = new_app()
    app.number_input[0].set_value(2).run()
    next(b for b in app.button if b.label.startswith("Run four")).click().run()
    assert not app.exception
    assert "live_results" in app.session_state
    assert set(app.session_state["live_results"]) == {"control", "ccl5", "wnt", "both"}
