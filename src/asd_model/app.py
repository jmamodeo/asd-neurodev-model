"""Streamlit demo of the Wnt/CCL5 model: a live mechanism view, the published results and a small live training run.

Started from the repository root with ``streamlit run app.py``. The helpers above ``main`` do not need Streamlit
to be running, so they can be tested directly.
"""

import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from asd_model.experiment import CONDITIONS, RELEASE_MAP_SEED_OFFSET, SHUFFLE_SEED_OFFSET
from asd_model.network_tuning import POISSON_MEAN_SCALE
from asd_model.neural_network import (
    ASD_CCL5_DELTA,
    ASD_WNT_DELTA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SESSIONS,
    SUPPLEMENT_LEARNING_RATE,
    NeuralNetwork,
)
from asd_model.paper_analysis import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    IMAGE_SET_LABELS,
    LN4,
    per_run_table,
)
from asd_model.reproduce import load_bundle
from asd_model.visual_input import encode_image_labels, image_to_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_IMAGE_DIR = os.path.join(ROOT, "demo_images")
DEMO_IMAGE_VECTOR_SIZE = 100
DEMO_HIDDEN_LAYERS = 4
DEMO_SESSIONS = 50
DEMO_CLASS_NAMES = ["trucks", "dogs", "cats", "cars"]
DEMO_N_CLASSES = len(DEMO_CLASS_NAMES)
MIN_POOL_SIZE = 500_000  # floats; about 2 MB at float32
POOL_WEIGHT_FACTOR = 4  # the pool holds at least this many times the number of weights of the network
MAX_DIAGRAM_NODES = 10  # evenly spaced display positions per layer bar
MAX_DIAGRAM_EDGES = 28  # sampled edges drawn between two neighboring layers
PRESETS = {
    "Control": (0.0, 0.0),
    "CCL5": (0.0, ASD_CCL5_DELTA),
    "Wnt": (ASD_WNT_DELTA, 0.0),
    "Wnt + CCL5": (ASD_WNT_DELTA, ASD_CCL5_DELTA),
}  # label: (wnt_delta, ccl5_delta)
PRESET_CONDITIONS = {"Control": "control", "CCL5": "ccl5", "Wnt": "wnt", "Wnt + CCL5": "both"}
LEARNING_RATES = [DEFAULT_LEARNING_RATE, SUPPLEMENT_LEARNING_RATE]
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x only has trapz


def demo_layer_sizes(
    wnt_delta,
    image_vector_size=DEMO_IMAGE_VECTOR_SIZE,
    hidden_layer_number=DEMO_HIDDEN_LAYERS,
    n_classes=DEMO_N_CLASSES,
):
    """Return the layer sizes of the network with the growth rule of ``NeuralNetwork.__init__``.

    Parameters
    ----------
    wnt_delta : float
        Fold change of Wnt above normal; every hidden layer is ``int(previous * wnt_delta)`` units larger.
    image_vector_size : int
        Number of input units.
    hidden_layer_number : int
        Number of hidden layers.
    n_classes : int
        Number of output units.

    Returns
    -------
    list of int
        Input size, the hidden layer sizes and the output size.
    """
    sizes = [image_vector_size]
    for _ in range(hidden_layer_number):
        sizes.append(sizes[-1] + int(sizes[-1] * wnt_delta))
    sizes.append(n_classes)
    return sizes


def demo_images_available():
    """Return True when every class folder of ``demo_images/train_subset`` and ``test_subset`` holds a PNG."""
    for subset in ("train_subset", "test_subset"):
        for name in DEMO_CLASS_NAMES:
            folder = os.path.join(DEMO_IMAGE_DIR, subset, name)
            if not os.path.isdir(folder) or not any(file.lower().endswith(".png") for file in os.listdir(folder)):
                return False
    return True


def _to_float32(image_dict):
    return {
        image_class: [vector.astype(np.float32) for vector in vectors] for image_class, vectors in image_dict.items()
    }


def load_demo_images(image_vector_size=DEMO_IMAGE_VECTOR_SIZE):
    """Load the shipped demo images.

    Parameters
    ----------
    image_vector_size : int
        Length of each image vector.

    Returns
    -------
    train_images : dict
        Class name to list of float32 image vectors (training set).
    test_images : dict
        Class name to list of float32 image vectors (held-out set).
    labels : dict
        Class name to float32 one-hot label.
    """
    train_images = _to_float32(
        image_to_matrix(os.path.join(DEMO_IMAGE_DIR, "train_subset"), DEMO_CLASS_NAMES, image_vector_size)
    )
    test_images = _to_float32(
        image_to_matrix(os.path.join(DEMO_IMAGE_DIR, "test_subset"), DEMO_CLASS_NAMES, image_vector_size)
    )
    labels = {key: value.astype(np.float32) for key, value in encode_image_labels(train_images).items()}
    return train_images, test_images, labels


def demo_pool_size(layer_sizes):
    """Return the number of floats in the release-map pool for a network (a few times its weight count)."""
    n_weights = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
    return max(MIN_POOL_SIZE, POOL_WEIGHT_FACTOR * n_weights)


def make_demo_release_sampler(ccl5_delta, seed, pool_size=MIN_POOL_SIZE):
    """Return a release-map sampler (same signature as ``get_release_maps``) that reads windows of a small pool.

    The pool is drawn once from Poisson(100 * (1 + ccl5_delta)) / 100 as float32; each call returns one random
    window of the pool, reshaped to the shape of every weight matrix.
    """
    rng = np.random.default_rng(seed)
    pool = (rng.poisson(POISSON_MEAN_SCALE * (1 + ccl5_delta), size=pool_size) / POISSON_MEAN_SCALE).astype(np.float32)

    def sampler(layer_weights, _ccl5_delta):
        release_maps = []
        for weights in layer_weights:
            if weights.size > pool_size:
                raise ValueError("The release-map pool is smaller than a weight matrix.")
            offset = int(rng.integers(0, pool_size - weights.size + 1))
            release_maps.append(pool[offset : offset + weights.size].reshape(weights.shape))
        return release_maps

    return sampler


def area_under_training_loss(train_loss):
    """Return the area under the per-session training-loss curve (trapezoid rule, one unit per session)."""
    return float(_trapezoid(np.asarray(train_loss, dtype=float)))


def run_live_condition(
    ccl5_delta,
    wnt_delta,
    sessions,
    learning_rate,
    seed,
    train_images,
    test_images,
    labels,
    progress=None,
):
    """Train one seeded network and record its learning curves.

    The initial weights use ``seed``; the shuffle order uses ``SHUFFLE_SEED_OFFSET + seed`` and the release maps
    ``RELEASE_MAP_SEED_OFFSET + seed``, so a seed gives the same result every time.

    Parameters
    ----------
    ccl5_delta, wnt_delta : float
        Fold changes of CCL5 and Wnt above normal.
    sessions : int
        Number of training sessions.
    learning_rate : float
        Learning rate.
    seed : int
        Seed of the run.
    train_images, test_images : dict
        Class name to list of image vectors.
    labels : dict
        Class name to one-hot label.
    progress : callable, optional
        Called with the number of finished sessions after each session.

    Returns
    -------
    dict
        ``train_loss``, ``val_loss`` and ``val_acc`` per session, ``layer_sizes``, ``final_train_loss``,
        ``final_val_loss``, ``final_val_acc`` and ``seconds``.
    """
    start_time = time.time()
    image_vector_size = len(next(iter(train_images.values()))[0])
    np.random.seed(seed)
    net = NeuralNetwork(
        DEMO_CLASS_NAMES, image_vector_size, DEMO_HIDDEN_LAYERS, ccl5_delta=ccl5_delta, wnt_delta=wnt_delta
    )
    net.layer_weights = [weights.astype(np.float32) for weights in net.layer_weights]
    net.layer_biases = [biases.astype(np.float32) for biases in net.layer_biases]
    net.release_map_fn = make_demo_release_sampler(
        ccl5_delta, RELEASE_MAP_SEED_OFFSET + seed, demo_pool_size(net.layer_sizes)
    )
    np.random.seed(SHUFFLE_SEED_OFFSET + seed)
    train_loss, val_loss, val_acc = [], [], []
    for session in range(sessions):
        losses, _ = net.run_session(train_images, labels, learning_rate)
        train_loss.append(float(np.mean(losses)))
        session_val_loss, session_val_acc = net.evaluate(test_images, labels, True)
        val_loss.append(float(session_val_loss))
        val_acc.append(float(session_val_acc))
        if progress is not None:
            progress(session + 1)
    return dict(
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        layer_sizes=list(net.layer_sizes),
        final_train_loss=train_loss[-1],
        final_val_loss=val_loss[-1],
        final_val_acc=val_acc[-1],
        seconds=time.time() - start_time,
    )


def run_live_comparison(sessions, learning_rate, seed, progress=None):
    """Train the four paper conditions at the paper ASD deltas on the demo images.

    Parameters
    ----------
    sessions : int
        Number of training sessions.
    learning_rate : float
        Learning rate.
    seed : int
        Seed shared by the four conditions.
    progress : callable, optional
        Called with the fraction (0 to 1) of the work that is done.

    Returns
    -------
    dict
        Result of ``run_live_condition`` for each of control, ccl5, wnt and both.
    """
    train_images, test_images, labels = load_demo_images()
    results = {}
    for index, condition in enumerate(CONDITION_ORDER):
        ccl5_delta, wnt_delta = CONDITIONS[condition]

        def session_progress(done, index=index):
            if progress is not None:
                progress((index + done / sessions) / len(CONDITION_ORDER))

        results[condition] = run_live_condition(
            ccl5_delta, wnt_delta, sessions, learning_rate, seed, train_images, test_images, labels, session_progress
        )
    return results


def live_metrics_table(results):
    """Return a table of AUC, final training loss and final held-out accuracy for the live conditions."""
    rows = []
    for condition in CONDITION_ORDER:
        if condition in results:
            result = results[condition]
            rows.append(
                {
                    "Condition": CONDITION_LABELS[condition],
                    "Layer sizes": ", ".join(str(size) for size in result["layer_sizes"]),
                    "AUC (train loss)": round(area_under_training_loss(result["train_loss"]), 2),
                    "Final train loss": round(result["final_train_loss"], 3),
                    "Held-out accuracy (%)": round(result["final_val_acc"], 1),
                    "Seconds": round(result["seconds"], 1),
                }
            )
    return pd.DataFrame(rows)


def plot_live_loss_curves(results):
    """Return a figure with training and held-out loss per session, one line per live condition."""
    fig = Figure(figsize=(9, 3.6))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, key, name in zip(axes, ("train_loss", "val_loss"), ("Training loss", "Held-out loss")):
        for condition in CONDITION_ORDER:
            if condition in results:
                curve = results[condition][key]
                ax.plot(
                    np.arange(1, len(curve) + 1),
                    curve,
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition],
                    lw=2,
                )
        ax.axhline(LN4, color="k", lw=0.8, ls=":", label="Chance (ln 4)")
        ax.set_title(name)
        ax.set_xlabel("training session")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _line_style(condition, highlight):
    """Return (line width, alpha) so the highlighted condition stands out."""
    if highlight is None:
        return 2.0, 1.0
    return (3.2, 1.0) if condition == highlight else (1.3, 0.45)


def _runs_for(runs, image_set):
    return [run for run in runs if run["data"] == image_set]


def plot_paper_loss_curves(runs, image_set, highlight=None):
    """Return a figure with the mean training and held-out loss (95% interval over seeds) for one image set.

    Parameters
    ----------
    runs : list of dict
        Bundled runs from ``load_bundle``.
    image_set : str
        ``"cifar3"`` or ``"original"``.
    highlight : str, optional
        Condition name drawn thicker than the others.
    """
    selected = _runs_for(runs, image_set)
    fig = Figure(figsize=(9, 3.6))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, key, name in zip(axes, ("train_loss", "val_loss"), ("Training loss", "Held-out loss")):
        for condition in CONDITION_ORDER:
            curves = np.array([run[key] for run in selected if run["cond"] == condition])
            if len(curves) == 0:
                continue
            mean = curves.mean(0)
            sessions = np.arange(1, curves.shape[1] + 1)
            width, alpha = _line_style(condition, highlight)
            ax.plot(
                sessions,
                mean,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                lw=width,
                alpha=alpha,
            )
            if len(curves) > 1:
                half_width = stats.t.ppf(0.975, len(curves) - 1) * curves.std(0, ddof=1) / np.sqrt(len(curves))
                ax.fill_between(
                    sessions,
                    mean - half_width,
                    mean + half_width,
                    color=CONDITION_COLORS[condition],
                    alpha=0.15 * alpha,
                )
        ax.axhline(LN4, color="k", lw=0.8, ls=":")
        ax.set_title(f"{name} ({IMAGE_SET_LABELS.get(image_set, image_set)})")
        ax.set_xlabel("training session")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_paper_sessions_to_criterion(runs, image_set, highlight=None):
    """Return a figure with the fraction of networks that reached the chance-level loss by each session."""
    per_run = per_run_table(_runs_for(runs, image_set))
    n_sessions = max(run["sessions"] for run in _runs_for(runs, image_set))
    fig = Figure(figsize=(5.5, 3.4))
    ax = fig.subplots()
    session_axis = np.arange(0, n_sessions + 1)
    for condition in CONDITION_ORDER:
        first_sessions = per_run[per_run.cond == condition].ept_abs.values
        if len(first_sessions) == 0:
            continue
        fraction = np.array([(first_sessions <= session).mean() for session in session_axis])
        median = np.median(first_sessions)
        label = (
            f"{CONDITION_LABELS[condition]}: {median:g}"
            if median <= n_sessions
            else f"{CONDITION_LABELS[condition]}: not reached"
        )
        width, alpha = _line_style(condition, highlight)
        ax.step(
            session_axis, fraction, where="post", color=CONDITION_COLORS[condition], lw=width, alpha=alpha, label=label
        )
    ax.set_xlabel("training session")
    ax.set_ylabel("fraction reaching chance-level loss")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", title="median session", fontsize=8)
    fig.tight_layout()
    return fig


def paper_summary_table(runs, image_set):
    """Return mean AUC, final loss, median sessions to chance-level loss and held-out accuracy per condition."""
    per_run = per_run_table(_runs_for(runs, image_set))
    n_sessions = max(run["sessions"] for run in _runs_for(runs, image_set))
    rows = []
    for condition in CONDITION_ORDER:
        subset = per_run[per_run.cond == condition]
        if subset.empty:
            continue
        median = float(np.median(subset.ept_abs))
        rows.append(
            {
                "Condition": CONDITION_LABELS[condition],
                "Networks": len(subset),
                "AUC (train loss)": round(float(subset.auc.mean()), 3),
                "Final train loss": round(float(subset.final_loss.mean()), 3),
                "Sessions to chance loss (median)": f"{median:g}" if median <= n_sessions else "not reached",
                "Held-out accuracy (%)": round(float(subset.final_val_acc.mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def plot_release_histogram(ccl5_delta, seed=0, n_samples=20_000):
    """Return a histogram of release-map values for ``ccl5_delta`` next to the control distribution.

    Values are Poisson(100 * (1 + delta)) / 100, the formula of ``get_release_maps``.
    """
    rng = np.random.default_rng(seed)
    control = rng.poisson(POISSON_MEAN_SCALE, size=n_samples) / POISSON_MEAN_SCALE
    current = rng.poisson(POISSON_MEAN_SCALE * (1 + ccl5_delta), size=n_samples) / POISSON_MEAN_SCALE
    upper = max(1.5, float(current.max()) + 0.05)
    bins = np.linspace(0.4, upper, 60)
    fig = Figure(figsize=(5.5, 3.2))
    ax = fig.subplots()
    ax.hist(control, bins=bins, color=CONDITION_COLORS["control"], alpha=0.5, label="Control (delta = 0)")
    ax.hist(current, bins=bins, color=CONDITION_COLORS["ccl5"], alpha=0.6, label=f"CCL5 delta = {ccl5_delta:g}")
    ax.axvline(1 + ccl5_delta, color=CONDITION_COLORS["ccl5"], lw=1.2, ls="--")
    ax.set_xlabel("release-map value (multiplies each weight)")
    ax.set_ylabel("number of synapses (of 20,000)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_network_diagram(
    wnt_delta,
    ccl5_delta,
    image_vector_size=DEMO_IMAGE_VECTOR_SIZE,
    hidden_layer_number=DEMO_HIDDEN_LAYERS,
    n_classes=DEMO_N_CLASSES,
    seed=0,
):
    """Return a diagram of the network for the current Wnt and CCL5 settings (no training involved).

    Each layer is a vertical bar whose height follows the layer's unit count (the Wnt growth rule). A sample of
    connections between neighboring layers is colored by release-map values drawn from
    Poisson(100 * (1 + ccl5_delta)) / 100 with a fixed seed.
    """
    sizes = demo_layer_sizes(wnt_delta, image_vector_size, hidden_layer_number, n_classes)
    names = ["Input", *[f"Hidden {i + 1}" for i in range(hidden_layer_number)], "Output"]
    reference = max(max(sizes), 4 * image_vector_size)
    heights = [max(size / reference, 0.04) * 8.0 for size in sizes]
    rng = np.random.default_rng(seed)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.6, vmax=1.8)
    cmap = "coolwarm"
    fig = Figure(figsize=(10, 5.2))
    ax = fig.add_axes((0.02, 0.02, 0.88, 0.88))
    x_positions = np.arange(len(sizes)) * 2.0
    bar_width = 0.5

    ports = []
    for height, size in zip(heights, sizes):
        n_display = min(size, MAX_DIAGRAM_NODES)
        ys = np.linspace(-height / 2, height / 2, n_display + 2)[1:-1] if n_display > 1 else np.array([0.0])
        ports.append(ys)
    for index in range(len(sizes) - 1):
        n_edges = min(MAX_DIAGRAM_EDGES, len(ports[index]) * len(ports[index + 1]))
        pairs = rng.choice(len(ports[index]) * len(ports[index + 1]), size=n_edges, replace=False)
        gains = rng.poisson(POISSON_MEAN_SCALE * (1 + ccl5_delta), size=n_edges) / POISSON_MEAN_SCALE
        for pair, gain in zip(pairs, gains):
            source, target = divmod(int(pair), len(ports[index + 1]))
            ax.plot(
                [x_positions[index] + bar_width / 2, x_positions[index + 1] - bar_width / 2],
                [ports[index][source], ports[index + 1][target]],
                color=ScalarMappable(norm=norm, cmap=cmap).to_rgba(gain),
                lw=1.0,
                alpha=0.8,
                zorder=1,
            )
    for x, height, size, name in zip(x_positions, heights, sizes, names):
        ax.add_patch(
            FancyBboxPatch(
                (x - bar_width / 2, -height / 2),
                bar_width,
                height,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor="#1F4E79",
                edgecolor="#17202A",
                alpha=0.9,
                zorder=2,
            )
        )
        ax.text(x, height / 2 + 0.25, name, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x, -height / 2 - 0.25, f"{size} units", ha="center", va="top", fontsize=9)
    ax.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)
    ax.set_ylim(-5.4, 5.4)
    ax.axis("off")
    ax.set_title(
        f"Bar height follows the Wnt layer-growth rule (delta = {wnt_delta:g}); "
        f"edge color follows the CCL5 release map (delta = {ccl5_delta:g})",
        fontsize=10,
    )
    colorbar_axes = fig.add_axes((0.92, 0.2, 0.015, 0.6))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axes)
    colorbar.set_label("synaptic gain (release-map value)", fontsize=9)
    return fig


@st.cache_data(show_spinner=False)
def _cached_bundle(learning_rate):
    return load_bundle(learning_rate)


def _apply_preset():
    wnt_delta, ccl5_delta = PRESETS[st.session_state["preset"]]
    st.session_state["wnt_delta"] = wnt_delta
    st.session_state["ccl5_delta"] = ccl5_delta


def _show(fig):
    st.pyplot(fig, clear_figure=True)


def _mechanism_pane(wnt_delta, ccl5_delta):
    st.header("1. Mechanism")
    st.caption("Moves with the sliders in the sidebar; nothing is trained here.")
    _show(plot_network_diagram(wnt_delta, ccl5_delta))
    sizes = demo_layer_sizes(wnt_delta)
    st.markdown("Layer sizes: " + ", ".join(str(size) for size in sizes))
    st.markdown(f"Hidden synapses: {sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)):,}")
    _show(plot_release_histogram(ccl5_delta))


def _paper_pane(highlight):
    st.header("2. Paper results")
    st.caption("Loaded from the bundles in reproduce_data/: 20 seeds, 900-wide input, 100 sessions per network.")
    left, right = st.columns(2)
    learning_rate = left.radio("Learning rate", LEARNING_RATES, horizontal=True, key="paper_lr")
    image_set = right.radio(
        "Image set",
        ("cifar3", "original"),
        format_func=IMAGE_SET_LABELS.get,
        horizontal=True,
        key="paper_set",
    )
    if learning_rate == SUPPLEMENT_LEARNING_RATE:
        st.caption("A learning rate of 0.1 can make the network unstable; it is the paper's supplementary setting.")
    runs = _cached_bundle(learning_rate)
    if not _runs_for(runs, image_set):
        st.warning("No bundled results for this choice.")
        return
    _show(plot_paper_loss_curves(runs, image_set, highlight))
    st.dataframe(paper_summary_table(runs, image_set), hide_index=True)
    _show(plot_paper_sessions_to_criterion(runs, image_set, highlight))


def _live_pane(sessions, learning_rate, seed, images_ready):
    st.header("3. Live training")
    st.caption(
        f"A {DEMO_IMAGE_VECTOR_SIZE}-wide, {DEMO_HIDDEN_LAYERS}-hidden-layer network trained for {sessions} sessions "
        "on the 24 shipped CIFAR-10 images. It is an illustration, not the paper's 900-wide, 20-seed runs."
    )
    if not images_ready:
        st.error("The demo images are missing. Add the PNG files to demo_images/ (see demo_images/README.md).")
    if st.button("Run four conditions", type="primary", disabled=not images_ready):
        bar = st.progress(0.0)
        with st.spinner("Training four networks..."):
            st.session_state["live_results"] = run_live_comparison(sessions, learning_rate, seed, bar.progress)
        bar.empty()
        st.session_state["live_settings"] = (sessions, learning_rate, seed)
    results = st.session_state.get("live_results")
    if results:
        sessions_done, rate_done, seed_done = st.session_state["live_settings"]
        st.caption(f"Showing the last run: {sessions_done} sessions, learning rate {rate_done:g}, seed {seed_done}.")
        _show(plot_live_loss_curves(results))
        st.dataframe(live_metrics_table(results), hide_index=True)


def main():
    """Draw the demo page."""
    st.set_page_config(page_title="ASD Neurodevelopment Model", layout="wide", initial_sidebar_state="expanded")
    st.title("Wnt and CCL5 in a learning network")
    st.caption(
        "Elevated Wnt grows the hidden layers (extra synapses); elevated CCL5 scales the weights by stochastic "
        "release maps; both can slow learning. The live network is a smaller illustration; the paper pane shows "
        "the published 20-seed results."
    )
    st.session_state.setdefault("wnt_delta", 0.0)
    st.session_state.setdefault("ccl5_delta", 0.0)
    with st.sidebar:
        st.header("Controls")
        preset = st.radio("Preset", list(PRESETS), key="preset", on_change=_apply_preset)
        wnt_delta = st.slider("Wnt delta", 0.0, 1.0, step=0.00001, format="%.5f", key="wnt_delta")
        st.caption(f"ASD = {ASD_WNT_DELTA}")
        ccl5_delta = st.slider("CCL5 delta", 0.0, 1.0, step=0.00001, format="%.5f", key="ccl5_delta")
        st.caption(f"ASD = {ASD_CCL5_DELTA}")
        st.subheader("Live training")
        learning_rate = st.radio("Learning rate", LEARNING_RATES, key="live_lr", horizontal=True)
        if learning_rate == SUPPLEMENT_LEARNING_RATE:
            st.caption("0.1 is the paper supplement and can be unstable.")
        sessions = st.number_input("Sessions", min_value=2, max_value=DEFAULT_SESSIONS, value=DEMO_SESSIONS, step=1)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
    _mechanism_pane(wnt_delta, ccl5_delta)
    _paper_pane(PRESET_CONDITIONS[preset])
    _live_pane(int(sessions), learning_rate, int(seed), demo_images_available())
