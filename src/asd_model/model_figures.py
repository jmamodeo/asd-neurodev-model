"""Figures that describe the model itself and need no training runs.

These are the network schematic and the release-map distributions. Both are generated programmatically (numbers come
from the same code and constants the model uses).
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend has to be chosen before pyplot is imported)
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

from asd_model import experiment  # noqa: E402
from asd_model.network_tuning import get_release_maps  # noqa: E402
from asd_model.paper_analysis import PAGE_WIDTH  # noqa: E402  (importing it also applies the shared font settings)

CCL5_DELTA = experiment.CONDITIONS["ccl5"][0]
WNT_DELTA = experiment.CONDITIONS["wnt"][1]
CONTROL_COLOR, CCL5_COLOR, WNT_COLOR = "#444444", "#1f77b4", "#d62728"
SCHEMATIC_HEIGHTS = {"model_schematic": 4.3, "release_map_distributions": 3.9}  # inches


def layer_sizes(wnt_delta):
    """Return the layer sizes exactly as ``NeuralNetwork.__init__`` builds them for a Wnt delta."""
    sizes = [experiment.IMAGE_VECTOR_SIZE]
    for _ in range(experiment.HIDDEN_LAYERS):
        sizes.append(sizes[-1] + int(sizes[-1] * wnt_delta))
    return sizes + [len(experiment.CLASS_LIST)]


def figure_model_schematic(path, height=None):
    """Draw the network schematic (layer sizes with and without the Wnt increase, release maps on every matrix)."""
    rows = [
        ("Control and CCL5", layer_sizes(0.0), CONTROL_COLOR),
        ("Wnt and Wnt + CCL5", layer_sizes(WNT_DELTA), WNT_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, height or SCHEMATIC_HEIGHTS["model_schematic"]))
    ax.set_xlim(-0.2, 11.6)
    ax.set_ylim(-1.3, 8.0)
    ax.axis("off")
    names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"]
    notes = ["30 × 30 pixels"] + ["sigmoid"] * 4 + ["4 classes, softmax"]
    x_positions = np.linspace(1.4, 10.2, 6)
    box_width = 0.9
    unit = 2.1 / max(layer_sizes(WNT_DELTA)[:-1])  # height per node, same scale in both rows
    for x, name, note in zip(x_positions, names, notes, strict=True):
        ax.text(x, 7.55, name, fontsize=8.5, ha="center", va="bottom", fontweight="bold")
        ax.text(x, 7.25, note, fontsize=8, ha="center", va="bottom", color="#555555")
    for row, (label, sizes, color) in enumerate(rows):
        y_center = 5.0 - row * 3.7
        ax.text(-0.1, y_center + 1.45, label, fontsize=9.5, fontweight="bold", color=color, ha="left", va="center")
        heights = [size * unit for size in sizes[:-1]] + [0.34]
        for i in range(5):  # weight matrix W_{i+1}, scaled elementwise by its release map M
            x_left, x_right = x_positions[i] + box_width / 2, x_positions[i + 1] - box_width / 2
            ax.add_patch(
                Polygon(
                    [
                        (x_left, y_center - heights[i] / 2),
                        (x_left, y_center + heights[i] / 2),
                        (x_right, y_center + heights[i + 1] / 2),
                        (x_right, y_center - heights[i + 1] / 2),
                    ],
                    closed=True,
                    fc="#e6e6e6",
                    ec="none",
                    zorder=1,
                )
            )
            x_middle = (x_left + x_right) / 2
            ax.add_patch(Circle((x_middle, y_center), 0.27, fc="white", ec=CCL5_COLOR, lw=1.6, zorder=3))
            ax.text(
                x_middle,
                y_center,
                "×M",
                fontsize=8,
                color=CCL5_COLOR,
                ha="center",
                va="center",
                zorder=4,
                fontweight="bold",
            )
            ax.text(
                x_middle, y_center - 0.42, f"W{i + 1}", fontsize=8, color="#555555", ha="center", va="top", zorder=4
            )
        for i, (x, box_height, size) in enumerate(zip(x_positions, heights, sizes, strict=True)):
            widened = row == 1 and 0 < i < 5
            ax.add_patch(
                FancyBboxPatch(
                    (x - box_width / 2, y_center - box_height / 2),
                    box_width,
                    box_height,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    fc="#f4d9d9" if widened else "#dcdcdc",
                    ec=WNT_COLOR if widened else "#555555",
                    lw=1.4,
                    zorder=2,
                )
            )
            ax.text(x, y_center, f"{size}", fontsize=8.5, ha="center", va="center", zorder=5)
    ax.text(
        0.0,
        -0.35,
        "×M  release map multiplying each weight matrix W elementwise (forward and backward pass)",
        fontsize=8,
        color=CCL5_COLOR,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        -0.95,
        f"Wnt: each hidden layer has 1 + ΔWnt = {1 + WNT_DELTA:g} times the nodes of the previous one",
        fontsize=8,
        color=WNT_COLOR,
        ha="left",
        va="top",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sample_release_values(delta, n=1_000_000, seed=0):
    """Return release-map entries drawn with the model's own sampling function (``get_release_maps``)."""
    np.random.seed(seed)
    side = int(np.sqrt(n))
    return get_release_maps([np.zeros((side, side))], delta)[0].ravel()


def figure_release_maps(path, height=None):
    """Plot the distribution of release-map values for the control and the CCL5 condition."""
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, height or SCHEMATIC_HEIGHTS["release_map_distributions"]))
    statistics_text = []
    conditions = (
        (0.0, CONTROL_COLOR, "Control (ΔCCL5 = 0)"),
        (CCL5_DELTA, CCL5_COLOR, f"CCL5 (ΔCCL5 = {CCL5_DELTA})"),
    )
    for delta, color, name in conditions:
        values = sample_release_values(delta)
        hundredths = np.round(values * 100).astype(int)
        fractions = np.bincount(hundredths) / len(hundredths)
        ax.bar(np.arange(len(fractions)) / 100, fractions, width=0.01, color=color, alpha=0.55, label=name, lw=0)
        mean, sd = values.mean(), values.std()
        ax.axvline(mean, color=color, lw=1.2, ls="--")
        statistics_text.append((color, f"{name}\nmean {mean:.3f}, SD {sd:.3f}, CV {sd / mean:.3f}"))
    for i, (color, text) in enumerate(statistics_text):
        ax.text(0.98, 0.95 - i * 0.2, text, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=color)
    ax.set_xlabel("synaptic gain (release-map value)")
    ax.set_ylabel("fraction of synapses")
    ax.set_xlim(0.5, 2.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
