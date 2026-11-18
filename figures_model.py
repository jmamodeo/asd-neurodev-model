"""Figures that describe the model itself and need no training runs: the network schematic and the release-map
distributions. Both are generated programmatically (numbers come from the same code and constants the model uses)."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

import experiment as ex
from analysis import PAGE_WIDTH   # also applies the shared font settings
from network_tuning import get_release_maps

CCL5_DELTA = ex.CONDITIONS["ccl5"][0]
WNT_DELTA = ex.CONDITIONS["wnt"][1]
INPUT_SIZE = ex.IMAGE_VECTOR_SIZE
HIDDEN_LAYERS = ex.HIDDEN_LAYERS
CLASSES = 4
C_CTRL, C_CCL5, C_WNT = "#444444", "#1f77b4", "#d62728"


def layer_sizes(wnt_delta):
    """Layer sizes exactly as NeuralNetwork.__init__ builds them."""
    sizes = [INPUT_SIZE]
    for _ in range(HIDDEN_LAYERS):
        sizes.append(sizes[-1] + int(sizes[-1] * wnt_delta))
    return sizes + [CLASSES]


FIG_HEIGHT = {"model_schematic": 4.3, "release_map_distributions": 3.9}   # inches


def figure_model_schematic(path, height=None):
    rows = [("Control and CCL5", layer_sizes(0.0), C_CTRL), ("Wnt and Wnt + CCL5", layer_sizes(WNT_DELTA), C_WNT)]
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, height or FIG_HEIGHT["model_schematic"]))
    ax.set_xlim(-0.2, 11.6); ax.set_ylim(-1.3, 8.0); ax.axis("off")
    names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"]
    notes = ["30 × 30 pixels"] + ["sigmoid"] * 4 + ["4 classes, softmax"]
    xs = np.linspace(1.4, 10.2, 6); w = 0.9
    unit = 2.1 / max(layer_sizes(WNT_DELTA)[:-1])          # height per node, same scale in both rows
    for x, n, t in zip(xs, names, notes):
        ax.text(x, 7.55, n, fontsize=8.5, ha="center", va="bottom", fontweight="bold")
        ax.text(x, 7.25, t, fontsize=8, ha="center", va="bottom", color="#555555")
    for r, (label, sizes, col) in enumerate(rows):
        yc = 5.0 - r * 3.7
        ax.text(-0.1, yc + 1.45, label, fontsize=9.5, fontweight="bold", color=col, ha="left", va="center")
        hts = [s * unit for s in sizes[:-1]] + [0.34]
        for i in range(5):                                   # weight matrix W_{i+1}, scaled elementwise by its release map M
            x0, x1 = xs[i] + w / 2, xs[i + 1] - w / 2
            ax.add_patch(Polygon([(x0, yc - hts[i] / 2), (x0, yc + hts[i] / 2), (x1, yc + hts[i + 1] / 2), (x1, yc - hts[i + 1] / 2)],
                                 closed=True, fc="#e6e6e6", ec="none", zorder=1))
            xm = (x0 + x1) / 2
            ax.add_patch(Circle((xm, yc), 0.27, fc="white", ec=C_CCL5, lw=1.6, zorder=3))
            ax.text(xm, yc, "×M", fontsize=8, color=C_CCL5, ha="center", va="center", zorder=4, fontweight="bold")
            ax.text(xm, yc - 0.42, f"W{i + 1}", fontsize=8, color="#555555", ha="center", va="top", zorder=4)
        for i, (x, h, s) in enumerate(zip(xs, hts, sizes)):
            wide = r == 1 and 0 < i < 5
            ax.add_patch(FancyBboxPatch((x - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="#f4d9d9" if wide else "#dcdcdc", ec=C_WNT if wide else "#555555", lw=1.4, zorder=2))
            ax.text(x, yc, f"{s}", fontsize=8.5, ha="center", va="center", zorder=5)
    ax.text(0.0, -0.35, "×M  release map multiplying each weight matrix W elementwise (forward and backward pass)", fontsize=8, color=C_CCL5, ha="left", va="top")
    ax.text(0.0, -0.95, f"Wnt: each hidden layer has 1 + ΔWnt = {1 + WNT_DELTA:g} times the nodes of the previous one", fontsize=8, color=C_WNT, ha="left", va="top")
    fig.tight_layout(pad=0.2); fig.savefig(path, dpi=200); plt.close(fig)


def sample_release_values(delta, n=1_000_000, seed=0):
    """Release-map entries drawn with the model's own sampling function (network_tuning.get_release_maps)."""
    np.random.seed(seed)
    side = int(np.sqrt(n))
    return get_release_maps([np.zeros((side, side))], delta)[0].ravel()


def figure_release_maps(path, height=None):
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, height or FIG_HEIGHT["release_map_distributions"]))
    stats_txt = []
    for delta, col, name in ((0.0, C_CTRL, "Control (ΔCCL5 = 0)"), (CCL5_DELTA, C_CCL5, f"CCL5 (ΔCCL5 = {CCL5_DELTA})")):
        v = sample_release_values(delta)
        k = np.round(v * 100).astype(int)
        counts = np.bincount(k) / len(k)
        ax.bar(np.arange(len(counts)) / 100, counts, width=0.01, color=col, alpha=.55, label=name, lw=0)
        m, s = v.mean(), v.std()
        ax.axvline(m, color=col, lw=1.2, ls="--")
        stats_txt.append((col, f"{name}\nmean {m:.3f}, SD {s:.3f}, CV {s / m:.3f}"))
    for i, (col, t) in enumerate(stats_txt):
        ax.text(0.98, 0.95 - i * 0.2, t, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=col)
    ax.set_xlabel("synaptic gain (release-map value)"); ax.set_ylabel("fraction of synapses")
    ax.set_xlim(0.5, 2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
