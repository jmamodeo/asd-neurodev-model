# ASD Neurodevelopment Model

A Python neural-network model of how elevated Wnt and mTOR-activated CCL5 signaling in autism spectrum disorder (ASD) could affect visual image classification.

## Background

ASD is a heterogeneous neurodevelopmental disorder, and vision is one of the senses particularly affected. Two pathways, Wnt and mTOR, have been identified as high risk. Wnt proteins stabilize synapses, and mTOR activates CCL5, which stimulates glutamate release. In excess, these proteins cause unnecessary synapses to form and survive, a hallmark of ASD.

The paper uses visual processing as a base to examine the effect of both proteins at ASD levels in a neural network, as a way to study population dynamics rather than only single cells. The revised paper (November 2026, `docs/paper/paper.pdf`) reports that, in a corrected and re-analyzed version of the model, both perturbations slow the reduction of training loss, that the two together slow it more than the sum of their separate effects, and that held-out classification accuracy stays near chance because the image sets are very small. The first version's classification claim does not hold; see `docs/paper/CHANGELOG.md`.

## How the model works

- **Input.** `visual_input.py` reads each image, converts it to grayscale, resizes it to a square whose side is `sqrt(image_vector_size)`, scales pixel values to 0-1, and flattens it into a vector of length `image_vector_size`. This vector is the input layer.
- **Layer sizes.** Hidden layers start equal in size to the input layer. With `wnt_delta`, each hidden layer gains `int(previous_size * wnt_delta)` nodes over the previous layer, modeling extra synapses (more connections and redundancy). The output layer has one node per class.
- **CCL5 / glutamate release.** For each forward pass, `network_tuning.py` draws a release map for every weight matrix from a Poisson distribution with mean `100 * (1 + ccl5_delta)`, divided by 100. The weights are multiplied by these maps, modeling stochastic glutamate release.
- **Learning.** Hidden layers use sigmoid activations and the output layer uses a softmax; loss is cross-entropy. Gradients are clipped to a maximum norm of 1.0. Each session presents all training images once in a freshly shuffled, class-mixed order. The default learning rate is **0.03** (the original code used 0.1, which makes the corrected network unstable; use `--lr 0.1` to select it).
- **Release maps in training and testing.** The release maps multiply the weights in both the forward and the backward pass, and the same distribution is used for training and testing (`use_release_maps` defaults to `True`).
- **Accuracy** is the highest-probability output node compared with the label, on held-out test images (chance = 25% with four classes).

Values used in the paper for ASD: `wnt_delta` = 0.41859 and `ccl5_delta` = 0.40467 (fold changes of 1.41859 and 1.40467 relative to normal). The paper's runs used `image_vector_size` 900 and 100 sessions, whereas `main.py` defaults to an `image_vector_size` of 400.

## Installation

Python 3.11 (the version used in the paper). `requirements.txt` pins package versions that were verified to run `main.py` end to end on Python 3.11; newer seaborn (0.13+) and pandas (3.x) releases break the training-loss plot.

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

The repository does not ship images. `main.py` expects:

```
image_set/
  train_subset/<class>/
  test_subset/<class>/
```

Class folder names must match `class_list` in `main.py` (default: `trucks`, `dogs`, `cats`, `cars`).

**Option A: CIFAR-10.** Run

```
python scripts/fetch_cifar10.py
```

This downloads CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html (about 163 MB) and writes PNGs, mapping automobile to `cars`, truck to `trucks`, cat to `cats`, and dog to `dogs`. Use `--train-per-class` (default 25) and `--test-per-class` (default 10) to change the subset size.

Citation: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.

**Option B: your own images.** Put images (any format OpenCV can read) in the class folders. Only image files may be present; hidden files such as `.DS_Store` will break loading.

The original paper used a small set of web images that are not redistributed here.

## Running

From the repository root:

```
python main.py
```

Parameters set in `main.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `image_vector_size` | 400 | Input layer size (image vector length) |
| `class_list` | `['trucks', 'dogs', 'cats', 'cars']` | Image classes; must match folder names |
| `hidden_layer_number` | 4 | Number of hidden layers |
| `ccl5_delta` | 0 | Fold difference in CCL5 relative to normal (e.g., 0.4 for 1.4 fold) |
| `wnt_delta` | 0 | Fold difference in Wnt relative to normal |
| `learn_r` | 0.03 (`--lr`) | Learning rate (the original code used 0.1) |
| `sessions` | 100 | Passes over the full training set |
| `use_release_maps` | `True` | Release-map weighting at test time (always on in training) |

The paper compares four conditions, which can be set as examples:

| Condition | `wnt_delta` | `ccl5_delta` |
|---|---|---|
| Control | 0 | 0 |
| CCL5 only | 0 | 0.40467 |
| Wnt only | 0.41859 | 0 |
| Both | 0.41859 | 0.40467 |

## Outputs

- The loss for each training session is printed.
- Held-out accuracy (output-based) is printed.
- `results/<YYYY-MM-DD HHhMMm> results/` contains `training_loss_plot.png` and `training_loss_table.xlsx`.

`main.py` sets no random seed, so runs differ; `reproduce.py` uses fixed seeds.

## Reproduce the figures

One command regenerates every figure and table in the paper (main configuration, learning rate 0.03, and the supplementary learning rate 0.1):

```
pip install -r requirements.txt
python reproduce.py            # or ./reproduce.sh, or: make figures
```

Output: `figures/` (Figures 1-3 and Supplementary Figures S1-S2 as `loss_curves_lr0.03.png`, `auc_interaction_lr0.03.png`, `summary_lr0.03.png`, `loss_curves_lr0.1.png`, `auc_interaction_lr0.1.png`, `summary_lr0.1.png`) and `tables/` (per-run values, all statistics as JSON, summary tables as Markdown).

- The repository ships the finished runs (`reproduce_data/runs_lr0.03.json.gz`, `runs_lr0.1.json.gz`: 4 conditions x 20 seeds x 2 image sets each). `python reproduce.py` therefore **re-analyzes and re-plots from those runs (about 15 seconds)**; it reproduces the numbers in the paper exactly (the statistics were verified to agree with the original analysis to better than 1e-12).
- `python reproduce.py --retrain` ignores the shipped runs and retrains everything from scratch on the CIFAR-10 subset (downloaded automatically with `scripts/fetch_cifar10.py`, about 163 MB). Each CIFAR-10 configuration is 80 networks x about 8 CPU-minutes for the Wnt conditions (and under a minute for the others), roughly 2-3 hours per learning rate on an 8-core machine (`--workers N`). The original images are not redistributed; to retrain on them as well pass `--original-images DIR` (folders `train_subset/` and `test_subset/` with `trucks`, `dogs`, `cats`, `cars`). Without them, the shipped runs still provide the original-image figures.
- `python reproduce.py --quick` is a smoke test (3 seeds, 20 sessions, CIFAR-10 only, about 1-2 minutes after the download) writing to `figures_quick/` and `tables_quick/`.
- Seeds are fixed. On the same machine and library versions a retrained run reproduces the shipped run **bit for bit**; across different numpy/BLAS builds the first session agrees to about 1e-4 but float32 rounding differences are amplified by the training dynamics, so individual curves differ while the statistics agree (see `docs/paper/CHANGELOG.md` and the verification note in `docs/paper/README.md`).
- Rebuild the paper PDF with `pip install -r docs/paper/requirements.txt && python docs/paper/build_paper.py` (numbers are filled in from `tables/`).

## Citation

Marri, R. & Amodeo, J. (2023; revised 2026). Modeling the effects of the Wnt and mTOR signaling pathways on ASD-impacted visual processing using neural networks to reflect population dynamics. [docs/paper/paper.pdf](docs/paper/paper.pdf) (revised November 2026; source in `docs/paper/paper.md`)

```bibtex
@misc{marri2023wntmtor,
  author = {Marri, Rithvik and Amodeo, John},
  title  = {Modeling the effects of the Wnt and mTOR signaling pathways on ASD-impacted visual processing using neural networks to reflect population dynamics},
  year   = {2023}
}
```

## Limitations

From the revised paper's Limitations section:

- Compute limited the work to 100 training sessions and 900-node layers.
- Held-out accuracy is near chance in all four conditions because the image sets contain only three training and three test images per class.
- As a computational model, it cannot account for environmental factors such as behavioral therapy or specialized education.

## License

MIT, see [LICENSE](LICENSE). The license covers the code; the paper PDF remains the authors' work.
