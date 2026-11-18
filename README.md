# ASD Neurodevelopment Model

A Python neural-network model of how elevated Wnt and mTOR-activated CCL5 signaling in autism spectrum disorder (ASD) could affect visual image classification.

## Background

ASD is a heterogeneous neurodevelopmental disorder, and vision is one of the senses particularly affected. Two pathways, Wnt and mTOR, have been identified as high risk. Wnt proteins stabilize synapses, and mTOR activates CCL5, which stimulates glutamate release. In excess, these proteins cause unnecessary synapses to form and survive, a hallmark of ASD.

The paper uses visual processing as a base to examine the effect of both proteins at ASD levels in a neural network, as a way to study population dynamics rather than only single cells. The paper ([docs/paper/paper.pdf](docs/paper/paper.pdf)) reports that both perturbations slow the reduction of training loss, that the two together slow it more than the sum of their separate effects, and that held-out classification accuracy stays near chance because the image sets are very small.

## How the model works

- **Input.** `visual_input.py` reads each image, converts it to grayscale, resizes it to a square whose side is `sqrt(image_vector_size)`, scales pixel values to 0-1, and flattens it into a vector of length `image_vector_size`. This vector is the input layer.
- **Layer sizes.** Hidden layers start equal in size to the input layer. With `wnt_delta`, each hidden layer gains `int(previous_size * wnt_delta)` nodes over the previous layer, modeling extra synapses (more connections and redundancy). The output layer has one node per class.
- **CCL5 / glutamate release.** For each forward pass, `network_tuning.py` draws a release map for every weight matrix from a Poisson distribution with mean `100 * (1 + ccl5_delta)`, divided by 100. The weights are multiplied by these maps, modeling stochastic glutamate release.
- **Learning.** Hidden layers use sigmoid activations and the output layer uses a softmax; loss is cross-entropy. Gradients are clipped to a maximum norm of 1.0. Each session presents all training images once in a freshly shuffled, class-mixed order. The default learning rate is **0.03**; `--lr 0.1` selects 0.1, which makes this network unstable (see the paper's supplementary analysis).
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

The paper's own image set (small web images) is not redistributed here.

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
| `learn_r` | 0.03 (`--lr`) | Learning rate |
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

**Everything at once (one command):**

```
pip install -r requirements.txt
python reproduce.py            # or ./reproduce.sh, or: make figures
```

**One figure at a time** (each script runs alone, trains only the runs that figure needs and shares the same run cache, so you can make just one):

| Paper figure | Script | Output |
|---|---|---|
| Figure 1: model schematic (layer widths with and without Wnt, release maps) | `python scripts/figures/fig1_model_schematic.py` | `figures/model_schematic.png` |
| Figure 2: release-map (synaptic gain) distributions, control vs CCL5 | `python scripts/figures/fig2_release_map_distributions.py` | `figures/release_map_distributions.png` |
| Figure 3: training and held-out loss curves (learning rate 0.03) | `python scripts/figures/fig3_loss_curves.py` | `figures/loss_curves_lr0.03.png` |
| Figure 4: sessions to criterion, cumulative fraction of networks reaching the chance-level loss | `python scripts/figures/fig4_sessions_to_criterion.py` | `figures/sessions_to_criterion_lr0.03.png` |
| Figure 5: AUC, Wnt x CCL5 interaction plot | `python scripts/figures/fig5_auc_interaction.py` | `figures/auc_interaction_lr0.03.png` |
| Figure 6: observed vs additive effect (AUC and final loss) | `python scripts/figures/fig6_observed_vs_additive.py` | `figures/observed_vs_additive_lr0.03.png` |
| Figure 7: final loss, sessions to chance-level loss, held-out accuracy | `python scripts/figures/fig7_summary.py` | `figures/summary_lr0.03.png` |
| Supplementary Figure S1: loss curves at learning rate 0.1 | `python scripts/figures/figS1_loss_curves_lr0.1.py` | `figures/loss_curves_lr0.1.png` |
| Supplementary Figure S2: interaction plot at learning rate 0.1 | `python scripts/figures/figS2_auc_interaction_lr0.1.py` | `figures/auc_interaction_lr0.1.png` |
| (extra) summary panels at learning rate 0.1 | `python scripts/figures/figS3_summary_lr0.1.py` | `figures/summary_lr0.1.png` |
| Tables 1-3: all statistics | `python scripts/figures/tables.py` | `tables/statistics_*.json`, `summary_*.md`, `per_run_*.csv` |

`python reproduce.py --only fig3 fig6` does the same for any subset. All scripts accept the options below. Figures 1 and 2 need no training runs at all (the schematic and the release-map histograms are drawn from the model's own constants and sampling code); Figures 3-7 need only the learning-rate-0.03 runs and Supplementary Figures S1-S2 only the learning-rate-0.1 runs.

- The repository ships the finished runs (`reproduce_data/runs_lr0.03.json.gz`, `runs_lr0.1.json.gz`: 4 conditions x 20 seeds x 2 image sets each). By default the scripts therefore **re-analyze and re-plot from those runs (seconds to about 30 seconds)** and reproduce the numbers in the paper exactly.
- `--retrain` ignores the shipped runs and retrains from scratch on the CIFAR-10 subset (downloaded automatically with `scripts/fetch_cifar10.py`, about 163 MB). Each configuration is 80 networks per image set (4 conditions x 20 seeds); the Wnt conditions take roughly 8 CPU-minutes per network and the others under a minute, so about 2-3 hours per learning rate on an 8-core machine (`--workers N`, default all cores). The authors' image set is not redistributed; to retrain on it as well pass `--original-images DIR` (folders `train_subset/` and `test_subset/` containing `trucks`, `dogs`, `cats`, `cars`). Without it the shipped runs still provide the authors'-image panels.
- `--quick` is a smoke test (3 seeds, 20 sessions, CIFAR-10 only) writing to `figures_quick/` and `tables_quick/`.
- `--seeds N`, `--sessions N` change the number of seeds (default 20) and sessions (default 100).
- Seeds are fixed. On the same machine and library versions a retrained run reproduces the shipped run bit for bit. Across different numpy/BLAS builds the first training session agrees to about 1e-4 but float32 rounding differences are amplified by the training dynamics, so individual curves differ while the group statistics agree.
- Rebuild the paper PDF with `pip install -r docs/paper/requirements.txt && python docs/paper/build_paper.py` (numbers are filled in from `tables/`; run `python reproduce.py` first). The pre-specified analysis plan is in `docs/paper/analysis_plan.md`.

## Citation

Marri, R. & Amodeo, J. (2023). Modeling the effects of the Wnt and mTOR signaling pathways on ASD-impacted visual processing using neural networks to reflect population dynamics. [docs/paper/paper.pdf](docs/paper/paper.pdf) (source in `docs/paper/paper.md`)

```bibtex
@misc{marri2023wntmtor,
  author = {Marri, Rithvik and Amodeo, John},
  title  = {Modeling the effects of the Wnt and mTOR signaling pathways on ASD-impacted visual processing using neural networks to reflect population dynamics},
  year   = {2023}
}
```

## Limitations

From the paper's Limitations section:

- Compute limited the work to 100 training sessions and 900-node layers.
- Held-out accuracy is near chance in all four conditions because the image sets contain only three training and three test images per class.
- As a computational model, it cannot account for environmental factors such as behavioral therapy or specialized education.

## License

MIT, see [LICENSE](LICENSE). The license covers the code; the paper PDF remains the authors' work.
