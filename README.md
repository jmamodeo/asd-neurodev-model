# ASD Neurodevelopment Model

A Python neural-network model of how elevated Wnt and mTOR-activated CCL5 signaling in autism spectrum disorder (ASD) could affect visual image classification.

## Background

ASD is a heterogeneous neurodevelopmental disorder, and vision is one of the senses particularly affected. Two pathways, Wnt and mTOR, have been identified as high risk. Wnt proteins stabilize synapses, and mTOR activates CCL5, which stimulates glutamate release. In excess, these proteins cause unnecessary synapses to form and survive, a hallmark of ASD.

The paper uses visual processing as a base to examine the effect of both proteins at ASD levels in a neural network, as a way to study population dynamics rather than only single cells. Its abstract reports that the perturbed networks failed to classify images as accurately as the control.

## How the model works

- **Input.** `visual_input.py` reads each image, converts it to grayscale, resizes it to a square whose side is `sqrt(image_vector_size)`, scales pixel values to 0-1, and flattens it into a vector of length `image_vector_size`. This vector is the input layer.
- **Layer sizes.** Hidden layers start equal in size to the input layer. With `wnt_delta`, each hidden layer gains `int(previous_size * wnt_delta)` nodes over the previous layer, modeling extra synapses (more connections and redundancy). The output layer has one node per class.
- **CCL5 / glutamate release.** For each forward pass, `network_tuning.py` draws a release map for every weight matrix from a Poisson distribution with mean `100 * (1 + ccl5_delta)`, divided by 100. The weights are multiplied by these maps, modeling stochastic glutamate release.
- **Learning.** Layers use sigmoid activations. Gradients are clipped to a maximum norm of 1.0, and loss is cross-entropy.
- **Release maps in training and testing.** Training always uses release maps. The `use_release_maps` setting in `main.py` only affects testing.

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
| `learn_r` | 0.1 | Learning rate |
| `sessions` | 100 | Passes over the full training set |
| `use_release_maps` | `True` | Release-map weighting at test time |

The paper compares four conditions, which can be set as examples:

| Condition | `wnt_delta` | `ccl5_delta` |
|---|---|---|
| Control | 0 | 0 |
| CCL5 only | 0 | 0.40467 |
| Wnt only | 0.41859 | 0 |
| Both | 0.41859 | 0.40467 |

## Outputs

- The loss for each training session is printed.
- Test accuracy is printed.
- `results/<YYYY-MM-DD HHhMMm> results/` contains `training_loss_plot.png` and `training_loss_table.xlsx`.

No random seed is set, so runs differ.

## Citation

Marri, R. & Amodeo, J. (2023). Modeling the effects of the Wnt and mTOR signaling pathways on ASD-impacted visual processing using neural networks to reflect population dynamics. [docs/Marri & Amodeo 2023.pdf](docs/Marri%20%26%20Amodeo%202023.pdf)

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
- Test accuracy was similar across the four networks, most likely because the image classes were very distinct.
- As a computational model, it cannot account for environmental factors such as behavioral therapy or specialized education.

## License

MIT, see [LICENSE](LICENSE). The license covers the code; the paper PDF remains the authors' work.
