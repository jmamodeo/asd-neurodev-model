"""Run one (image set, condition, seed) experiment and save the per-session curves as JSON.

Used by reproduce.py. Conditions are the four paper conditions (a 2x2 of Wnt x CCL5):
control, ccl5, wnt, both, with the ASD fold changes from the paper (dWnt = 0.41859, dCCL5 = 0.40467).

Two engineering choices (statistically equivalent to the plain code, for speed; identical for all conditions):
  * float32 instead of float64 weights/activations;
  * release maps are drawn as random windows of a large pre-generated i.i.d. Poisson(100*(1+dCCL5))/100 pool
    (the same distribution as network_tuning.get_release_maps, but without a fresh Poisson draw for each of the
    up to 17.5 million synapses on every forward pass).
Everything else (layer sizes, N(0,1) init, clipping, per-image SGD, 4 hidden layers, 900 inputs, 100 sessions)
is as in neural_network.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker: deterministic and parallel-friendly
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
import warnings

import numpy as np

from asd_model.network_tuning import POISSON_MEAN_SCALE
from asd_model.neural_network import ASD_CCL5_DELTA, ASD_WNT_DELTA, NeuralNetwork
from asd_model.visual_input import encode_image_labels, image_to_matrix

CONDITIONS = {  # name: (ccl5_delta, wnt_delta)
    "control": (0.0, 0.0),
    "ccl5": (ASD_CCL5_DELTA, 0.0),
    "wnt": (0.0, ASD_WNT_DELTA),
    "both": (ASD_CCL5_DELTA, ASD_WNT_DELTA),
}
CLASS_NAMES = ["trucks", "dogs", "cats", "cars"]
IMAGE_VECTOR_SIZE = 900
HIDDEN_LAYERS = 4
POOL_SIZE = 40_000_000
# Seeds are offset so that the initial weights (seed), the shuffle order and the release-map stream are independent
SHUFFLE_SEED_OFFSET = 10_000
RELEASE_MAP_SEED_OFFSET = 20_000


def make_pooled_sampler(ccl5_delta, seed):
    """Return a release-map sampler that takes random windows of one large pre-drawn Poisson pool."""
    rng = np.random.default_rng(seed)
    pool = (rng.poisson(POISSON_MEAN_SCALE * (1 + ccl5_delta), size=POOL_SIZE) / POISSON_MEAN_SCALE).astype(np.float32)

    def sampler(layer_weights, _ccl5_delta):
        release_maps = []
        for weights in layer_weights:
            offset = int(rng.integers(0, POOL_SIZE - weights.size + 1))
            release_maps.append(pool[offset : offset + weights.size].reshape(weights.shape))
        return release_maps

    return sampler


def _to_float32(image_dict):
    return {
        image_class: [vector.astype(np.float32) for vector in vectors] for image_class, vectors in image_dict.items()
    }


def run_path(out_dir, data_name, condition, seed):
    """Return the path of the JSON file holding one run."""
    return os.path.join(out_dir, f"{data_name}_{condition}_{seed}.json")


def run_experiment(data_name, data_dir, condition, seed, learning_rate, sessions, out_dir):
    """Train and evaluate one network and write its curves to ``out_dir`` (skipped if the result already exists).

    Parameters
    ----------
    data_name : str
        Name of the image set (``"cifar3"`` or ``"original"``), stored in the result.
    data_dir : str
        Folder with ``train_subset/`` and ``test_subset/``.
    condition : str
        One of ``CONDITIONS``: control, ccl5, wnt or both.
    seed : int
        Seed for the initial weights, the shuffle order and the release maps.
    learning_rate : float
        Learning rate.
    sessions : int
        Number of training sessions.
    out_dir : str
        Folder for the JSON result.

    Returns
    -------
    str
        Path of the JSON file.
    """
    path = run_path(out_dir, data_name, condition, seed)
    if os.path.exists(path):
        return path
    ccl5_delta, wnt_delta = CONDITIONS[condition]
    train_images = _to_float32(image_to_matrix(os.path.join(data_dir, "train_subset"), CLASS_NAMES, IMAGE_VECTOR_SIZE))
    test_images = _to_float32(image_to_matrix(os.path.join(data_dir, "test_subset"), CLASS_NAMES, IMAGE_VECTOR_SIZE))
    labels = {key: value.astype(np.float32) for key, value in encode_image_labels(train_images).items()}
    np.random.seed(seed)  # initial weights (matched between control and CCL5)
    net = NeuralNetwork(CLASS_NAMES, IMAGE_VECTOR_SIZE, HIDDEN_LAYERS, ccl5_delta=ccl5_delta, wnt_delta=wnt_delta)
    net.layer_weights = [weights.astype(np.float32) for weights in net.layer_weights]
    net.layer_biases = [biases.astype(np.float32) for biases in net.layer_biases]
    net.release_map_fn = make_pooled_sampler(ccl5_delta, RELEASE_MAP_SEED_OFFSET + seed)
    np.random.seed(SHUFFLE_SEED_OFFSET + seed)  # shuffle order: identical across conditions for a seed
    start_time = time.time()
    train_loss, val_loss, val_acc = [], [], []
    for _ in range(sessions):
        losses, _ = net.run_session(train_images, labels, learning_rate)
        train_loss.append(float(np.mean(losses)))
        session_val_loss, session_val_acc = net.evaluate(test_images, labels, True)
        val_loss.append(float(session_val_loss))
        val_acc.append(float(session_val_acc))
    final_train_loss, final_train_acc = net.evaluate(train_images, labels, True)
    _, acc_fixed = net.evaluate(test_images, labels, False)
    result = dict(
        data=data_name,
        cond=condition,
        seed=seed,
        lr=learning_rate,
        layer_sizes=net.layer_sizes,
        sessions=sessions,
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        final_train_eval_loss=float(final_train_loss),
        final_train_acc=float(final_train_acc),
        final_test_acc_fixed_weights=float(acc_fixed),
        seconds=time.time() - start_time,
    )
    temp_path = path + f".tmp{os.getpid()}"
    with open(temp_path, "w") as f:
        json.dump(result, f)
    os.replace(temp_path, path)
    return path


def run_job(job):
    """Run one job tuple (the arguments of ``run_experiment``); used by the worker processes of reproduce.py."""
    warnings.filterwarnings("ignore")
    return run_experiment(*job)
