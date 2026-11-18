"""Run one (dataset, condition, seed) experiment and save the per-session curves as JSON.

Used by reproduce.py. Conditions are the four paper conditions (a 2x2 of Wnt x CCL5):
control, ccl5, wnt, both, with the ASD fold changes from the paper (dWnt = 0.41859, dCCL5 = 0.40467).

Two engineering choices (statistically equivalent to the plain code, for speed; identical for all conditions):
  * float32 instead of float64 weights/activations;
  * release maps are drawn as random windows of a large pre-generated i.i.d. Poisson(100*(1+dCCL5))/100 pool
    (the same distribution as network_tuning.get_release_maps, but without a fresh Poisson draw for each of the
    up to 17.5 million synapses on every forward pass).
Everything else (layer sizes, N(0,1) init, clipping, per-image SGD, 4 hidden layers, 900 inputs, 100 sessions)
is as in NeuralNetwork.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker: deterministic and parallel-friendly
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
import numpy as np

from visual_input import image2matrix, encode_image_labels
from NeuralNetwork import NeuralNetwork

CONDITIONS = {  # name: (ccl5_delta, wnt_delta)
    "control": (0.0, 0.0),
    "ccl5": (0.40467, 0.0),
    "wnt": (0.0, 0.41859),
    "both": (0.40467, 0.41859),
}
CLASS_LIST = ['trucks', 'dogs', 'cats', 'cars']
IMAGE_VECTOR_SIZE = 900
HIDDEN_LAYERS = 4
POOL_SIZE = 40_000_000


def make_pooled_sampler(ccl5_delta, seed):
    rng = np.random.default_rng(seed)
    pool = (rng.poisson(100 * (1 + ccl5_delta), size=POOL_SIZE) / 100).astype(np.float32)

    def sampler(layer_weights, _ccl5_delta):
        maps = []
        for w in layer_weights:
            offset = int(rng.integers(0, POOL_SIZE - w.size + 1))
            maps.append(pool[offset:offset + w.size].reshape(w.shape))
        return maps
    return sampler


def _float32(image_dict):
    return {k: [v.astype(np.float32) for v in vs] for k, vs in image_dict.items()}


def run_path(out_dir, data_name, cond, seed):
    return os.path.join(out_dir, f"{data_name}_{cond}_{seed}.json")


def run_one(data_name, data_dir, cond, seed, lr, sessions, out_dir):
    """Train/evaluate one network. Skips if the result already exists (cache). Returns the JSON path."""
    path = run_path(out_dir, data_name, cond, seed)
    if os.path.exists(path):
        return path
    ccl5, wnt = CONDITIONS[cond]
    train = _float32(image2matrix(os.path.join(data_dir, "train_subset"), CLASS_LIST, IMAGE_VECTOR_SIZE))
    test = _float32(image2matrix(os.path.join(data_dir, "test_subset"), CLASS_LIST, IMAGE_VECTOR_SIZE))
    labels = {k: v.astype(np.float32) for k, v in encode_image_labels(train).items()}
    np.random.seed(seed)                                       # initial weights (matched between control and CCL5)
    net = NeuralNetwork(CLASS_LIST, IMAGE_VECTOR_SIZE, HIDDEN_LAYERS, ccl5_delta=ccl5, wnt_delta=wnt)
    net.layer_weights = [w.astype(np.float32) for w in net.layer_weights]
    net.layer_biases = [b.astype(np.float32) for b in net.layer_biases]
    net.release_map_fn = make_pooled_sampler(ccl5, 20000 + seed)
    np.random.seed(10000 + seed)                               # shuffle order: identical across conditions for a seed
    t0 = time.time()
    train_loss, val_loss, val_acc = [], [], []
    for _ in range(sessions):
        losses, _ = net.run_session(train, labels, lr)
        train_loss.append(float(np.mean(losses)))
        loss, acc = net.evaluate(test, labels, True)
        val_loss.append(float(loss)); val_acc.append(float(acc))
    final_train_loss, final_train_acc = net.evaluate(train, labels, True)
    _, acc_fixed = net.evaluate(test, labels, False)
    result = dict(data=data_name, cond=cond, seed=seed, lr=lr, layer_sizes=net.layer_sizes, sessions=sessions,
                  train_loss=train_loss, val_loss=val_loss, val_acc=val_acc,
                  final_train_eval_loss=float(final_train_loss), final_train_acc=float(final_train_acc),
                  final_test_acc_fixed_weights=float(acc_fixed), seconds=time.time() - t0)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)
    return path


def _job(args):
    import warnings
    warnings.filterwarnings("ignore")
    return run_one(*args)
