import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from NeuralNetwork import NeuralNetwork
from network_tuning import get_release_maps


def make_net(n_in=5, hidden=2, classes=("a", "b", "c"), ccl5_delta=0.0):
    np.random.seed(0)
    return NeuralNetwork(list(classes), n_in, hidden, ccl5_delta=ccl5_delta)


def test_accuracy_uses_output():
    net = make_net(n_in=4, hidden=1)
    for w, b in zip(net.layer_weights, net.layer_biases):
        w[:] = 0
        b[:] = 0
    net.layer_biases[-1][:, 0] = [0, 0, 10]  # output always predicts class 2
    x = np.array([[9.0], [0.0], [0.0], [0.0]])  # input max at index 0
    images = {"c": [x, x], "b": [x, x]}
    labels_hit = {"c": np.array([[0], [0], [1.0]]), "b": np.array([[0], [0], [1.0]])}
    labels_miss = {"c": np.array([[0], [1.0], [0]]), "b": np.array([[0], [1.0], [0]])}
    assert net.test(images, labels_hit, use_release_maps=False) == 100.0
    assert net.test(images, labels_miss, use_release_maps=False) == 0.0
    loss, acc = net.evaluate(images, labels_hit, use_release_maps=False)
    assert acc == 100.0 and loss < 1e-3


def _loss(net, x, label, maps):
    net.release_map_fn = lambda w, d: maps
    out = net.feedforward(x, True)[-1]
    return -np.sum(label * np.log(out + 1e-10))


def test_gradient_includes_release_map_finite_difference():
    net = make_net(ccl5_delta=0.4)
    rng = np.random.RandomState(1)
    maps = [rng.uniform(0.5, 2.0, size=w.shape) for w in net.layer_weights]
    x = rng.rand(5, 1)
    label = np.array([[0], [1.0], [0]])

    net.release_map_fn = lambda w, d: maps
    acts = net.feedforward(x, True)
    trained = copy.deepcopy(net)
    trained.feedback(label, acts, 1.0, max_gradient_norm=1e9)

    eps = 1e-6
    for params_before, params_after, name in (
            (net.layer_weights, trained.layer_weights, "w"),
            (net.layer_biases, trained.layer_biases, "b")):
        for k, p in enumerate(params_before):
            analytic = -(params_after[k] - p)
            numeric = np.zeros_like(p)
            for i in np.ndindex(*p.shape):
                orig = p[i]
                p[i] = orig + eps
                lp = _loss(net, x, label, maps)
                p[i] = orig - eps
                lm = _loss(net, x, label, maps)
                p[i] = orig
                numeric[i] = (lp - lm) / (2 * eps)
            err = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))
            assert err < 1e-5, (name, k, err)
            if name == "w":
                # gradient ignoring M differs from the true gradient
                assert not np.allclose(analytic, analytic / maps[k])

    # Unit maps give a different gradient than the non-trivial maps
    net.release_map_fn = lambda w, d: [np.ones_like(m) for m in maps]
    plain = copy.deepcopy(net)
    plain.feedback(label, plain.feedforward(x, True), 1.0, max_gradient_norm=1e9)
    assert not np.allclose(plain.layer_weights[0], trained.layer_weights[0])


def test_ccl5_release_map_statistics():
    # Documented observation of the release-map formula Poisson(100*(1+d))/100, not a requirement:
    # mean ~ 1+d and CV ~ 1/sqrt(100*(1+d)), so CCL5 elevation gives a slightly LOWER CV than control.
    np.random.seed(0)
    w = [np.zeros((300, 300))]
    cvs = {}
    for delta in (0.0, 0.40467):
        m = get_release_maps(w, delta)[0]
        assert abs(m.mean() - (1 + delta)) < 0.01
        cvs[delta] = m.std() / m.mean()
        assert abs(cvs[delta] - 1 / np.sqrt(100 * (1 + delta))) < 0.005
    assert cvs[0.40467] < cvs[0.0]


def _two_class_data():
    rng = np.random.RandomState(3)
    images = {c: [rng.rand(5, 1) for _ in range(5)] for c in ("a", "b")}
    labels = {"a": np.array([[1.0], [0]]), "b": np.array([[0], [1.0]])}
    return images, labels


def test_shuffling():
    net = make_net(n_in=5, hidden=1, classes=("a", "b"))
    images, labels = _two_class_data()
    np.random.seed(42)
    orders = []
    for _ in range(6):
        losses, names = net.run_session(images, labels, 0.01)
        assert len(losses) == 10 and len(names) == 10
        assert names.count("a") == 5 and names.count("b") == 5
        orders.append(tuple(names))
    blocked = [o in (("a",) * 5 + ("b",) * 5, ("b",) * 5 + ("a",) * 5) for o in orders]
    assert not all(blocked)
    assert len(set(orders)) > 1


def test_softmax_output():
    net = make_net()
    out = net.feedforward(np.random.rand(5, 1), True)[-1]
    assert abs(out.sum() - 1) < 1e-12
    assert np.all(out > 0) and np.all(out < 1)


def test_train_and_test_same_release_distribution():
    net = make_net(ccl5_delta=0.40467)
    np.random.seed(5)
    net.feedforward(np.random.rand(5, 1))  # default must use release maps
    maps = np.concatenate([m.ravel() for m in net.last_release_maps])
    assert abs(maps.mean() - 1.40467) < 0.02
    import inspect
    assert inspect.signature(net.test).parameters["use_release_maps"].default is True
