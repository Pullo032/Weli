import numpy as np
import pytest

from weli.layers import BatchNorm1D, BatchNorm2D, Conv2D, Dense, GRU, LSTM, SimpleRNN

EPS = 1e-6

CASES = [
    ("dense", lambda: Dense(3, activation='tanh'), (4,), 5),
    ("conv2d", lambda: Conv2D(2, 3, activation='relu'), (5, 5, 2), 3),
    ("batchnorm1d", lambda: BatchNorm1D(), (4,), 6),
    ("batchnorm2d", lambda: BatchNorm2D(), (4, 4, 2), 3),
    ("simplernn", lambda: SimpleRNN(3, return_sequences=True), (6, 2), 4),
    ("lstm", lambda: LSTM(3, return_sequences=True), (5, 2), 4),
    ("gru", lambda: GRU(3, return_sequences=True), (5, 2), 4),
]


def _setup(make, sample_shape, batch):
    rng = np.random.default_rng(0)
    layer = make()
    layer.initialize((None,) + sample_shape)
    x = rng.normal(size=(batch,) + sample_shape)
    R = rng.normal(size=layer.forward(x).shape)
    return layer, x, R


def _loss(layer, x, R, batch):
    return np.sum(layer.forward(x) * R) / batch


def _numerical(fn, arr):
    grad = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        orig = arr[idx]
        arr[idx] = orig + EPS
        plus = fn()
        arr[idx] = orig - EPS
        minus = fn()
        arr[idx] = orig
        grad[idx] = (plus - minus) / (2 * EPS)
    return grad


@pytest.mark.parametrize("name, make, sample_shape, batch", CASES, ids=[c[0] for c in CASES])
def test_parameter_gradients_match_finite_differences(name, make, sample_shape, batch):
    layer, x, R = _setup(make, sample_shape, batch)
    layer.forward(x)
    layer.backward(R / batch)
    analytic = {key: value.copy() for key, value in layer.gradients.items()}

    for key, param in layer.parameters.items():
        numerical = _numerical(lambda: _loss(layer, x, R, batch), param)
        np.testing.assert_allclose(analytic[key], numerical, rtol=1e-4, atol=1e-6,
                                   err_msg=f"{name}.{key}")


@pytest.mark.parametrize("name, make, sample_shape, batch", CASES, ids=[c[0] for c in CASES])
def test_input_gradient_matches_finite_differences(name, make, sample_shape, batch):
    layer, x, R = _setup(make, sample_shape, batch)
    layer.forward(x)
    dx = layer.backward(R / batch)

    numerical = _numerical(lambda: _loss(layer, x, R, batch), x)
    np.testing.assert_allclose(dx, numerical, rtol=1e-4, atol=1e-6, err_msg=f"{name} dx")
