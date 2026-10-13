import numpy as np
import pytest

from weli.layers import Dense, Conv2D, LayerNorm, GroupNorm, InstanceNorm, WeightNorm
from weli.models.registry import registry

EPS = 1e-6

CASES = [
    ("layernorm_2d", lambda: LayerNorm(), (4,), 6),
    ("layernorm_3d", lambda: LayerNorm(), (5, 4), 3),
    ("groupnorm_2d", lambda: GroupNorm(groups=2), (4,), 5),
    ("groupnorm_4d", lambda: GroupNorm(groups=2), (3, 3, 4), 3),
    ("instancenorm_4d", lambda: InstanceNorm(), (3, 3, 4), 3),
    ("weightnorm_dense", lambda: WeightNorm(Dense(3, activation='tanh')), (4,), 5),
    ("weightnorm_conv2d", lambda: WeightNorm(Conv2D(2, 3, activation='relu')), (5, 5, 2), 3),
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


def test_layernorm_normalizes_each_sample_over_features():
    rng = np.random.default_rng(1)
    layer = LayerNorm()
    layer.initialize((None, 8))
    out = layer.forward(rng.normal(3.0, 5.0, size=(16, 8)))

    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_groupnorm_with_one_group_per_channel_equals_instancenorm():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 4, 4, 6))

    group = GroupNorm(groups=6)
    group.initialize((None, 4, 4, 6))
    instance = InstanceNorm()
    instance.initialize((None, 4, 4, 6))

    np.testing.assert_allclose(group.forward(x), instance.forward(x), atol=1e-12)


def test_instancenorm_zero_mean_per_channel_and_sample():
    rng = np.random.default_rng(3)
    layer = InstanceNorm()
    layer.initialize((None, 5, 5, 3))
    out = layer.forward(rng.normal(size=(4, 5, 5, 3)))

    np.testing.assert_allclose(out.mean(axis=(1, 2)), 0.0, atol=1e-10)


def test_groupnorm_rejects_indivisible_channels():
    layer = GroupNorm(groups=3)
    with pytest.raises(ValueError):
        layer.initialize((None, 8))


def test_layernorm_rejects_1d_input():
    with pytest.raises(ValueError):
        LayerNorm().initialize((None,))


def test_weightnorm_forward_matches_reparameterized_dense():
    rng = np.random.default_rng(4)
    inner = Dense(3)
    layer = WeightNorm(inner)
    layer.initialize((None, 4))
    v = layer.parameters['v']
    g = layer.parameters['g']
    x = rng.normal(size=(5, 4))

    expected = x @ (g * v / np.linalg.norm(v, axis=0, keepdims=True)) + layer.parameters['b']
    np.testing.assert_allclose(layer.forward(x), expected, atol=1e-12)


def test_weightnorm_exposes_only_wrapper_parameters():
    layer = WeightNorm(Dense(3))
    layer.initialize((None, 4))
    assert set(layer.parameters) == {'g', 'v', 'b'}


def test_weightnorm_requires_weight_parameter():
    class NoWeight(LayerNorm):
        pass

    layer = WeightNorm(NoWeight())
    with pytest.raises(ValueError):
        layer.initialize((None, 4))


@pytest.mark.parametrize("make", [
    lambda: LayerNorm(epsilon=1e-4),
    lambda: GroupNorm(groups=2, epsilon=1e-3),
    lambda: InstanceNorm(epsilon=1e-3),
    lambda: WeightNorm(Dense(3, activation='tanh')),
], ids=["layernorm", "groupnorm", "instancenorm", "weightnorm"])
def test_config_round_trip_through_registry(make):
    original = make()
    restored = registry.deserialize_layer(original.get_config())

    assert type(restored) is type(original)
    assert restored.get_config() == original.get_config()
