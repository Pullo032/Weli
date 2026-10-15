import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from matlotlibs import (
    plot_loss, plot_accuracy, plot_loss_and_accuracy,
    plot_data_2d, plot_data_3d, plot_training_3d,
)

HISTORY = {
    'train_loss': [1.0, 0.7, 0.5, 0.4],
    'val_loss': [1.1, 0.8, 0.6, 0.55],
    'train_acc': [0.4, 0.6, 0.75, 0.8],
    'val_acc': [0.35, 0.55, 0.7, 0.72],
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_plot_loss_draws_train_and_validation_curves():
    fig = plot_loss(HISTORY, show=False)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == 'Loss'


def test_plot_accuracy_draws_available_keys_only():
    fig = plot_accuracy({'train_acc': [0.1, 0.2]}, show=False)
    assert len(fig.axes[0].lines) == 1


def test_plot_accuracy_without_matching_keys_raises():
    with pytest.raises(ValueError):
        plot_accuracy({'train_loss': [1.0]}, show=False)


def test_plot_loss_and_accuracy_uses_two_axes():
    fig = plot_loss_and_accuracy(HISTORY, show=False)
    assert len(fig.axes) == 2


def test_history_plot_saves_file(tmp_path):
    target = tmp_path / "loss.png"
    plot_loss(HISTORY, save_path=target, show=False)
    assert target.exists()


def test_plot_data_2d_one_hot_targets_are_grouped_by_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.eye(3)[rng.integers(0, 3, size=30)]
    fig = plot_data_2d(X, y, show=False)
    assert len(fig.axes[0].collections) == len(np.unique(np.argmax(y, axis=1)))


def test_plot_data_2d_continuous_targets_add_colorbar():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=(30, 1))
    fig = plot_data_2d(X, y, show=False)
    assert len(fig.axes) == 2


def test_plot_data_2d_single_feature_puts_target_on_y_axis():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.linspace(0, 1, 20)
    fig = plot_data_2d(X, y, show=False)
    assert fig.axes[0].get_ylabel() == 'target'


def test_plot_data_3d_requires_two_features():
    with pytest.raises(ValueError):
        plot_data_3d(np.ones((10, 1)), np.ones(10), show=False)


def test_plot_data_3d_returns_3d_axes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 3))
    y = rng.integers(0, 2, size=25)
    fig = plot_data_3d(X, y, show=False)
    assert fig.axes[0].name == '3d'


def test_plot_training_3d_draws_train_and_validation_trajectories():
    fig = plot_training_3d(HISTORY, show=False)
    ax = fig.axes[0]
    assert ax.name == '3d'
    assert len(ax.lines) == 2


def test_plot_training_3d_without_validation_draws_train_only():
    history = {'train_loss': [1.0, 0.5], 'train_acc': [0.5, 0.9]}
    fig = plot_training_3d(history, show=False)
    assert len(fig.axes[0].lines) == 1


def test_plot_training_3d_missing_train_keys_raises():
    with pytest.raises(ValueError):
        plot_training_3d({'val_loss': [1.0]}, show=False)
