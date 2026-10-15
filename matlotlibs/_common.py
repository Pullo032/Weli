import numpy as np
import matplotlib.pyplot as plt


def finish(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def flatten_features(x):
    x = np.asarray(x)
    return x.reshape(len(x), -1)


def targets_and_kind(y):
    """Retourne (valeurs, is_class). Les labels one-hot sont convertis en indices de classe."""
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y.reshape(len(y), -1), axis=1), True
    y = y.reshape(len(y))
    is_class = bool(np.allclose(y, np.round(y)) and len(np.unique(y)) <= 20)
    return y, is_class
