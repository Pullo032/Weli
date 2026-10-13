import matplotlib.pyplot as plt

from ._common import finish, flatten_features, targets_and_kind


def _plot_history(ax, history, keys, ylabel, title):
    series = [(key, history[key]) for key in keys if key in history]
    if not series:
        raise ValueError(f"Aucune des clés {keys} n'est présente dans l'historique")
    for key, values in series:
        ax.plot(range(1, len(values) + 1), values, linewidth=2, label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_loss(history, keys=('train_loss', 'val_loss'), save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_history(ax, history, keys, 'Loss', 'Loss')
    return finish(fig, save_path, show)


def plot_accuracy(history, keys=('train_acc', 'val_acc'), save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_history(ax, history, keys, 'Accuracy', 'Accuracy')
    return finish(fig, save_path, show)


def plot_loss_and_accuracy(history, loss_keys=('train_loss', 'val_loss'),
                           acc_keys=('train_acc', 'val_acc'), save_path=None, show=True):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_history(ax_loss, history, loss_keys, 'Loss', 'Loss')
    _plot_history(ax_acc, history, acc_keys, 'Accuracy', 'Accuracy')
    return finish(fig, save_path, show)


def plot_data_2d(x_train, y_train, feature_indices=(0, 1), title='Training data',
                 save_path=None, show=True):
    """
    Nuage de points 2D des données d'entraînement, coloré par label (classes) ou par cible (régression).
    Avec une seule feature, la cible est placée en ordonnée.
    """
    X = flatten_features(x_train)
    labels, is_class = targets_and_kind(y_train)
    i, j = feature_indices

    if X.shape[1] == 1:
        px, py = X[:, 0], labels
        xlabel, ylabel = 'feature 0', 'target'
    else:
        px, py = X[:, i], X[:, j]
        xlabel, ylabel = f'feature {i}', f'feature {j}'

    fig, ax = plt.subplots(figsize=(8, 6))
    if is_class:
        for cls in sorted(set(labels.tolist())):
            mask = labels == cls
            ax.scatter(px[mask], py[mask], s=25, alpha=0.7, label=str(cls))
        ax.legend(title='class')
    else:
        scatter = ax.scatter(px, py, c=labels, cmap='viridis', s=25, alpha=0.8)
        fig.colorbar(scatter, ax=ax, label='target')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return finish(fig, save_path, show)
