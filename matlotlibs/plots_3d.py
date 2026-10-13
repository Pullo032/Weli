import numpy as np
import matplotlib.pyplot as plt

from ._common import finish, flatten_features, targets_and_kind


def plot_data_3d(x_train, y_train, feature_indices=(0, 1), title='Training data 3D',
                 save_path=None, show=True):
    """
    Nuage de points 3D: deux features en x/y, la cible (ou l'indice de classe) en z.
    """
    X = flatten_features(x_train)
    if X.shape[1] < 2:
        raise ValueError("plot_data_3d requiert au moins 2 features")
    labels, is_class = targets_and_kind(y_train)
    i, j = feature_indices

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')
    if is_class:
        for cls in sorted(set(labels.tolist())):
            mask = labels == cls
            ax.scatter(X[mask, i], X[mask, j], labels[mask], s=20, alpha=0.7, label=str(cls))
        ax.legend(title='class')
    else:
        scatter = ax.scatter(X[:, i], X[:, j], labels, c=labels, cmap='viridis', s=20, alpha=0.8)
        fig.colorbar(scatter, ax=ax, shrink=0.6, label='target')

    ax.set_xlabel(f'feature {i}')
    ax.set_ylabel(f'feature {j}')
    ax.set_zlabel('target')
    ax.set_title(title, fontweight='bold')
    return finish(fig, save_path, show)


def plot_training_3d(history, loss_keys=('train_loss', 'train_acc'),
                     val_keys=('val_loss', 'val_acc'), title='Loss / Accuracy / Epoch',
                     save_path=None, show=True):
    """
    Trajectoire 3D de l'entraînement: epoch en x, loss en y, accuracy en z.
    La trajectoire de validation est ajoutée si ses clés sont présentes.
    """
    loss_key, acc_key = loss_keys
    if loss_key not in history or acc_key not in history:
        raise ValueError(f"L'historique doit contenir {loss_key} et {acc_key}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')

    n = min(len(history[loss_key]), len(history[acc_key]))
    epochs = np.arange(1, n + 1)
    ax.plot(epochs, history[loss_key][:n], history[acc_key][:n], linewidth=2, label='train')

    val_loss_key, val_acc_key = val_keys
    if val_loss_key in history and val_acc_key in history:
        m = min(len(history[val_loss_key]), len(history[val_acc_key]))
        ax.plot(np.arange(1, m + 1), history[val_loss_key][:m], history[val_acc_key][:m],
                linewidth=2, label='validation')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_zlabel('Accuracy')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    return finish(fig, save_path, show)
