"""
Fonctions de perte pour la classification.
"""
import numpy as np
from typing import Optional
from .base import Loss


def _to_pm_one(y_true: np.ndarray) -> np.ndarray:
    """Convertit les labels en {-1, 1} ; les labels 0/1 sont convertis en 2y - 1."""
    y_true = np.asarray(y_true, dtype=float)
    if np.any(y_true < 0):
        return y_true
    return 2 * y_true - 1


class CrossEntropy(Loss):
    """
    Cross Entropy Loss (Entropie Croisée).

    Formule: L = -Σ y_true * log(y_pred)

    Args:
        from_logits: Si True, y_pred sont des logits (avant softmax)
        label_smoothing: Niveau de lissage des labels (0.0 à 1.0)
        ignore_index: Étiquette entière ignorée (sans effet sur les cibles one-hot)
        name: Nom de la fonction de perte
        reduction: Méthode de réduction
    """

    def __init__(self, from_logits: bool = False,
                 label_smoothing: float = 0.0,
                 ignore_index: int = -100,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'CrossEntropy', reduction)

        self.from_logits = from_logits
        self.label_smoothing = max(0.0, min(1.0, label_smoothing))
        self.ignore_index = ignore_index

        self.y_pred_probs = None
        self.y_true_smoothed = None
        self.valid_mask = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]

        if y_true.shape in ((batch_size,), (batch_size, 1)):
            indices = y_true.reshape(-1).astype(int)
            valid = indices != self.ignore_index
            valid_indices = indices[valid]
            if np.any(valid_indices < 0) or np.any(valid_indices >= num_classes):
                raise ValueError(
                    f"Class indices out of range for y_pred shape {y_pred.shape}: "
                    f"min={valid_indices.min()}, max={valid_indices.max()}"
                )
            targets = np.zeros((batch_size, num_classes))
            targets[np.nonzero(valid)[0], valid_indices] = 1
        else:
            targets = y_true.astype(float)
            valid = np.ones(batch_size, dtype=bool)

        self._validate_inputs(y_pred, targets)
        self._prepare_inputs(y_pred, targets)
        self.valid_mask = valid

        if self.from_logits:
            self.y_pred_probs = self._softmax(y_pred)
        else:
            self.y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        if self.label_smoothing > 0:
            self.y_true_smoothed = (1 - self.label_smoothing) * targets + \
                                   self.label_smoothing / num_classes
        else:
            self.y_true_smoothed = targets

        log_probs = np.log(self.y_pred_probs + self.epsilon)
        ce_per_sample = -np.sum(self.y_true_smoothed * log_probs, axis=1)
        ce_valid = ce_per_sample[valid] if np.any(valid) else np.zeros(1)

        self.loss_value = self._apply_reduction(ce_valid)
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()

        if self.from_logits:
            gradient = self.y_pred_probs - self.y_true_smoothed
        else:
            gradient = -self.y_true_smoothed / (self.y_pred_probs + self.epsilon)

        gradient = gradient * self.valid_mask[:, np.newaxis]
        return self._apply_reduction_to_gradient(gradient)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax stable numériquement."""
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing,
            'ignore_index': self.ignore_index
        })
        return config


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy (pour classification binaire).

    Formule: L = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    """

    def __init__(self, from_logits: bool = False,
                 label_smoothing: float = 0.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'BinaryCrossEntropy', reduction)

        self.from_logits = from_logits
        self.label_smoothing = max(0.0, min(1.0, label_smoothing))

        self.y_pred_probs = None
        self.y_true_smoothed = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)

        if self.from_logits:
            y_pred_probs = self._sigmoid(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs

        if self.label_smoothing > 0:
            y_true_smoothed = (1 - self.label_smoothing) * y_true + \
                             self.label_smoothing * 0.5
            self.y_true_smoothed = y_true_smoothed
        else:
            self.y_true_smoothed = y_true.copy()

        bce = -(self.y_true_smoothed * np.log(y_pred_probs + self.epsilon) +
                (1 - self.y_true_smoothed) * np.log(1 - y_pred_probs + self.epsilon))

        self.loss_value = self._apply_reduction(bce)
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()

        if self.from_logits:
            gradient = self.y_pred_probs - self.y_true_smoothed
        else:
            gradient = (self.y_pred_probs - self.y_true_smoothed) / \
                      ((self.y_pred_probs + self.epsilon) *
                       (1 - self.y_pred_probs + self.epsilon))

        return self._apply_reduction_to_gradient(gradient)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid stable numériquement."""
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config


class CategoricalCrossEntropy(CrossEntropy):
    """
    Categorical Cross Entropy (alias pour multi-classes avec one-hot).
    """

    def __init__(self, from_logits: bool = False,
                 label_smoothing: float = 0.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(from_logits=from_logits,
                        label_smoothing=label_smoothing,
                        ignore_index=-100,
                        name=name or 'CategoricalCrossEntropy',
                        reduction=reduction)


class SparseCategoricalCrossEntropy(Loss):
    """
    Sparse Categorical Cross Entropy (pour labels entiers).
    """

    def __init__(self, from_logits: bool = False,
                 ignore_index: int = -100,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'SparseCategoricalCrossEntropy', reduction)

        self.from_logits = from_logits
        self.ignore_index = ignore_index

        self.y_pred_probs = None
        self.y_true_indices = None
        self.valid_mask = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            indices = np.argmax(y_true, axis=1)
        else:
            indices = y_true.reshape(-1).astype(int)

        if indices.shape[0] != batch_size:
            raise ValueError(
                f"y_true contient {indices.shape[0]} étiquettes pour {batch_size} prédictions"
            )

        valid = indices != self.ignore_index
        valid_indices = indices[valid]
        if np.any(valid_indices < 0) or np.any(valid_indices >= num_classes):
            raise ValueError(
                f"Invalid class indices: min={valid_indices.min()}, max={valid_indices.max()}"
            )

        self._prepare_inputs(y_pred, y_true)
        self.valid_mask = valid
        self.y_true_indices = np.where(valid, indices, 0)

        if self.from_logits:
            self.y_pred_probs = self._softmax(y_pred)
        else:
            self.y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        true_class_probs = self.y_pred_probs[np.arange(batch_size), self.y_true_indices]
        ce_per_sample = -np.log(true_class_probs + self.epsilon)
        ce_valid = ce_per_sample[valid] if np.any(valid) else np.zeros(1)

        self.loss_value = self._apply_reduction(ce_valid)
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()

        rows = np.arange(self.y_pred.shape[0])
        if self.from_logits:
            gradient = self.y_pred_probs.copy()
            gradient[rows, self.y_true_indices] -= 1
        else:
            gradient = np.zeros_like(self.y_pred, dtype=float)
            gradient[rows, self.y_true_indices] = \
                -1 / (self.y_pred_probs[rows, self.y_true_indices] + self.epsilon)

        gradient[~self.valid_mask] = 0
        return self._apply_reduction_to_gradient(gradient)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'from_logits': self.from_logits,
            'ignore_index': self.ignore_index
        })
        return config


class HingeLoss(Loss):
    """
    Hinge Loss (pour SVM).

    Formule: L = max(0, margin - y_true * y_pred)
    où y_true ∈ {-1, 1} (les labels 0/1 sont acceptés et convertis en -1/1)
    """

    def __init__(self, margin: float = 1.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'HingeLoss', reduction)

        self.margin = margin

        self.mask = None
        self.y_true_transformed = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)

        self.y_true_transformed = _to_pm_one(y_true)

        hinge = np.maximum(0, self.margin - self.y_true_transformed * y_pred)
        self.mask = hinge > 0

        self.loss_value = self._apply_reduction(hinge)
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()

        gradient = np.where(self.mask, -self.y_true_transformed, 0.0)
        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config


class SquaredHingeLoss(Loss):
    """
    Squared Hinge Loss.

    Formule: L = max(0, margin - y_true * y_pred)²
    où y_true ∈ {-1, 1} (les labels 0/1 sont acceptés et convertis en -1/1)
    """

    def __init__(self, margin: float = 1.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'SquaredHingeLoss', reduction)

        self.margin = margin

        self.mask = None
        self.y_true_transformed = None
        self.hinge_values = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)

        self.y_true_transformed = _to_pm_one(y_true)

        hinge = np.maximum(0, self.margin - self.y_true_transformed * y_pred)
        self.hinge_values = hinge
        self.mask = hinge > 0

        self.loss_value = self._apply_reduction(np.square(hinge))
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()

        gradient = np.where(self.mask, -2 * self.y_true_transformed * self.hinge_values, 0.0)
        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config
