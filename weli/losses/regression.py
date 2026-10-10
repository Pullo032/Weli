"""
Fonctions de perte pour la régression.
"""
import numpy as np
from typing import Optional
from .base import Loss

class MSE(Loss):
    """
    Mean Squared Error (Erreur Quadratique Moyenne).

    Formule: L = 1/N * Σ(y_pred - y_true)²
    Gradient: dL/dy_pred = 2/N * (y_pred - y_true)
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'MSE', reduction)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        squared_error = np.square(y_pred - y_true)

        self.loss_value = self._apply_reduction(squared_error)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = 2 * (self.y_pred - self.y_true)
        return self._apply_reduction_to_gradient(gradient)

class MAE(Loss):
    """
    Mean Absolute Error (Erreur Absolue Moyenne).

    Formule: L = 1/N * Σ|y_pred - y_true|
    Gradient: dL/dy_pred = sign(y_pred - y_true) / N
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'MAE', reduction)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        absolute_error = np.abs(y_pred - y_true)

        self.loss_value = self._apply_reduction(absolute_error)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = np.sign(self.y_pred - self.y_true)
        return self._apply_reduction_to_gradient(gradient)

class HuberLoss(Loss):
    """
    Huber Loss (combinaison de MSE et MAE).

    Formule:
        L = 0.5 * (y_pred - y_true)²          si |y_pred - y_true| <= delta
        L = delta * |y_pred - y_true| - 0.5 * delta²  sinon

    Gradient:
        dL/dy_pred = (y_pred - y_true)        si |y_pred - y_true| <= delta
        dL/dy_pred = delta * sign(y_pred - y_true)  sinon
    """

    def __init__(self, delta: float = 1.0, name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'HuberLoss', reduction)
        self.delta = delta
        self.mask = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        diff = y_pred - y_true
        abs_diff = np.abs(diff)

        self.mask = abs_diff <= self.delta

        loss = np.zeros_like(diff)
        loss[self.mask] = 0.5 * np.square(diff[self.mask])
        loss[~self.mask] = self.delta * abs_diff[~self.mask] - 0.5 * self.delta ** 2

        self.loss_value = self._apply_reduction(loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None or self.mask is None:
            raise RuntimeError("Must call forward() before backward()")

        diff = self.y_pred - self.y_true
        gradient = np.zeros_like(diff)
        gradient[self.mask] = diff[self.mask]
        gradient[~self.mask] = self.delta * np.sign(diff[~self.mask])

        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config['delta'] = self.delta
        return config

class MSLE(Loss):
    """
    Mean Squared Logarithmic Error.

    Formule: L = 1/N * Σ(log(1 + y_pred) - log(1 + y_true))²
    Gradient: dL/dy_pred = 2/N * (log(1 + y_pred) - log(1 + y_true)) / (1 + y_pred)
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'MSLE', reduction)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        clipped_pred = np.maximum(y_pred, 0)
        clipped_true = np.maximum(y_true, 0)

        log_error = np.square(np.log1p(clipped_pred) - np.log1p(clipped_true))

        self.loss_value = self._apply_reduction(log_error)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        clipped_pred = np.maximum(self.y_pred, 0)
        clipped_true = np.maximum(self.y_true, 0)

        gradient = 2 * (np.log1p(clipped_pred) - np.log1p(clipped_true)) / (1 + clipped_pred)
        gradient = np.where(self.y_pred < 0, 0.0, gradient)

        return self._apply_reduction_to_gradient(gradient)
