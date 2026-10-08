"""Wasserstein Loss pour Weli.

Implémentation légère et pédagogique pour les GANs et critiques de Wasserstein.
On optimise la distance de Wasserstein via une perte linéaire sur les scores.
"""

import numpy as np

from .base import Loss


class WassersteinLoss(Loss):
    """Perte de Wasserstein pour scores de critic (réel/faux)."""

    def __init__(self, name=None, reduction: str = 'mean'):
        super().__init__(name or 'WassersteinLoss', reduction)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)

        # En pratique, la perte est négative pour les critiques réels et positive
        # pour les faux, de la forme -E[y_true * y_pred].
        per_sample = -y_true * y_pred
        self.loss_value = self._apply_reduction(per_sample)
        return self.loss_value

    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        gradient = -self.y_true.copy()
        gradient = self._apply_reduction_to_gradient(gradient)
        return gradient


__all__ = ["WassersteinLoss"]

