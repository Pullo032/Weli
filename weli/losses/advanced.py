"""
Fonctions de perte avancées.
"""
import numpy as np
from typing import Optional
from .base import Loss


def _cosine_grad(u: np.ndarray, v: np.ndarray, cos: np.ndarray,
                 norm_u: np.ndarray, norm_v: np.ndarray) -> np.ndarray:
    """Gradient de cos(u, v) par rapport à u."""
    return v / (norm_u * norm_v) - cos * u / (norm_u ** 2)


class KLDivergence(Loss):
    """
    Kullback-Leibler Divergence.

    Formule: L = Σ p * log(p / q), avec p = y_true / Σy_true et q = y_pred / Σy_pred
    Gradient: dL/dy_pred = (1 - p / q) / Σy_pred
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'KLDivergence', reduction)
        self.epsilon = 1e-7
        self.p = None
        self.q = None
        self.q_sum = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        p_raw = np.clip(y_true, self.epsilon, None)
        q_raw = np.clip(y_pred, self.epsilon, None)

        self.p = p_raw / np.sum(p_raw, axis=1, keepdims=True)
        self.q_sum = np.sum(q_raw, axis=1, keepdims=True)
        self.q = q_raw / self.q_sum

        kl_sum = np.sum(self.p * np.log(self.p / self.q), axis=1)

        self.loss_value = self._apply_reduction(kl_sum)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = (1 - self.p / self.q) / self.q_sum
        return self._apply_reduction_to_gradient(gradient)


class PoissonLoss(Loss):
    """
    Poisson Loss (pour données de comptage).

    Formule: L = Σ_features (y_pred - y_true * log(y_pred)), moyenne sur le lot
    Gradient: dL/dy_pred = 1 - y_true / y_pred
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'PoissonLoss', reduction)
        self.epsilon = 1e-7

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        y_pred_clipped = np.clip(y_pred, self.epsilon, None)

        poisson = y_pred_clipped - y_true * np.log(y_pred_clipped)

        if poisson.ndim > 1:
            poisson = np.sum(poisson, axis=1)

        self.loss_value = self._apply_reduction(poisson)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        y_pred_clipped = np.clip(self.y_pred, self.epsilon, None)
        gradient = 1 - self.y_true / y_pred_clipped
        return self._apply_reduction_to_gradient(gradient)


class CosineSimilarityLoss(Loss):
    """
    Cosine Similarity Loss.

    Formule: L = 1 - cos(y_pred, y_true)
    Gradient: dL/dy_pred = -(y_true / (|y_pred| |y_true|) - cos * y_pred / |y_pred|²)
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'CosineSimilarityLoss', reduction)
        self.epsilon = 1e-7
        self.norm_pred = None
        self.norm_true = None
        self.cosine_sim = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        self.norm_pred = np.clip(np.linalg.norm(y_pred, axis=1, keepdims=True), self.epsilon, None)
        self.norm_true = np.clip(np.linalg.norm(y_true, axis=1, keepdims=True), self.epsilon, None)

        self.cosine_sim = np.sum(y_pred * y_true, axis=1, keepdims=True) / \
                          (self.norm_pred * self.norm_true)

        loss = 1 - self.cosine_sim[:, 0]

        self.loss_value = self._apply_reduction(loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = -(self.y_true / (self.norm_pred * self.norm_true) -
                     self.cosine_sim * self.y_pred / self.norm_pred ** 2)
        return self._apply_reduction_to_gradient(gradient)


class LogCoshLoss(Loss):
    """
    Log-Cosh Loss (lisse approximation de MAE).

    Formule: L = log(cosh(y_pred - y_true)), sommée sur les features
    Gradient: dL/dy_pred = tanh(y_pred - y_true)
    """

    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'LogCoshLoss', reduction)
        self.epsilon = 1e-7
        self.diff = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true

        abs_diff = np.abs(self.diff)
        loss = abs_diff + np.log1p(np.exp(-2 * abs_diff)) - np.log(2)

        if loss.ndim > 1:
            loss = np.sum(loss, axis=1)

        self.loss_value = self._apply_reduction(loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = np.tanh(self.diff)
        return self._apply_reduction_to_gradient(gradient)


class FocalLoss(Loss):
    """
    Focal Loss pour déséquilibre de classes.

    Formule: L = -α(1 - p_t)^γ * log(p_t) où p_t est la probabilité de la vraie classe
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 from_logits: bool = False, name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'FocalLoss', reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.epsilon = 1e-7
        self.y_pred_probs = None
        self.y_true_one_hot = None
        self.pt = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]

        if self.from_logits:
            self.y_pred_probs = self._softmax(y_pred)
        else:
            self.y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        if y_true.ndim == 1:
            one_hot = np.zeros((batch_size, num_classes))
            one_hot[np.arange(batch_size), y_true.astype(int)] = 1
        else:
            one_hot = y_true.astype(float)
        self.y_true_one_hot = one_hot

        pt = np.sum(one_hot * self.y_pred_probs, axis=1)
        self.pt = np.clip(pt, self.epsilon, 1 - self.epsilon)

        focal_loss = -self.alpha * np.power(1 - self.pt, self.gamma) * np.log(self.pt)

        self.loss_value = self._apply_reduction(focal_loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        pt = self.pt[:, np.newaxis]
        dloss_dpt = self.alpha * (
            self.gamma * np.power(1 - pt, self.gamma - 1) * np.log(pt)
            - np.power(1 - pt, self.gamma) / pt
        )

        if self.from_logits:
            gradient = dloss_dpt * self.y_pred_probs * (self.y_true_one_hot - pt)
        else:
            gradient = dloss_dpt * self.y_true_one_hot

        return self._apply_reduction_to_gradient(gradient)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax stable numériquement."""
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class DiceLoss(Loss):
    """
    Dice Loss pour segmentation d'images.

    Formule: L = 1 - (2 * Σ p t + smooth) / (Σ p + Σ t + smooth), par exemple du lot
    """

    def __init__(self, smooth: float = 1.0,
                 name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'DiceLoss', reduction)
        self.smooth = smooth
        self.epsilon = 1e-7
        self.p = None
        self.t = None
        self.numerator = None
        self.denominator = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        batch_size = y_pred.shape[0]
        self.p = np.clip(y_pred, self.epsilon, 1 - self.epsilon).reshape(batch_size, -1)
        self.t = y_true.reshape(batch_size, -1)

        intersection = np.sum(self.p * self.t, axis=1)
        self.numerator = 2 * intersection + self.smooth
        self.denominator = np.sum(self.p, axis=1) + np.sum(self.t, axis=1) + \
                           self.smooth + self.epsilon

        loss = 1 - self.numerator / self.denominator

        self.loss_value = self._apply_reduction(loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")

        numerator = self.numerator[:, np.newaxis]
        denominator = self.denominator[:, np.newaxis]
        gradient = numerator / denominator ** 2 - 2 * self.t / denominator
        gradient = gradient.reshape(self.y_pred.shape)

        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config['smooth'] = self.smooth
        return config


class TripletLoss(Loss):
    """
    Triplet Loss pour apprentissage de métriques.

    Formule: L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """

    def __init__(self, margin: float = 1.0, distance: str = 'euclidean',
                 name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'TripletLoss', reduction)
        self.margin = margin
        self.distance = distance
        self.epsilon = 1e-7
        self.valid_triplets = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        y_pred: embeddings de shape (batch_size * 3, embedding_dim)
                [anchor1, positive1, negative1, anchor2, positive2, negative2, ...]
        y_true: labels non utilisés (pour compatibilité)
        """
        if y_pred.shape[0] % 3 != 0:
            raise ValueError(
                f"TripletLoss attend un nombre de lignes multiple de 3, reçu {y_pred.shape[0]}"
            )

        self.y_pred = y_pred
        self.batch_size = y_pred.shape[0] // 3
        self.anchor = y_pred[0::3]
        self.positive = y_pred[1::3]
        self.negative = y_pred[2::3]

        if self.distance == 'euclidean':
            pos_dist = np.sum(np.square(self.anchor - self.positive), axis=1)
            neg_dist = np.sum(np.square(self.anchor - self.negative), axis=1)
        elif self.distance == 'cosine':
            self.norm_a = np.clip(np.linalg.norm(self.anchor, axis=1, keepdims=True), self.epsilon, None)
            self.norm_p = np.clip(np.linalg.norm(self.positive, axis=1, keepdims=True), self.epsilon, None)
            self.norm_n = np.clip(np.linalg.norm(self.negative, axis=1, keepdims=True), self.epsilon, None)
            self.cos_ap = np.sum(self.anchor * self.positive, axis=1, keepdims=True) / \
                          (self.norm_a * self.norm_p)
            self.cos_an = np.sum(self.anchor * self.negative, axis=1, keepdims=True) / \
                          (self.norm_a * self.norm_n)
            pos_dist = 1 - self.cos_ap[:, 0]
            neg_dist = 1 - self.cos_an[:, 0]
        else:
            raise ValueError(f"Distance '{self.distance}' non supportée")

        losses = np.maximum(0, pos_dist - neg_dist + self.margin)
        self.valid_triplets = losses > 0

        self.loss_value = self._apply_reduction(losses)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.valid_triplets is None:
            raise RuntimeError("Must call forward() before backward()")

        a, p, n = self.anchor, self.positive, self.negative
        if self.distance == 'euclidean':
            grad_a = 2 * (n - p)
            grad_p = 2 * (p - a)
            grad_n = 2 * (a - n)
        else:
            grad_a = -_cosine_grad(a, p, self.cos_ap, self.norm_a, self.norm_p) + \
                      _cosine_grad(a, n, self.cos_an, self.norm_a, self.norm_n)
            grad_p = -_cosine_grad(p, a, self.cos_ap, self.norm_p, self.norm_a)
            grad_n = _cosine_grad(n, a, self.cos_an, self.norm_n, self.norm_a)

        active = self.valid_triplets[:, np.newaxis]
        gradient = np.zeros_like(self.y_pred, dtype=float)
        gradient[0::3] = np.where(active, grad_a, 0.0)
        gradient[1::3] = np.where(active, grad_p, 0.0)
        gradient[2::3] = np.where(active, grad_n, 0.0)

        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'margin': self.margin,
            'distance': self.distance
        })
        return config


class ContrastiveLoss(Loss):
    """
    Contrastive Loss pour apprentissage par similarité.

    Formule:
      Pour paires similaires: L = d²
      Pour paires dissimilaires: L = max(0, margin - d)²
    """

    def __init__(self, margin: float = 1.0,
                 name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'ContrastiveLoss', reduction)
        self.margin = margin
        self.distance = None
        self.embedding1 = None
        self.embedding2 = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        y_pred: embeddings de shape (batch_size * 2, embedding_dim)
                [sample1_A, sample1_B, sample2_A, sample2_B, ...]
        y_true: labels de similarité (1 pour similaire, 0 pour dissimilaire)
                shape: (batch_size,)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0] // 2

        self.embedding1 = y_pred[0::2]
        self.embedding2 = y_pred[1::2]

        self.distance = np.sqrt(np.sum(np.square(self.embedding1 - self.embedding2), axis=1) + 1e-7)

        similar_loss = y_true * np.square(self.distance)
        dissimilar_loss = (1 - y_true) * np.square(np.maximum(0, self.margin - self.distance))

        self.loss_value = self._apply_reduction(similar_loss + dissimilar_loss)
        return self.loss_value

    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.distance is None:
            raise RuntimeError("Must call forward() before backward()")

        gradient = np.zeros_like(self.y_pred, dtype=float)

        for i in range(self.batch_size):
            dist = self.distance[i]
            diff = self.embedding1[i] - self.embedding2[i]

            if self.y_true[i] == 1:
                grad = 2 * diff
            elif dist < self.margin:
                grad = -2 * (self.margin - dist) * diff / dist
            else:
                grad = np.zeros_like(diff)

            gradient[2*i] += grad
            gradient[2*i + 1] += -grad

        return self._apply_reduction_to_gradient(gradient)

    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config
