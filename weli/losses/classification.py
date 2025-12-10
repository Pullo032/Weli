"""
Fonctions de perte pour la classification.
"""
import numpy as np
from typing import Optional
from .base import Loss

class CrossEntropy(Loss):
    """
    Cross Entropy Loss (Entropie Croisée).
    
    Formule: L = -Σ y_true * log(y_pred)
    
    Args:
        from_logits: Si True, y_pred sont des logits (avant softmax)
        label_smoothing: Niveau de lissage des labels (0.0 à 1.0)
        ignore_index: Index à ignorer (pour segmentation)
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
        
        # Cache pour backward
        self.y_pred_probs = None
        self.y_true_smoothed = None
        self.valid_mask = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calcule la cross-entropy.
        """
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1] if y_pred.ndim > 1 else 1
        
        # Appliquer softmax si from_logits
        if self.from_logits:
            y_pred_probs = self._softmax(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Préparer les labels
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Labels entiers -> convertir en one-hot
            y_true_one_hot = np.zeros((batch_size, num_classes))
            indices = y_true.astype(int).flatten()
            y_true_one_hot[np.arange(batch_size), indices] = 1
        else:
            # Déjà en one-hot
            y_true_one_hot = y_true.copy()
        
        # Appliquer label smoothing
        if self.label_smoothing > 0:
            y_true_smoothed = (1 - self.label_smoothing) * y_true_one_hot + \
                             self.label_smoothing / num_classes
            self.y_true_smoothed = y_true_smoothed
        else:
            self.y_true_smoothed = y_true_one_hot
        
        # Calcul de la cross-entropy
        log_probs = np.log(y_pred_probs + self.epsilon)
        ce_per_sample = -np.sum(self.y_true_smoothed * log_probs, axis=1)
        
        # Gérer ignore_index
        if self.ignore_index != -100 and y_true.ndim == 1:
            self.valid_mask = y_true.flatten() != self.ignore_index
            if np.any(self.valid_mask):
                ce_per_sample = ce_per_sample[self.valid_mask]
            else:
                ce_per_sample = np.array([0.0])
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(ce_per_sample)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        """
        Gradient: dL/dy_pred = y_pred - y_true (après softmax)
        """
        self._check_backward_prerequisites()
        
        batch_size = self.y_pred.shape[0]
        
        if self.from_logits:
            # Gradient pour softmax + cross-entropy
            gradient = self.y_pred_probs - self.y_true_smoothed
        else:
            # Gradient pour probabilités directement
            gradient = -self.y_true_smoothed / (self.y_pred_probs + self.epsilon)
        
        # Appliquer ignore_index
        if self.ignore_index != -100 and self.y_true.ndim == 1 and self.valid_mask is not None:
            full_gradient = np.zeros_like(gradient)
            if np.any(self.valid_mask):
                full_gradient[self.valid_mask] = gradient[self.valid_mask]
            gradient = full_gradient
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
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
        
        # Cache
        self.y_pred_probs = None
        self.y_true_smoothed = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        # Appliquer sigmoid si from_logits
        if self.from_logits:
            y_pred_probs = self._sigmoid(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Appliquer label smoothing
        if self.label_smoothing > 0:
            y_true_smoothed = (1 - self.label_smoothing) * y_true + \
                             self.label_smoothing * 0.5
            self.y_true_smoothed = y_true_smoothed
        else:
            self.y_true_smoothed = y_true.copy()
        
        # Calcul de la binary cross-entropy
        bce = -(self.y_true_smoothed * np.log(y_pred_probs + self.epsilon) +
                (1 - self.y_true_smoothed) * np.log(1 - y_pred_probs + self.epsilon))
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(bce)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        if self.from_logits:
            # Gradient pour sigmoid + BCE
            gradient = self.y_pred_probs - self.y_true_smoothed
        else:
            # Gradient pour probabilités
            gradient = (self.y_pred_probs - self.y_true_smoothed) / \
                      ((self.y_pred_probs + self.epsilon) * 
                       (1 - self.y_pred_probs + self.epsilon))
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
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
        
        # Cache
        self.y_pred_probs = None
        self.y_true_indices = None
        self.valid_mask = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        
        # Convertir y_true en indices si nécessaire
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            # One-hot -> indices
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            # Déjà indices
            y_true_indices = y_true.astype(int).flatten()
        
        self.y_true_indices = y_true_indices
        
        # Vérifier les indices
        if np.any(y_true_indices < 0) or np.any(y_true_indices >= num_classes):
            invalid = np.where((y_true_indices < 0) | (y_true_indices >= num_classes))[0]
            if len(invalid) > 10:  # Limiter l'affichage
                invalid = list(invalid[:10]) + ['...']
            raise ValueError(f"Invalid class indices at positions {invalid}")
        
        # Appliquer softmax si from_logits
        if self.from_logits:
            y_pred_probs = self._softmax(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Sélectionner les probabilités des vraies classes
        true_class_probs = y_pred_probs[np.arange(batch_size), y_true_indices]
        
        # Calcul de la cross-entropy sparse
        ce_per_sample = -np.log(true_class_probs + self.epsilon)
        
        # Gérer ignore_index
        if self.ignore_index != -100:
            self.valid_mask = y_true_indices != self.ignore_index
            if np.any(self.valid_mask):
                ce_per_sample = ce_per_sample[self.valid_mask]
            else:
                ce_per_sample = np.array([0.0])
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(ce_per_sample)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        batch_size = self.y_pred.shape[0]
        num_classes = self.y_pred.shape[1]
        
        # Initialiser le gradient
        gradient = np.zeros_like(self.y_pred)
        
        if self.from_logits:
            # Gradient pour softmax + sparse CE
            gradient[np.arange(batch_size), self.y_true_indices] = 1
            gradient = self.y_pred_probs - gradient
        else:
            # Gradient pour probabilités
            indices = np.arange(batch_size)
            gradient[indices, self.y_true_indices] = \
                -1 / (self.y_pred_probs[indices, self.y_true_indices] + self.epsilon)
        
        # Appliquer ignore_index
        if self.ignore_index != -100 and self.valid_mask is not None:
            # Mettre à zéro les gradients des indices ignorés
            ignore_mask = ~self.valid_mask
            if np.any(ignore_mask):
                gradient[ignore_mask] = 0
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
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
    
    Formule: L = max(0, 1 - y_true * y_pred)
    où y_true ∈ {-1, 1}
    """
    
    def __init__(self, margin: float = 1.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'HingeLoss', reduction)
        
        self.margin = margin
        
        # Cache
        self.mask = None
        self.y_true_transformed = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        # Convertir y_true de {0, 1} à {-1, 1}
        y_true_transformed = 2 * y_true - 1
        self.y_true_transformed = y_true_transformed
        
        # Calcul de la hinge loss
        hinge = np.maximum(0, self.margin - y_true_transformed * y_pred)
        
        # Masque pour les points avec perte > 0
        self.mask = hinge > 0
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(hinge)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        # Initialiser le gradient
        gradient = np.zeros_like(self.y_pred)
        
        # Gradient pour hinge standard
        gradient[self.mask] = -self.y_true_transformed[self.mask]
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config


class SquaredHingeLoss(Loss):
    """
    Squared Hinge Loss.
    
    Formule: L = max(0, 1 - y_true * y_pred)²
    où y_true ∈ {-1, 1}
    """
    
    def __init__(self, margin: float = 1.0,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'SquaredHingeLoss', reduction)
        
        self.margin = margin
        
        # Cache
        self.mask = None
        self.y_true_transformed = None
        self.hinge_values = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        # Convertir y_true de {0, 1} à {-1, 1}
        y_true_transformed = 2 * y_true - 1
        self.y_true_transformed = y_true_transformed
        
        # Calcul de la hinge loss
        hinge = np.maximum(0, self.margin - y_true_transformed * y_pred)
        self.hinge_values = hinge
        
        # Squared hinge
        squared_hinge = np.square(hinge)
        
        # Masque pour les points avec perte > 0
        self.mask = hinge > 0
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(squared_hinge)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        # Initialiser le gradient
        gradient = np.zeros_like(self.y_pred)
        
        # Gradient pour squared hinge
        if np.any(self.mask):
            gradient[self.mask] = -2 * self.y_true_transformed[self.mask] * \
                                 (self.margin - self.y_true_transformed[self.mask] * self.y_pred[self.mask])
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config


class FocalLoss(Loss):
    """
    Focal Loss pour déséquilibre de classes.
    
    Formule: L = -α(1 - p)^γ * log(p)
    où p est la probabilité de la vraie classe.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 from_logits: bool = False,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'FocalLoss', reduction)
        
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
        # Cache
        self.y_pred_probs = None
        self.y_true_one_hot = None
        self.pt = None  # probabilité de la vraie classe
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        
        # Appliquer softmax si from_logits
        if self.from_logits:
            y_pred_probs = self._softmax(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Préparer les labels
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Labels entiers -> one-hot
            y_true_one_hot = np.zeros((batch_size, num_classes))
            indices = y_true.astype(int).flatten()
            y_true_one_hot[np.arange(batch_size), indices] = 1
            self.y_true_one_hot = y_true_one_hot
        else:
            # Déjà en one-hot
            self.y_true_one_hot = y_true
        
        # Probabilité de la vraie classe
        pt = np.sum(self.y_true_one_hot * y_pred_probs, axis=1)
        pt = np.clip(pt, self.epsilon, 1 - self.epsilon)
        self.pt = pt
        
        # Facteur de modulation (1 - pt)^gamma
        modulating_factor = np.power(1 - pt, self.gamma)
        
        # Calcul de la focal loss
        focal_loss = -self.alpha * modulating_factor * np.log(pt)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(focal_loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        batch_size = self.y_pred.shape[0]
        
        # Calcul des termes du gradient
        if self.from_logits:
            # Gradient pour logits
            common_term = self.alpha * np.power(1 - self.pt, self.gamma - 1) * \
                         (self.gamma * self.pt * np.log(self.pt) + self.pt - 1)
            
            gradient = self.y_pred_probs * common_term[:, np.newaxis]
            gradient -= self.y_true_one_hot * self.alpha * np.power(1 - self.pt, self.gamma - 1) * \
                       (self.gamma * np.log(self.pt) + 1)[:, np.newaxis] * self.y_pred_probs
            gradient += self.y_true_one_hot * self.alpha * np.power(1 - self.pt, self.gamma - 1) * \
                       (self.gamma * np.log(self.pt) + 1)[:, np.newaxis] * \
                       self.y_pred_probs * self.y_pred_probs
        else:
            # Gradient pour probabilités
            gradient = -self.alpha * np.power(1 - self.pt, self.gamma - 1) * \
                      (self.gamma * np.log(self.pt) + 1)[:, np.newaxis] * \
                      self.y_true_one_hot / (self.y_pred_probs + self.epsilon)
            gradient += self.alpha * np.power(1 - self.pt, self.gamma) / \
                       (1 - self.y_pred_probs + self.epsilon)
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
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


class KLDivergence(Loss):
    """
    Kullback-Leibler Divergence (pour régularisation).
    
    Formule: L = Σ p * log(p / q)
    où p est la distribution cible, q est la distribution prédite.
    """
    
    def __init__(self, name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'KLDivergence', reduction)
        
        # Cache
        self.p_normalized = None
        self.q_normalized = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        y_pred: distribution q
        y_true: distribution p
        """
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        # Normaliser les distributions
        p = y_true / (np.sum(y_true, axis=1, keepdims=True) + self.epsilon)
        q = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + self.epsilon)
        
        # Clip pour éviter log(0)
        p = np.clip(p, self.epsilon, 1)
        q = np.clip(q, self.epsilon, 1)
        
        self.p_normalized = p
        self.q_normalized = q
        
        # Calcul de la KL divergence
        kl = p * np.log(p / q)
        kl_per_sample = np.sum(kl, axis=1)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(kl_per_sample)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        # Gradient: dL/dq = -p / q
        gradient = -self.p_normalized / (self.q_normalized + self.epsilon)
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
    def get_config(self) -> dict:
        return super().get_config()


class DiceLoss(Loss):
    """
    Dice Loss pour segmentation.
    
    Formule: L = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    
    def __init__(self, smooth: float = 1.0,
                 from_logits: bool = False,
                 name: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__(name or 'DiceLoss', reduction)
        
        self.smooth = smooth
        self.from_logits = from_logits
        
        # Cache
        self.y_pred_probs = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_inputs(y_pred, y_true)
        self._prepare_inputs(y_pred, y_true)
        
        # Appliquer sigmoid/softmax si from_logits
        if self.from_logits:
            if y_pred.shape[-1] == 1 or len(y_pred.shape) == 2:
                # Binary: sigmoid
                y_pred_probs = self._sigmoid(y_pred)
            else:
                # Multi-class: softmax
                y_pred_probs = self._softmax(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Aplatir pour le calcul
        if y_pred_probs.ndim > 2:
            # Pour les images: (batch, height, width, channels)
            y_pred_flat = y_pred_probs.reshape(y_pred_probs.shape[0], -1)
            y_true_flat = y_true.reshape(y_true.shape[0], -1)
        else:
            y_pred_flat = y_pred_probs
            y_true_flat = y_true
        
        # Calcul de l'intersection et des unions
        intersection = np.sum(y_pred_flat * y_true_flat, axis=1)
        pred_sum = np.sum(y_pred_flat, axis=1)
        true_sum = np.sum(y_true_flat, axis=1)
        
        # Calcul du dice coefficient
        dice = (2 * intersection + self.smooth) / \
               (pred_sum + true_sum + self.smooth + self.epsilon)
        
        # Dice loss
        dice_loss = 1 - dice
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(dice_loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        self._check_backward_prerequisites()
        
        batch_size = self.y_pred.shape[0]
        
        # Aplatir pour le calcul
        if self.y_pred_probs.ndim > 2:
            original_shape = self.y_pred_probs.shape
            y_pred_flat = self.y_pred_probs.reshape(batch_size, -1)
            y_true_flat = self.y_true.reshape(batch_size, -1)
        else:
            y_pred_flat = self.y_pred_probs
            y_true_flat = self.y_true
        
        # Calcul de l'intersection et des unions
        intersection = np.sum(y_pred_flat * y_true_flat, axis=1, keepdims=True)
        pred_sum = np.sum(y_pred_flat, axis=1, keepdims=True)
        true_sum = np.sum(y_true_flat, axis=1, keepdims=True)
        
        # Calcul du gradient
        numerator = 2 * (true_sum * y_pred_flat - intersection)
        denominator = np.square(pred_sum + true_sum + self.smooth)
        
        gradient_flat = numerator / (denominator + self.epsilon)
        
        # Reshape si nécessaire
        if self.y_pred_probs.ndim > 2:
            gradient = gradient_flat.reshape(original_shape)
        else:
            gradient = gradient_flat
        
        # Si from_logits, appliquer la dérivée de sigmoid
        if self.from_logits and (self.y_pred.shape[-1] == 1 or len(self.y_pred.shape) == 2):
            gradient = gradient * self.y_pred_probs * (1 - self.y_pred_probs)
        
        # Appliquer la réduction
        gradient = self._apply_reduction_to_gradient(gradient)
        
        return gradient
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 2:
            x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
            return x_exp / np.sum(x_exp, axis=1, keepdims=True)
        else:
            # Pour les tenseurs 3D+
            x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return x_exp / np.sum(x_exp, axis=-1, keepdims=True)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'from_logits': self.from_logits
        })
        return config