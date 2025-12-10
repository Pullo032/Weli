"""
Fonctions de perte avancées.
"""
import numpy as np
from typing import Optional
from .base import Loss

class KLDivergence(Loss):
    """
    Kullback-Leibler Divergence.
    
    Formule: L = Σ p * log(p / q)
    Gradient: dL/dq = -p / q
    """
    
    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'KLDivergence', reduction)
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        y_pred: distribution q
        y_true: distribution p
        """
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Clip pour éviter log(0) et division par 0
        p = np.clip(y_true, self.epsilon, 1)
        q = np.clip(y_pred, self.epsilon, 1)
        
        # Normaliser pour que les distributions somment à 1
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)
        
        # Re-clip après normalisation
        p = np.clip(p, self.epsilon, 1)
        q = np.clip(q, self.epsilon, 1)
        
        # Calcul de la KL divergence
        kl = p * np.log(p / q)
        
        # Somme sur les features
        kl_sum = np.sum(kl, axis=1)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(kl_sum)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # Clip pour éviter division par 0
        p = np.clip(self.y_true, self.epsilon, 1)
        q = np.clip(self.y_pred, self.epsilon, 1)
        
        # Normaliser
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)
        
        # Re-clip après normalisation
        p = np.clip(p, self.epsilon, 1)
        q = np.clip(q, self.epsilon, 1)
        
        # Gradient: dL/dq = -p / q
        gradient = -p / q
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

class PoissonLoss(Loss):
    """
    Poisson Loss (pour données de comptage).
    
    Formule: L = y_pred - y_true * log(y_pred)
    Gradient: dL/dy_pred = 1 - y_true / y_pred
    """
    
    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'PoissonLoss', reduction)
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Clip pour éviter log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, None)
        
        # Calcul de la poisson loss
        poisson = y_pred_clipped - y_true * np.log(y_pred_clipped)
        
        # Somme sur les dimensions si nécessaire
        if poisson.ndim > 1:
            poisson = np.sum(poisson, axis=1)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(poisson)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # Clip pour éviter division par 0
        y_pred_clipped = np.clip(self.y_pred, self.epsilon, None)
        
        # Gradient: dL/dy_pred = 1 - y_true / y_pred
        gradient = 1 - self.y_true / y_pred_clipped
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

class CosineSimilarityLoss(Loss):
    """
    Cosine Similarity Loss.
    
    Formule: L = -cosine_similarity(y_true, y_pred)
    Gradient: dL/dy_pred = (y_true * cos - y_pred) / (||y_true|| * ||y_pred||)
    """
    
    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'CosineSimilarityLoss', reduction)
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Calcul des normes
        norm_pred = np.linalg.norm(y_pred, axis=1, keepdims=True)
        norm_true = np.linalg.norm(y_true, axis=1, keepdims=True)
        
        # Clip pour éviter division par 0
        norm_pred = np.clip(norm_pred, self.epsilon, None)
        norm_true = np.clip(norm_true, self.epsilon, None)
        
        # Normaliser
        y_pred_norm = y_pred / norm_pred
        y_true_norm = y_true / norm_true
        
        # Calcul de la similarité cosinus
        cosine_sim = np.sum(y_pred_norm * y_true_norm, axis=1)
        
        # Perte: 1 - cosine similarity (pour minimisation)
        loss = 1 - cosine_sim
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # Calcul des normes
        norm_pred = np.linalg.norm(self.y_pred, axis=1, keepdims=True)
        norm_true = np.linalg.norm(self.y_true, axis=1, keepdims=True)
        
        # Clip pour éviter division par 0
        norm_pred = np.clip(norm_pred, self.epsilon, None)
        norm_true = np.clip(norm_true, self.epsilon, None)
        
        # Calcul de la similarité cosinus
        dot_product = np.sum(self.y_pred * self.y_true, axis=1, keepdims=True)
        cosine_sim = dot_product / (norm_pred * norm_true + self.epsilon)
        
        # Gradient: dL/dy_pred = (y_true - cos * y_pred / norm_pred) / norm_pred
        gradient = (self.y_true / norm_true - 
                   cosine_sim * self.y_pred / norm_pred) / norm_pred
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

class LogCoshLoss(Loss):
    """
    Log-Cosh Loss (lisse approximation de MAE).
    
    Formule: L = log(cosh(y_pred - y_true))
    Gradient: dL/dy_pred = tanh(y_pred - y_true)
    """
    
    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'LogCoshLoss', reduction)
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Différence
        diff = y_pred - y_true
        
        # Calcul de log(cosh(x)) stable numériquement
        # Pour grand |x|: log(cosh(x)) ≈ |x| - log(2)
        # Pour petit |x|: utiliser la formule directe
        
        # Séparer les grands et petits |x|
        large_x = np.abs(diff) > 50
        small_x = ~large_x
        
        loss = np.zeros_like(diff)
        
        # Pour grand |x|: log(cosh(x)) ≈ |x| - log(2)
        loss[large_x] = np.abs(diff[large_x]) - np.log(2)
        
        # Pour petit |x|: log(cosh(x)) = log((exp(x) + exp(-x)) / 2)
        # Version stable: log(cosh(x)) = x + log(1 + exp(-2x)) - log(2)
        x_small = diff[small_x]
        loss[small_x] = x_small + np.log1p(np.exp(-2 * np.abs(x_small))) - np.log(2)
        
        # Somme sur les dimensions si nécessaire
        if loss.ndim > 1:
            loss = np.sum(loss, axis=1)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # Différence
        diff = self.y_pred - self.y_true
        
        # Gradient: dL/dy_pred = tanh(diff)
        # Version stable pour grands |diff|
        gradient = np.tanh(np.clip(diff, -10, 10))  # Clip pour éviter overflow
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

class FocalLoss(Loss):
    """
    Focal Loss pour déséquilibre de classes.
    
    Formule: L = -α(1 - p)^γ * log(p) où p est la probabilité prédite
    Gradient: plus complexe, dérivée de la formule ci-dessus
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 from_logits: bool = False, name: Optional[str] = None, 
                 reduction: str = 'mean'):
        super().__init__(name or 'FocalLoss', reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        
        # Appliquer softmax si from_logits=True
        if self.from_logits:
            y_pred_probs = self._softmax(y_pred)
            self.y_pred_probs = y_pred_probs
        else:
            y_pred_probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            self.y_pred_probs = y_pred_probs
        
        # Convertir y_true en one-hot si nécessaire
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((batch_size, num_classes))
            y_true_one_hot[np.arange(batch_size), y_true.astype(int)] = 1
        else:
            y_true_one_hot = y_true
        
        # Calcul des probabilités pour la classe vraie
        pt = np.sum(y_true_one_hot * y_pred_probs, axis=1)
        pt = np.clip(pt, self.epsilon, 1 - self.epsilon)
        
        # Calcul du modulating factor (1 - pt)^gamma
        modulating_factor = np.power(1 - pt, self.gamma)
        
        # Calcul du focal loss
        focal_loss = -self.alpha * modulating_factor * np.log(pt)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(focal_loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        num_classes = self.y_pred.shape[1]
        
        # Convertir y_true en one-hot si nécessaire
        if len(self.y_true.shape) == 1:
            y_true_one_hot = np.zeros((batch_size, num_classes))
            y_true_one_hot[np.arange(batch_size), self.y_true.astype(int)] = 1
        else:
            y_true_one_hot = self.y_true
        
        # Probabilités prédites
        p = self.y_pred_probs
        
        # Calcul de pt (probabilité pour la classe vraie)
        pt = np.sum(y_true_one_hot * p, axis=1, keepdims=True)
        pt = np.clip(pt, self.epsilon, 1 - self.epsilon)
        
        # Facteur de modulation
        modulating_factor = np.power(1 - pt, self.gamma - 1)
        
        if self.from_logits:
            # Gradient pour logits
            common_term = self.alpha * modulating_factor * \
                         (self.gamma * pt * np.log(pt) + pt - 1)
            
            gradient = p * common_term
            gradient -= y_true_one_hot * self.alpha * modulating_factor * \
                       (self.gamma * np.log(pt) + 1) * p
            gradient += y_true_one_hot * self.alpha * modulating_factor * \
                       (self.gamma * np.log(pt) + 1) * p * p
            
        else:
            # Gradient pour probabilités
            gradient = -self.alpha * modulating_factor * \
                      (self.gamma * np.log(pt) + 1) * y_true_one_hot / p
            gradient += self.alpha * modulating_factor * \
                       np.power(1 - pt, self.gamma) / (1 - p)
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient
    
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
    
    Formule: L = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    Gradient: dérivée de la formule ci-dessus
    """
    
    def __init__(self, smooth: float = 1.0, 
                 name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'DiceLoss', reduction)
        self.smooth = smooth
        self.epsilon = 1e-7
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Clip pour éviter les problèmes numériques
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Flatten pour le calcul
        if y_pred.ndim > 2:
            # Pour les images: (batch, height, width, channels)
            y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
            y_true_flat = y_true.reshape(y_true.shape[0], -1)
        else:
            y_pred_flat = y_pred
            y_true_flat = y_true
        
        # Calcul de l'intersection et des unions
        intersection = np.sum(y_pred_flat * y_true_flat, axis=1)
        pred_sum = np.sum(y_pred_flat, axis=1)
        true_sum = np.sum(y_true_flat, axis=1)
        
        # Calcul du dice coefficient
        dice = (2 * intersection + self.smooth) / \
               (pred_sum + true_sum + self.smooth + self.epsilon)
        
        # Dice loss: 1 - dice coefficient
        dice_loss = 1 - dice
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(dice_loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # Clip pour éviter les problèmes numériques
        y_pred = np.clip(self.y_pred, self.epsilon, 1 - self.epsilon)
        
        # Flatten pour le calcul
        if y_pred.ndim > 2:
            # Pour les images
            original_shape = y_pred.shape
            y_pred_flat = y_pred.reshape(batch_size, -1)
            y_true_flat = self.y_true.reshape(batch_size, -1)
        else:
            y_pred_flat = y_pred
            y_true_flat = self.y_true
        
        # Calcul de l'intersection et des unions
        intersection = np.sum(y_pred_flat * y_true_flat, axis=1, keepdims=True)
        pred_sum = np.sum(y_pred_flat, axis=1, keepdims=True)
        true_sum = np.sum(y_true_flat, axis=1, keepdims=True)
        
        # Calcul des termes pour le gradient
        numerator = 2 * (true_sum * intersection - y_true_flat * pred_sum)
        denominator = np.square(pred_sum + true_sum + self.smooth)
        
        # Gradient
        gradient_flat = -2 * numerator / (denominator + self.epsilon)
        
        # Reshape si nécessaire
        if self.y_pred.ndim > 2:
            gradient = gradient_flat.reshape(original_shape)
        else:
            gradient = gradient_flat
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient
    
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
        # Stocker pour backward
        self.y_pred = y_pred
        self.batch_size = y_pred.shape[0] // 3
        
        # Séparer anchor, positive, negative
        anchor = y_pred[0::3]      # indices: 0, 3, 6, ...
        positive = y_pred[1::3]    # indices: 1, 4, 7, ...
        negative = y_pred[2::3]    # indices: 2, 5, 8, ...
        
        # Calcul des distances
        if self.distance == 'euclidean':
            pos_dist = np.sum(np.square(anchor - positive), axis=1)
            neg_dist = np.sum(np.square(anchor - negative), axis=1)
        elif self.distance == 'cosine':
            # Distance cosinus: 1 - similarité cosinus
            pos_sim = np.sum(anchor * positive, axis=1) / \
                     (np.linalg.norm(anchor, axis=1) * np.linalg.norm(positive, axis=1) + self.epsilon)
            neg_sim = np.sum(anchor * negative, axis=1) / \
                     (np.linalg.norm(anchor, axis=1) * np.linalg.norm(negative, axis=1) + self.epsilon)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        else:
            raise ValueError(f"Distance '{self.distance}' non supportée")
        
        # Calcul de la triplet loss
        losses = np.maximum(0, pos_dist - neg_dist + self.margin)
        
        # Identifier les triplets valides (loss > 0)
        self.valid_triplets = losses > 0
        self.pos_dist = pos_dist
        self.neg_dist = neg_dist
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(losses)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.valid_triplets is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.batch_size
        gradient = np.zeros_like(self.y_pred)
        
        if self.distance == 'euclidean':
            # Gradients pour la distance euclidienne
            for i in range(batch_size):
                if self.valid_triplets[i]:
                    # Gradient pour anchor
                    grad_anchor = 2 * (self.negative[i] - self.positive[i])
                    
                    # Gradient pour positive
                    grad_positive = 2 * (self.positive[i] - self.anchor[i])
                    
                    # Gradient pour negative
                    grad_negative = 2 * (self.anchor[i] - self.negative[i])
                    
                    # Assigner les gradients
                    gradient[3*i] += grad_anchor      # anchor
                    gradient[3*i + 1] += grad_positive  # positive
                    gradient[3*i + 2] += grad_negative  # negative
        
        elif self.distance == 'cosine':
            # Gradients pour la distance cosinus
            for i in range(batch_size):
                if self.valid_triplets[i]:
                    norm_a = np.linalg.norm(self.anchor[i])
                    norm_p = np.linalg.norm(self.positive[i])
                    norm_n = np.linalg.norm(self.negative[i])
                    
                    # Similarités
                    sim_ap = np.dot(self.anchor[i], self.positive[i]) / (norm_a * norm_p + self.epsilon)
                    sim_an = np.dot(self.anchor[i], self.negative[i]) / (norm_a * norm_n + self.epsilon)
                    
                    # Gradient pour anchor
                    grad_anchor = (
                        (self.positive[i] / (norm_a * norm_p) - 
                         self.anchor[i] * sim_ap / (norm_a * norm_a)) -
                        (self.negative[i] / (norm_a * norm_n) - 
                         self.anchor[i] * sim_an / (norm_a * norm_a))
                    )
                    
                    # Gradient pour positive
                    grad_positive = (
                        self.anchor[i] / (norm_a * norm_p) - 
                        self.positive[i] * sim_ap / (norm_p * norm_p)
                    )
                    
                    # Gradient pour negative (négatif car on soustrait)
                    grad_negative = -(
                        self.anchor[i] / (norm_a * norm_n) - 
                        self.negative[i] * sim_an / (norm_n * norm_n)
                    )
                    
                    # Assigner les gradients
                    gradient[3*i] += grad_anchor
                    gradient[3*i + 1] += grad_positive
                    gradient[3*i + 2] += grad_negative
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient
    
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
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        y_pred: embeddings de shape (batch_size * 2, embedding_dim)
                [sample1_A, sample1_B, sample2_A, sample2_B, ...]
        y_true: labels de similarité (1 pour similaire, 0 pour dissimilaire)
                shape: (batch_size,)
        """
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0] // 2
        
        # Séparer les paires
        embedding1 = y_pred[0::2]  # indices: 0, 2, 4, ...
        embedding2 = y_pred[1::2]  # indices: 1, 3, 5, ...
        
        # Calcul de la distance euclidienne
        distance = np.sqrt(np.sum(np.square(embedding1 - embedding2), axis=1) + 1e-7)
        
        # Calcul de la loss pour paires similaires
        similar_loss = y_true * np.square(distance)
        
        # Calcul de la loss pour paires dissimilaires
        dissimilar_loss = (1 - y_true) * np.square(np.maximum(0, self.margin - distance))
        
        # Loss totale
        loss = similar_loss + dissimilar_loss
        
        # Stocker pour backward
        self.distance = distance
        self.embedding1 = embedding1
        self.embedding2 = embedding2
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.distance is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.batch_size
        gradient = np.zeros_like(self.y_pred)
        
        for i in range(batch_size):
            dist = self.distance[i]
            diff = self.embedding1[i] - self.embedding2[i]
            
            if self.y_true[i] == 1:  # Paire similaire
                # Gradient pour paire similaire: 2 * (x1 - x2)
                grad = 2 * diff
                
            else:  # Paire dissimilaire
                if dist < self.margin:
                    # Gradient pour paire dissimilaire: -2 * (margin - dist) * (x1 - x2) / dist
                    grad = -2 * (self.margin - dist) * diff / (dist + 1e-7)
                else:
                    grad = np.zeros_like(diff)
            
            # Assigner les gradients
            gradient[2*i] += grad      # embedding1
            gradient[2*i + 1] += -grad  # embedding2 (symétrique)
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin
        return config