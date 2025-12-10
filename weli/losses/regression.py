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
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Calcul de l'erreur quadratique
        squared_error = np.square(y_pred - y_true)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(squared_error)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        # Gradient: dL/dy_pred = 2 * (y_pred - y_true) / N
        batch_size = self.y_pred.shape[0]
        gradient = 2 * (self.y_pred - self.y_true)
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

class MAE(Loss):
    """
    Mean Absolute Error (Erreur Absolue Moyenne).
    
    Formule: L = 1/N * Σ|y_pred - y_true|
    Gradient: dL/dy_pred = sign(y_pred - y_true) / N
    """
    
    def __init__(self, name: Optional[str] = None, reduction: str = 'mean'):
        super().__init__(name or 'MAE', reduction)
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Calcul de l'erreur absolue
        absolute_error = np.abs(y_pred - y_true)
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(absolute_error)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        # Gradient: dL/dy_pred = sign(y_pred - y_true) / N
        batch_size = self.y_pred.shape[0]
        gradient = np.sign(self.y_pred - self.y_true)
        
        # Éviter les valeurs infinies à zéro
        gradient[gradient == 0] = 0
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient

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
        self.mask = None  # Pour stocker le masque
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Différence
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        
        # Masque pour les petites erreurs (MSE)
        self.mask = abs_diff <= self.delta
        
        # Calcul de la perte de Huber
        loss = np.zeros_like(diff)
        
        # Partie quadratique (MSE)
        loss[self.mask] = 0.5 * np.square(diff[self.mask])
        
        # Partie linéaire (MAE)
        loss[~self.mask] = self.delta * abs_diff[~self.mask] - 0.5 * self.delta ** 2
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(loss)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None or self.mask is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        diff = self.y_pred - self.y_true
        
        # Gradient
        gradient = np.zeros_like(diff)
        
        # Partie quadratique: gradient = diff
        gradient[self.mask] = diff[self.mask]
        
        # Partie linéaire: gradient = delta * sign(diff)
        gradient[~self.mask] = self.delta * np.sign(diff[~self.mask])
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient
    
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
        # Stocker pour backward
        self.y_pred = y_pred
        self.y_true = y_true
        
        # S'assurer que les valeurs sont positives
        y_pred = np.maximum(y_pred, 0)
        y_true = np.maximum(y_true, 0)
        
        # Calcul de l'erreur logarithmique quadratique
        log_error = np.square(np.log1p(y_pred) - np.log1p(y_true))
        
        # Application de la réduction
        self.loss_value = self._apply_reduction(log_error)
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.y_pred.shape[0]
        
        # S'assurer que les valeurs sont positives
        y_pred = np.maximum(self.y_pred, 0)
        y_true = np.maximum(self.y_true, 0)
        
        # Gradient: dL/dy_pred = 2 * (log(1 + y_pred) - log(1 + y_true)) / (1 + y_pred)
        gradient = 2 * (np.log1p(y_pred) - np.log1p(y_true)) / (1 + y_pred)
        
        # Ajuster selon la réduction
        if self.reduction == 'mean':
            gradient /= batch_size
        
        return gradient