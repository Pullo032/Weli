"""
Implémentation de SGD (Stochastic Gradient Descent) pour Weli.
"""
import numpy as np
from typing import Optional
from .base import Optimizer


class SGD(Optimizer):
    """
    Optimiseur SGD (Stochastic Gradient Descent) avec support du momentum.
    
    Formule SGD simple:
        param = param - lr * grad
    
    Formule SGD avec momentum:
        velocity = momentum * velocity - lr * grad
        param = param + velocity
    
    Args:
        lr: Learning rate (taux d'apprentissage). Default: 0.01
        momentum: Coefficient de momentum. Si > 0, utilise le momentum. Default: 0.0
        decay: Taux de décroissance du learning rate. Default: 0.0
        nesterov: Si True, utilise Nesterov accelerated gradient. Default: False
        clipnorm: Valeur maximale pour le gradient clipping. Default: None
        clipvalue: Valeur maximale absolue pour le gradient clipping. Default: None
        name: Nom de l'optimiseur. Default: None
    
    Exemple:
        >>> from weli.optimizers import SGD
        >>> optimizer = SGD(lr=0.01, momentum=0.9)
        >>> model.compile(optimizer=optimizer, ...)
    """
    
    def __init__(self,
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 decay: float = 0.0,
                 nesterov: bool = False,
                 clipnorm: Optional[float] = None,
                 clipvalue: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur SGD.
        """
        super().__init__(
            lr=lr,
            momentum=momentum,
            decay=decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            name=name or 'SGD'
        )
        
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov momentum requires a momentum value > 0")
        
        self.nesterov = bool(nesterov)
    
    def update_parameter(self,
                        param: np.ndarray,
                        grad: np.ndarray,
                        layer_id: int,
                        param_name: str) -> np.ndarray:
        """
        Met à jour un paramètre selon l'algorithme SGD.
        
        Args:
            param: Paramètre actuel
            grad: Gradient du paramètre (déjà clippé si nécessaire)
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Nouveau paramètre mis à jour
        """
        current_lr = self.get_lr()
        
        # SGD simple sans momentum
        if self.momentum == 0.0:
            # Mise à jour directe: param = param - lr * grad
            updated_param = param - current_lr * grad
            return updated_param
        
        # SGD avec momentum
        # Initialiser la vitesse si nécessaire
        self._init_velocity(layer_id, param_name, param.shape)
        
        # Récupérer la vitesse actuelle
        velocity = self._get_velocity(layer_id, param_name)
        
        if velocity is None:
            # Si pas de vitesse, initialiser à zéro
            velocity = np.zeros_like(param)
        
        # Mise à jour de la vitesse: v = momentum * v - lr * grad
        velocity = self.momentum * velocity - current_lr * grad
        
        # Sauvegarder la nouvelle vitesse
        self._update_velocity(layer_id, param_name, velocity)
        
        # Mise à jour du paramètre
        if self.nesterov:
            # Nesterov accelerated gradient
            # On utilise la vitesse mise à jour pour calculer le gradient
            # param = param + momentum * velocity - lr * grad
            updated_param = param + self.momentum * velocity - current_lr * grad
        else:
            # Momentum classique: param = param + velocity
            updated_param = param + velocity
        
        return updated_param
    
    def get_config(self) -> dict:
        """
        Retourne la configuration de l'optimiseur SGD.
        
        Returns:
            Dictionnaire contenant la configuration
        """
        config = super().get_config()
        config['nesterov'] = self.nesterov
        return config
    
    def set_config(self, config: dict) -> None:
        """
        Configure l'optimiseur SGD à partir d'un dictionnaire.
        
        Args:
            config: Dictionnaire de configuration
        """
        super().set_config(config)
        if 'nesterov' in config:
            self.nesterov = bool(config['nesterov'])
    
    def __repr__(self) -> str:
        """
        Représentation string de l'optimiseur SGD.
        """
        config = self.get_config()
        nesterov_str = ", nesterov=True" if self.nesterov else ""
        return (
            f"SGD("
            f"lr={config['lr']}, "
            f"momentum={config['momentum']}, "
            f"decay={config['decay']}"
            f"{nesterov_str}, "
            f"iterations={config['iterations']})"
        )

