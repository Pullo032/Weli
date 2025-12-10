"""
Implémentation de RMSprop (Root Mean Square Propagation) pour Weli.
"""
import numpy as np
from typing import Optional
from .base import Optimizer


class RMSprop(Optimizer):
    """
    Optimiseur RMSprop (Root Mean Square Propagation).
    
    RMSprop adapte le learning rate pour chaque paramètre en utilisant une
    moyenne mobile exponentielle des carrés des gradients.
    
    Formule RMSprop:
        E[g²]_t = rho * E[g²]_{t-1} + (1 - rho) * g²_t
        param = param - lr * g_t / (sqrt(E[g²]_t) + epsilon)
    
    Args:
        lr: Learning rate (taux d'apprentissage). Default: 0.001
        rho: Facteur de décroissance pour la moyenne mobile. Default: 0.9
        epsilon: Petite valeur pour éviter la division par zéro. Default: 1e-8
        decay: Taux de décroissance du learning rate. Default: 0.0
        clipnorm: Valeur maximale pour le gradient clipping. Default: None
        clipvalue: Valeur maximale absolue pour le gradient clipping. Default: None
        name: Nom de l'optimiseur. Default: None
    
    Exemple:
        >>> from weli.optimizers import RMSprop
        >>> optimizer = RMSprop(lr=0.001, rho=0.9)
        >>> model.compile(optimizer=optimizer, ...)
    
    Référence:
        Tieleman, T. & Hinton, G. (2012). Lecture 6.5 - rmsprop: Divide the
        gradient by a running average of its recent magnitude.
    """
    
    def __init__(self,
                 lr: float = 0.001,
                 rho: float = 0.9,
                 epsilon: float = 1e-8,
                 decay: float = 0.0,
                 clipnorm: Optional[float] = None,
                 clipvalue: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur RMSprop.
        """
        super().__init__(
            lr=lr,
            momentum=0.0,  # RMSprop n'utilise pas le momentum classique
            decay=decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            name=name or 'RMSprop'
        )
        
        if rho < 0 or rho >= 1:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        self.rho = float(rho)
        self.epsilon = float(epsilon)
        
        # Stockage des moyennes mobiles des carrés des gradients
        # Structure: {layer_id: {param_name: mean_square}}
        self.mean_square = {}
    
    def _get_layer_id(self, layer) -> int:
        """
        Génère un ID unique pour une couche.
        
        Args:
            layer: Couche du réseau
            
        Returns:
            ID unique de la couche
        """
        return id(layer)
    
    def _init_mean_square(self, layer_id: int, param_name: str, param_shape: tuple) -> None:
        """
        Initialise la moyenne mobile des carrés des gradients.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            param_shape: Shape du paramètre
        """
        if layer_id not in self.mean_square:
            self.mean_square[layer_id] = {}
        
        if param_name not in self.mean_square[layer_id]:
            self.mean_square[layer_id][param_name] = np.zeros(param_shape, dtype=np.float32)
    
    def _get_mean_square(self, layer_id: int, param_name: str) -> np.ndarray:
        """
        Récupère la moyenne mobile des carrés des gradients.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Moyenne mobile des carrés des gradients
        """
        return self.mean_square.get(layer_id, {}).get(param_name, None)
    
    def _update_mean_square(self, layer_id: int, param_name: str, value: np.ndarray) -> None:
        """
        Met à jour la moyenne mobile des carrés des gradients.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            value: Nouvelle valeur de la moyenne mobile
        """
        if layer_id not in self.mean_square:
            self.mean_square[layer_id] = {}
        self.mean_square[layer_id][param_name] = value
    
    def update_parameter(self,
                        param: np.ndarray,
                        grad: np.ndarray,
                        layer_id: int,
                        param_name: str) -> np.ndarray:
        """
        Met à jour un paramètre selon l'algorithme RMSprop.
        
        Args:
            param: Paramètre actuel
            grad: Gradient du paramètre (déjà clippé si nécessaire)
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Nouveau paramètre mis à jour
        """
        current_lr = self.get_lr()
        
        # Initialiser la moyenne mobile si nécessaire
        self._init_mean_square(layer_id, param_name, param.shape)
        
        # Récupérer la moyenne mobile actuelle
        mean_sq = self._get_mean_square(layer_id, param_name)
        
        if mean_sq is None:
            # Si pas de moyenne, initialiser à zéro
            mean_sq = np.zeros_like(param)
        
        # Mettre à jour la moyenne mobile des carrés des gradients
        # E[g²]_t = rho * E[g²]_{t-1} + (1 - rho) * g²_t
        mean_sq = self.rho * mean_sq + (1.0 - self.rho) * (grad ** 2)
        
        # Sauvegarder la nouvelle moyenne mobile
        self._update_mean_square(layer_id, param_name, mean_sq)
        
        # Mettre à jour le paramètre
        # param = param - lr * g_t / (sqrt(E[g²]_t) + epsilon)
        updated_param = param - current_lr * grad / (np.sqrt(mean_sq) + self.epsilon)
        
        return updated_param
    
    def zero_grad(self) -> None:
        """
        Réinitialise les moyennes mobiles des carrés des gradients.
        Utile si vous voulez réinitialiser l'état de l'optimiseur.
        """
        super().zero_grad()  # Réinitialise aussi les velocities de la classe de base
        self.mean_square = {}
    
    def get_config(self) -> dict:
        """
        Retourne la configuration de l'optimiseur RMSprop.
        
        Returns:
            Dictionnaire contenant la configuration
        """
        config = super().get_config()
        config['rho'] = self.rho
        config['epsilon'] = self.epsilon
        return config
    
    def set_config(self, config: dict) -> None:
        """
        Configure l'optimiseur RMSprop à partir d'un dictionnaire.
        
        Args:
            config: Dictionnaire de configuration
        """
        super().set_config(config)
        if 'rho' in config:
            self.rho = float(config['rho'])
        if 'epsilon' in config:
            self.epsilon = float(config['epsilon'])
    
    def __repr__(self) -> str:
        """
        Représentation string de l'optimiseur RMSprop.
        """
        config = self.get_config()
        return (
            f"RMSprop("
            f"lr={config['lr']}, "
            f"rho={config['rho']}, "
            f"epsilon={config['epsilon']}, "
            f"decay={config['decay']}, "
            f"iterations={config['iterations']})"
        )

