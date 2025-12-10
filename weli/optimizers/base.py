"""
Classe de base pour tous les optimiseurs de Weli.
"""
import numpy as np
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Classe de base abstraite pour tous les optimiseurs.
    
    Tous les optimiseurs doivent hériter de cette classe et implémenter
    la méthode update_parameter() pour définir leur stratégie de mise à jour.
    
    Args:
        lr: Learning rate (taux d'apprentissage). Default: 0.01
        momentum: Coefficient de momentum. Default: 0.0
        decay: Taux de décroissance du learning rate. Default: 0.0
        clipnorm: Valeur maximale pour le gradient clipping. Si None, pas de clipping. Default: None
        clipvalue: Valeur maximale absolue pour le gradient clipping. Si None, pas de clipping. Default: None
        name: Nom de l'optimiseur. Default: None
    """
    
    def __init__(self, 
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 decay: float = 0.0,
                 clipnorm: Optional[float] = None,
                 clipvalue: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur de base.
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if decay < 0:
            raise ValueError(f"Decay must be non-negative, got {decay}")
        
        self.lr = float(lr)
        self.initial_lr = float(lr)
        self.momentum = float(momentum)
        self.decay = float(decay)
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.name = name or self.__class__.__name__
        
        # Compteur d'itérations pour le decay
        self.iterations = 0
        
        # Stockage des états pour le momentum (par paramètre)
        # Structure: {layer_id: {param_name: velocity}}
        self.velocities = {}
    
    def get_lr(self) -> float:
        """
        Retourne le learning rate actuel (avec decay appliqué).
        
        Returns:
            Learning rate actuel
        """
        if self.decay > 0:
            return self.initial_lr / (1.0 + self.decay * self.iterations)
        return self.lr
    
    def clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Applique le gradient clipping si nécessaire.
        
        Args:
            grad: Gradient à clipper
            
        Returns:
            Gradient clippé
        """
        if self.clipnorm is not None:
            # Clipping par norme
            norm = np.linalg.norm(grad)
            if norm > self.clipnorm:
                grad = grad * (self.clipnorm / norm)
        
        if self.clipvalue is not None:
            # Clipping par valeur absolue
            grad = np.clip(grad, -self.clipvalue, self.clipvalue)
        
        return grad
    
    def _get_layer_id(self, layer) -> int:
        """
        Génère un ID unique pour une couche.
        
        Args:
            layer: Couche du réseau
            
        Returns:
            ID unique de la couche
        """
        return id(layer)
    
    def _init_velocity(self, layer_id: int, param_name: str, param_shape: tuple) -> None:
        """
        Initialise la vitesse (velocity) pour le momentum si nécessaire.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            param_shape: Shape du paramètre
        """
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
        
        if param_name not in self.velocities[layer_id]:
            self.velocities[layer_id][param_name] = np.zeros(param_shape, dtype=np.float32)
    
    def _get_velocity(self, layer_id: int, param_name: str) -> np.ndarray:
        """
        Récupère la vitesse (velocity) pour un paramètre.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Vitesse du paramètre
        """
        return self.velocities.get(layer_id, {}).get(param_name, None)
    
    def _update_velocity(self, layer_id: int, param_name: str, velocity: np.ndarray) -> None:
        """
        Met à jour la vitesse (velocity) pour un paramètre.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            velocity: Nouvelle vitesse
        """
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
        self.velocities[layer_id][param_name] = velocity
    
    @abstractmethod
    def update_parameter(self, 
                        param: np.ndarray, 
                        grad: np.ndarray,
                        layer_id: int,
                        param_name: str) -> np.ndarray:
        """
        Met à jour un paramètre selon la stratégie de l'optimiseur.
        
        Cette méthode doit être implémentée par chaque optimiseur spécifique.
        
        Args:
            param: Paramètre actuel
            grad: Gradient du paramètre
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Nouveau paramètre mis à jour
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement update_parameter()"
        )
    
    def step(self, layers: List) -> None:
        """
        Effectue une étape de mise à jour pour toutes les couches.
        
        Cette méthode parcourt toutes les couches, récupère leurs paramètres
        et gradients, puis les met à jour en utilisant update_parameter().
        
        Args:
            layers: Liste des couches à mettre à jour
        """
        current_lr = self.get_lr()
        
        for layer in layers:
            # Ignorer les couches non entraînables
            if not layer.trainable:
                continue
            
            # Ignorer les couches sans paramètres
            if not layer.parameters:
                continue
            
            layer_id = self._get_layer_id(layer)
            
            # Mettre à jour chaque paramètre de la couche
            for param_name, param in layer.parameters.items():
                # Récupérer le gradient correspondant
                if param_name not in layer.gradients:
                    continue
                
                grad = layer.gradients[param_name].copy()
                
                # Vérifier que le gradient a la même shape que le paramètre
                if grad.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch between parameter '{param_name}' and its gradient: "
                        f"param shape {param.shape}, grad shape {grad.shape}"
                    )
                
                # Appliquer le gradient clipping si nécessaire
                grad = self.clip_gradient(grad)
                
                # Mettre à jour le paramètre selon la stratégie de l'optimiseur
                updated_param = self.update_parameter(
                    param=param,
                    grad=grad,
                    layer_id=layer_id,
                    param_name=param_name
                )
                
                # Mettre à jour le paramètre dans la couche
                layer.parameters[param_name] = updated_param
        
        # Incrémenter le compteur d'itérations pour le decay
        self.iterations += 1
    
    def zero_grad(self) -> None:
        """
        Réinitialise les vitesses (velocities) pour le momentum.
        Utile si vous voulez réinitialiser l'état de l'optimiseur.
        """
        self.velocities = {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de l'optimiseur.
        
        Returns:
            Dictionnaire contenant la configuration
        """
        return {
            'name': self.name,
            'lr': self.initial_lr,
            'momentum': self.momentum,
            'decay': self.decay,
            'clipnorm': self.clipnorm,
            'clipvalue': self.clipvalue,
            'iterations': self.iterations,
            'class_name': self.__class__.__name__
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Configure l'optimiseur à partir d'un dictionnaire.
        
        Args:
            config: Dictionnaire de configuration
        """
        if 'lr' in config:
            self.lr = float(config['lr'])
            self.initial_lr = float(config['lr'])
        if 'momentum' in config:
            self.momentum = float(config['momentum'])
        if 'decay' in config:
            self.decay = float(config['decay'])
        if 'clipnorm' in config:
            self.clipnorm = config['clipnorm']
        if 'clipvalue' in config:
            self.clipvalue = config['clipvalue']
        if 'iterations' in config:
            self.iterations = int(config['iterations'])
    
    def __repr__(self) -> str:
        """
        Représentation string de l'optimiseur.
        """
        config = self.get_config()
        return (
            f"{self.__class__.__name__}("
            f"lr={config['lr']}, "
            f"momentum={config['momentum']}, "
            f"decay={config['decay']}, "
            f"iterations={config['iterations']})"
        )

