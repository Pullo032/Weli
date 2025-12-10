"""
Implémentation de Adam (Adaptive Moment Estimation) pour Weli.
"""
import numpy as np
from typing import Optional
from .base import Optimizer


class Adam(Optimizer):
    """
    Optimiseur Adam (Adaptive Moment Estimation).
    
    Adam combine les avantages de RMSprop et du momentum. Il calcule des
    moyennes mobiles adaptatives des gradients et de leurs carrés.
    
    Formule Adam:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)
        param = param - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
    
    Args:
        lr: Learning rate (taux d'apprentissage). Default: 0.001
        beta1: Facteur de décroissance pour le premier moment (moyenne des gradients). Default: 0.9
        beta2: Facteur de décroissance pour le deuxième moment (moyenne des carrés). Default: 0.999
        epsilon: Petite valeur pour éviter la division par zéro. Default: 1e-8
        decay: Taux de décroissance du learning rate. Default: 0.0
        clipnorm: Valeur maximale pour le gradient clipping. Default: None
        clipvalue: Valeur maximale absolue pour le gradient clipping. Default: None
        name: Nom de l'optimiseur. Default: None
    
    Exemple:
        >>> from weli.optimizers import Adam
        >>> optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        >>> model.compile(optimizer=optimizer, ...)
    
    Référence:
        Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
        arXiv preprint arXiv:1412.6980.
    """
    
    def __init__(self,
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 decay: float = 0.0,
                 clipnorm: Optional[float] = None,
                 clipvalue: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur Adam.
        """
        super().__init__(
            lr=lr,
            momentum=0.0,  # Adam n'utilise pas le momentum classique
            decay=decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            name=name or 'Adam'
        )
        
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        
        # Stockage des moments (premier et deuxième)
        # Structure: {layer_id: {param_name: {'m': first_moment, 'v': second_moment}}}
        self.moments = {}
    
    def _get_layer_id(self, layer) -> int:
        """
        Génère un ID unique pour une couche.
        
        Args:
            layer: Couche du réseau
            
        Returns:
            ID unique de la couche
        """
        return id(layer)
    
    def _init_moments(self, layer_id: int, param_name: str, param_shape: tuple) -> None:
        """
        Initialise les moments (premier et deuxième) pour un paramètre.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            param_shape: Shape du paramètre
        """
        if layer_id not in self.moments:
            self.moments[layer_id] = {}
        
        if param_name not in self.moments[layer_id]:
            self.moments[layer_id][param_name] = {
                'm': np.zeros(param_shape, dtype=np.float32),  # Premier moment
                'v': np.zeros(param_shape, dtype=np.float32)   # Deuxième moment
            }
    
    def _get_moments(self, layer_id: int, param_name: str) -> dict:
        """
        Récupère les moments (premier et deuxième) pour un paramètre.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Dictionnaire avec 'm' (premier moment) et 'v' (deuxième moment)
        """
        return self.moments.get(layer_id, {}).get(param_name, None)
    
    def _update_moments(self, layer_id: int, param_name: str, m: np.ndarray, v: np.ndarray) -> None:
        """
        Met à jour les moments (premier et deuxième) pour un paramètre.
        
        Args:
            layer_id: ID de la couche
            param_name: Nom du paramètre
            m: Nouveau premier moment
            v: Nouveau deuxième moment
        """
        if layer_id not in self.moments:
            self.moments[layer_id] = {}
        self.moments[layer_id][param_name] = {
            'm': m,
            'v': v
        }
    
    def update_parameter(self,
                        param: np.ndarray,
                        grad: np.ndarray,
                        layer_id: int,
                        param_name: str) -> np.ndarray:
        """
        Met à jour un paramètre selon l'algorithme Adam.
        
        Args:
            param: Paramètre actuel
            grad: Gradient du paramètre (déjà clippé si nécessaire)
            layer_id: ID de la couche
            param_name: Nom du paramètre
            
        Returns:
            Nouveau paramètre mis à jour
        """
        current_lr = self.get_lr()
        t = self.iterations + 1  # t commence à 1
        
        # Initialiser les moments si nécessaire
        self._init_moments(layer_id, param_name, param.shape)
        
        # Récupérer les moments actuels
        moments = self._get_moments(layer_id, param_name)
        
        if moments is None:
            # Si pas de moments, initialiser à zéro
            m = np.zeros_like(param)
            v = np.zeros_like(param)
        else:
            m = moments['m']
            v = moments['v']
        
        # Mettre à jour les moments biaisés
        # Premier moment (moyenne des gradients)
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        
        # Deuxième moment (moyenne des carrés des gradients)
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
        
        # Correction du biais pour les moments
        # m_hat = m / (1 - beta1^t)
        m_hat = m / (1.0 - (self.beta1 ** t))
        
        # v_hat = v / (1 - beta2^t)
        v_hat = v / (1.0 - (self.beta2 ** t))
        
        # Sauvegarder les nouveaux moments
        self._update_moments(layer_id, param_name, m, v)
        
        # Mettre à jour le paramètre
        # param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        updated_param = param - current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_param
    
    def zero_grad(self) -> None:
        """
        Réinitialise les moments (premier et deuxième).
        Utile si vous voulez réinitialiser l'état de l'optimiseur.
        """
        super().zero_grad()  # Réinitialise aussi les velocities de la classe de base
        self.moments = {}
    
    def get_config(self) -> dict:
        """
        Retourne la configuration de l'optimiseur Adam.
        
        Returns:
            Dictionnaire contenant la configuration
        """
        config = super().get_config()
        config['beta1'] = self.beta1
        config['beta2'] = self.beta2
        config['epsilon'] = self.epsilon
        return config
    
    def set_config(self, config: dict) -> None:
        """
        Configure l'optimiseur Adam à partir d'un dictionnaire.
        
        Args:
            config: Dictionnaire de configuration
        """
        super().set_config(config)
        if 'beta1' in config:
            self.beta1 = float(config['beta1'])
        if 'beta2' in config:
            self.beta2 = float(config['beta2'])
        if 'epsilon' in config:
            self.epsilon = float(config['epsilon'])
    
    def __repr__(self) -> str:
        """
        Représentation string de l'optimiseur Adam.
        """
        config = self.get_config()
        return (
            f"Adam("
            f"lr={config['lr']}, "
            f"beta1={config['beta1']}, "
            f"beta2={config['beta2']}, "
            f"epsilon={config['epsilon']}, "
            f"decay={config['decay']}, "
            f"iterations={config['iterations']})"
        )

