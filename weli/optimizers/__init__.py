"""
Optimiseurs pour Weli.

Ce module contient toutes les implémentations d'optimiseurs pour l'entraînement
des modèles de deep learning.
"""

# Import de la classe de base
from .base import Optimizer

# Import de SGD
try:
    from .sgd import SGD
except ImportError:
    # Si SGD n'est pas encore implémenté, créer une classe placeholder
    class SGD:
        """Placeholder pour SGD optimizer"""
        def __init__(self, lr=0.01):
            self.lr = lr

# Import de RMSprop
try:
    from .rmsprop import RMSprop
except ImportError:
    # Si RMSprop n'est pas encore implémenté, créer une classe placeholder
    class RMSprop:
        """Placeholder pour RMSprop optimizer"""
        def __init__(self, lr=0.001):
            self.lr = lr

# Liste des exports
__all__ = [
    'Optimizer',  # Classe de base pour tous les optimiseurs
    'SGD',        # Stochastic Gradient Descent
    'RMSprop',    # Root Mean Square Propagation
]
