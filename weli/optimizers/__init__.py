"""
Optimiseurs pour Weli.

Ce module contient toutes les implémentations d'optimiseurs pour l'entraînement
des modèles de deep learning.
"""

# Import de la classe de base
from .base import Optimizer

from .sgd import SGD
from .rmsprop import RMSprop
from .adam import Adam

# Liste des exports
__all__ = [
    'Optimizer',  # Classe de base pour tous les optimiseurs
    'SGD',        # Stochastic Gradient Descent
    'RMSprop',    # Root Mean Square Propagation
    'Adam',       # Adaptive Moment Estimation
]
