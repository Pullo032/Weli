"""
Fonctions de perte (loss functions) pour Weli.
"""

from .base import Loss
from .regression import MSE, MAE, HuberLoss, MSLE
from .classification import (
    CrossEntropy, BinaryCrossEntropy, 
    CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
    HingeLoss, SquaredHingeLoss
)
from .advanced import (
    KLDivergence, PoissonLoss, 
    CosineSimilarityLoss, LogCoshLoss
)

__all__ = [
    # Classe de base
    'Loss',
    
    # Régression
    'MSE', 'MAE', 'HuberLoss', 'MSLE',
    
    # Classification
    'CrossEntropy', 'BinaryCrossEntropy', 'CategoricalCrossEntropy',
    'SparseCategoricalCrossEntropy', 'HingeLoss', 'SquaredHingeLoss',
    
    # Avancées
    'KLDivergence', 'PoissonLoss', 'CosineSimilarityLoss', 'LogCoshLoss'
]