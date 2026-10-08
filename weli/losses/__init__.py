"""
Fonctions de perte (loss functions) pour Weli.
"""

from .base import Loss, CombinedLoss, LossFunctionRegistry
from .regression import MSE, MAE, HuberLoss, MSLE
from .classification import (
    CrossEntropy,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    HingeLoss,
    SquaredHingeLoss,
)
from .advanced import (
    KLDivergence,
    PoissonLoss,
    CosineSimilarityLoss,
    LogCoshLoss,
    DiceLoss,
)
from .focal_loss import FocalLoss
from .triplet_loss import TripletLoss
from .contrastive_loss import ContrastiveLoss
from .wasserstein_loss import WassersteinLoss

__all__ = [
    # Base
    'Loss',
    'CombinedLoss',
    'LossFunctionRegistry',

    # Régression
    'MSE', 'MAE', 'HuberLoss', 'MSLE',

    # Classification
    'CrossEntropy',
    'BinaryCrossEntropy',
    'CategoricalCrossEntropy',
    'SparseCategoricalCrossEntropy',
    'HingeLoss',
    'SquaredHingeLoss',

    # Avancées
    'KLDivergence',
    'PoissonLoss',
    'CosineSimilarityLoss',
    'LogCoshLoss',
    'DiceLoss',
    'FocalLoss',
    'TripletLoss',
    'ContrastiveLoss',
    'WassersteinLoss',
]