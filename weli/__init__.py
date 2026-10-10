"""
Weli - Framework de Deep Learning en Python

Un framework simple et intuitif pour créer et entraîner des réseaux de neurones.
"""

__version__ = '1.0.0'
__author__ = 'Weli Team'

# Imports principaux
from . import models
from . import layers
from . import losses
from . import optimizers
from . import utils
from . import backend
from . import core

# Exports principaux pour faciliter l'utilisation
from .models import (
    Sequential, Functional, Model,
    save_model, load_model,
    resnet18, text_classifier_gru, text_classifier_lstm,
    generator_mlp, discriminator_mlp, gan_mlp,
    text_to_image_mlp, text_to_video_mlp,
)
from .layers import (
    Layer, Dense,
    ReLU, Sigmoid, Tanh, Softmax,
    Conv2D, MaxPool2D, Flatten,
    Dropout, BatchNorm1D, BatchNorm2D,
    SimpleRNN, LSTM, GRU
)
from .losses import (
    Loss,
    MSE, MAE, HuberLoss, MSLE,
    CrossEntropy, BinaryCrossEntropy,
    CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
    HingeLoss, SquaredHingeLoss,
    KLDivergence, PoissonLoss, CosineSimilarityLoss, LogCoshLoss,
    DiceLoss, FocalLoss, TripletLoss, ContrastiveLoss, WassersteinLoss,
)
from .optimizers import SGD, Adam, RMSprop

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Modules
    'models',
    'layers',
    'losses',
    'optimizers',
    'utils',
    'backend',
    'core',
    
    # Modèles
    'Sequential',
    'Functional',
    'Model',
    'save_model',
    'load_model',
    'resnet18',
    'text_classifier_gru',
    'text_classifier_lstm',
    'generator_mlp',
    'discriminator_mlp',
    'gan_mlp',
    'text_to_image_mlp',
    'text_to_video_mlp',
    
    # Couches
    'Layer',
    'Dense',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax',
    'Conv2D', 'MaxPool2D', 'Flatten',
    'Dropout',
    'BatchNorm1D', 'BatchNorm2D',
    'SimpleRNN', 'LSTM', 'GRU',
    
    # Losses
    'Loss',
    'MSE', 'MAE', 'HuberLoss', 'MSLE',
    'CrossEntropy', 'BinaryCrossEntropy',
    'CategoricalCrossEntropy', 'SparseCategoricalCrossEntropy',
    'HingeLoss', 'SquaredHingeLoss',
    'KLDivergence', 'PoissonLoss', 'CosineSimilarityLoss', 'LogCoshLoss',
    'DiceLoss', 'FocalLoss', 'TripletLoss', 'ContrastiveLoss', 'WassersteinLoss',

    # Optimizers
    'SGD',
    'Adam',
    'RMSprop',
]
