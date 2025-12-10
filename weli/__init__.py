"""
Weli - Framework de Deep Learning en Python

Un framework simple et intuitif pour créer et entraîner des réseaux de neurones.
"""

__version__ = '0.1.0'
__author__ = 'Weli Team'

# Imports principaux
from . import models
from . import layers
from . import losses
from . import optimizers
from . import utils
from . import backend

# Exports principaux pour faciliter l'utilisation
from .models import (
    Sequential, Functional, Model,
    save_model, load_model
)
from .layers import (
    Layer, Dense,
    ReLU, Sigmoid, Tanh, Softmax,
    Conv2D, MaxPool2D, Flatten,
    Dropout, BatchNorm1D, BatchNorm2D,
    SimpleRNN, LSTM, GRU
)
from .losses import (
    Loss, MSE, MAE,
    CrossEntropy, BinaryCrossEntropy
)
from .optimizers import SGD

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
    
    # Modèles
    'Sequential',
    'Functional',
    'Model',
    'save_model',
    'load_model',
    
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
    'MSE', 'MAE',
    'CrossEntropy', 'BinaryCrossEntropy',
    
    # Optimizers
    'SGD',
]

