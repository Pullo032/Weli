"""
Modèles pour le framework Weli.
"""

from .model import Model
from .sequential import Sequential
from .functional import (
    Functional, Input, 
    add, concatenate, multiply,
    create_residual_block, create_inception_module
)

# Import conditionnel des wrappers
try:
    from .functional import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNorm2D
    _FUNCTIONAL_WRAPPERS_AVAILABLE = True
except ImportError:
    _FUNCTIONAL_WRAPPERS_AVAILABLE = False

__all__ = [
    'Model',
    'Sequential',
    'Functional',
    'Input',
    'add',
    'concatenate',
    'multiply',
    'create_residual_block',
    'create_inception_module'
]

if _FUNCTIONAL_WRAPPERS_AVAILABLE:
    __all__.extend(['Dense', 'Conv2D', 'MaxPool2D', 'Flatten', 'Dropout', 'BatchNorm2D'])