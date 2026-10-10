import numpy as np
from typing import Optional, Dict, Any
from .base import Layer

class Activation(Layer):
    """Classe de base pour les fonctions d'activation."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
    
    def initialize(self, input_shape: tuple) -> tuple:
        return input_shape

class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.input > 0)

class Sigmoid(Activation):
    """Sigmoïde: σ(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        sig = self.output
        return dout * sig * (1.0 - sig)

class Tanh(Activation):
    """Tangente hyperbolique: tanh(x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (1.0 - self.output ** 2)

class Softmax(Activation):
    """
    Softmax pour la classification multi-classes.
    Pour la stabilité numérique, on utilise: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        # Stabilité numérique
        x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.output * (dout - np.sum(dout * self.output, axis=-1, keepdims=True))

class LeakyReLU(Activation):
    """Leaky ReLU: f(x) = x si x > 0, sinon αx"""
    
    def __init__(self, alpha: float = 0.01, name: Optional[str] = None):
        super().__init__(name)
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * np.where(self.input > 0, 1.0, self.alpha)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['alpha'] = self.alpha
        return config

class ELU(Activation):
    """Exponential Linear Unit: f(x) = x si x > 0, sinon α(exp(x) - 1)"""
    
    def __init__(self, alpha: float = 1.0, name: Optional[str] = None):
        super().__init__(name)
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        grad = np.where(self.input > 0, 1.0, self.alpha * np.exp(self.input))
        return dout * grad

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['alpha'] = self.alpha
        return config

def get_activation(name: str, **kwargs):
    """Factory pour obtenir une fonction d'activation."""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'leaky_relu': LeakyReLU,
        'elu': ELU
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Activation '{name}' not recognized")
    
    return activations[name_lower](**kwargs)