import numpy as np
from typing import Optional, Dict, Any
from .base import Layer

class Dropout(Layer):
    """
    Dropout pour la régularisation.
    Pendant l'entraînement: désactive des neurones aléatoirement
    Pendant l'inférence: échelle les valeurs par (1 - rate)
    """
    
    def __init__(self, rate: float = 0.5, name: Optional[str] = None):
        super().__init__(name)
        self.rate = min(1.0, max(0.0, rate))  # Clip entre 0 et 1
        self.trainable = False
        self.mask = None
    
    def initialize(self, input_shape: tuple) -> tuple:
        return input_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        if self.training and self.rate > 0.0:
            # Génère le masque
            self.mask = (np.random.random(x.shape) >= self.rate).astype(float)
            # Échelle pour maintenir la somme attendue
            self.mask /= (1.0 - self.rate)
            self.output = x * self.mask
        else:
            self.output = x
        
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.training and self.rate > 0.0:
            return dout * self.mask
        return dout
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'rate': self.rate
        })
        return config