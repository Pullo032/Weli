import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class Layer:
    """
    Classe de base pour toutes les couches du réseau.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.trainable = True
        self.parameters = {}
        self.gradients = {}
        self.input = None
        self.output = None
        self.training = True
        self.input_shape = None
        self.output_shape = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement backward()")
    
    def initialize(self, input_shape: Tuple) -> Tuple:
        """
        Initialise les paramètres et retourne la shape de sortie.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape  # Par défaut, ne change pas la shape
        return self.output_shape
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        for key, value in params.items():
            if key in self.parameters:
                if self.parameters[key].shape != value.shape:
                    raise ValueError(
                        f"Shape mismatch for parameter '{key}': "
                        f"expected {self.parameters[key].shape}, got {value.shape}"
                    )
                self.parameters[key] = value
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        return self.gradients.copy()
    
    def zero_grad(self):
        for key in self.gradients:
            self.gradients[key].fill(0)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'trainable': self.trainable,
            'class_name': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        config = self.get_config()
        params = sum(p.size for p in self.parameters.values()) if self.parameters else 0
        return f"{self.__class__.__name__}(name={self.name}, params={params})"