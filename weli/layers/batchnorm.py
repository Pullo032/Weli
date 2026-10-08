import numpy as np
from typing import Optional, Dict, Any
from .base import Layer

class BatchNorm1D(Layer):
    """
    Normalisation par batch pour les couches Dense.
    """
    
    def __init__(self,
                 momentum: float = 0.99,
                 epsilon: float = 1e-5,
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Paramètres apprenables
        self.gamma = None  # Scale
        self.beta = None   # Shift
        
        # Statistiques courantes
        self.running_mean = None
        self.running_var = None
        
        # Cache pour la backward pass
        self.x_norm = None
        self.std = None
    
    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) != 2:
            raise ValueError(f"BatchNorm1D expects 2D input, got {len(input_shape)}D")
        
        _, features = input_shape
        
        # Initialisation des paramètres
        self.parameters['gamma'] = np.ones((1, features))
        self.parameters['beta'] = np.zeros((1, features))
        
        self.gradients['gamma'] = np.zeros_like(self.parameters['gamma'])
        self.gradients['beta'] = np.zeros_like(self.parameters['beta'])
        
        # Statistiques courantes
        self.running_mean = np.zeros((1, features))
        self.running_var = np.ones((1, features))
        self._restore_running_statistics()
        
        return input_shape

    def _restore_running_statistics(self):
        state = getattr(self, '_serialized_state', {})
        for name in ('running_mean', 'running_var'):
            value = state.get(name)
            if value is not None:
                setattr(self, name, value)
        if state:
            del self._serialized_state
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        if self.training:
            # Calcul des statistiques du batch
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Mise à jour des statistiques courantes
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalisation
            self.std = np.sqrt(batch_var + self.epsilon)
            self.x_norm = (x - batch_mean) / self.std
        else:
            # Mode inference: utilise les statistiques courantes
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        self.output = self.parameters['gamma'] * self.x_norm + self.parameters['beta']
        
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size = self.input.shape[0]
        
        # Gradient par rapport à gamma et beta
        self.gradients['gamma'] = np.sum(dout * self.x_norm, axis=0, keepdims=True) / batch_size
        self.gradients['beta'] = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        # Gradient par rapport à x_norm
        dx_norm = dout * self.parameters['gamma']
        
        if self.training:
            # Gradient par rapport à x (formule de la backprop pour batch norm)
            dvar = np.sum(dx_norm * (self.input - np.mean(self.input, axis=0, keepdims=True)) * 
                         -0.5 * (self.std ** -3), axis=0, keepdims=True)
            
            dmean = np.sum(dx_norm * -1 / self.std, axis=0, keepdims=True) + \
                    dvar * np.mean(-2 * (self.input - np.mean(self.input, axis=0, keepdims=True)), axis=0, keepdims=True)
            
            dx = dx_norm / self.std + dvar * 2 * (self.input - np.mean(self.input, axis=0, keepdims=True)) / batch_size + \
                 dmean / batch_size
        else:
            dx = dx_norm / np.sqrt(self.running_var + self.epsilon)
        
        return dx

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'momentum': self.momentum, 'epsilon': self.epsilon})
        config['running_mean'] = self.running_mean
        config['running_var'] = self.running_var
        return config

class BatchNorm2D(BatchNorm1D):
    """
    Normalisation par batch pour les couches Conv2D.
    """
    
    def __init__(self,
                 momentum: float = 0.99,
                 epsilon: float = 1e-5,
                 name: Optional[str] = None):
        super().__init__(momentum, epsilon, name)
    
    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) != 4:
            raise ValueError(f"BatchNorm2D expects 4D input, got {len(input_shape)}D")
        
        batch_size, height, width, channels = input_shape
        
        # Reshape pour traiter les canaux séparément
        self.original_shape = input_shape
        self.features = channels
        
        # Initialisation des paramètres
        self.parameters['gamma'] = np.ones((1, 1, 1, channels))
        self.parameters['beta'] = np.zeros((1, 1, 1, channels))
        
        self.gradients['gamma'] = np.zeros_like(self.parameters['gamma'])
        self.gradients['beta'] = np.zeros_like(self.parameters['beta'])
        
        # Statistiques courantes
        self.running_mean = np.zeros((1, 1, 1, channels))
        self.running_var = np.ones((1, 1, 1, channels))
        self._restore_running_statistics()
        
        return input_shape

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, height, width, channels = x.shape
        
        # Reshape pour le calcul
        x_reshaped = x.transpose(0, 3, 1, 2).reshape(batch_size, channels, -1).transpose(0, 2, 1)
        x_reshaped = x_reshaped.reshape(-1, channels)
        
        if self.training:
            batch_mean = np.mean(x_reshaped, axis=0, keepdims=True)
            batch_var = np.var(x_reshaped, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            self.std = np.sqrt(batch_var + self.epsilon)
            self.x_norm = (x_reshaped - batch_mean) / self.std
        else:
            self.x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Reshape back
        x_norm_reshaped = self.x_norm
        x_norm_reshaped = x_norm_reshaped.reshape(batch_size, height, width, channels)
        self.output = self.parameters['gamma'] * x_norm_reshaped + self.parameters['beta']
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, height, width, channels = self.input.shape
        sample_count = batch_size * height * width
        dout_flat = dout.reshape(sample_count, channels)
        gamma = self.parameters['gamma'].reshape(1, channels)

        self.gradients['gamma'] = np.sum(
            dout * self.x_norm.reshape(batch_size, height, width, channels),
            axis=(0, 1, 2), keepdims=True
        )
        self.gradients['beta'] = np.sum(dout, axis=(0, 1, 2), keepdims=True)

        dx_norm = dout_flat * gamma
        if self.training:
            dx_flat = (dx_norm - np.mean(dx_norm, axis=0, keepdims=True)
                       - self.x_norm * np.mean(
                           dx_norm * self.x_norm, axis=0, keepdims=True
                       )) / self.std
        else:
            dx_flat = dx_norm / np.sqrt(self.running_var.reshape(1, channels) + self.epsilon)
        return dx_flat.reshape(batch_size, height, width, channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'momentum': self.momentum, 'epsilon': self.epsilon})
        return config