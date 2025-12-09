import numpy as np
from typing import Optional, Dict, Any
from .base import Layer

class Dense(Layer):
    """
    Couche Dense (Fully Connected): y = xW + b
    """
    
    def __init__(self, 
                 units: int,
                 input_dim: Optional[int] = None,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.units = units
        self.input_dim = input_dim
        self.activation_name = activation
        self.use_bias = use_bias
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        self.activation = None
        
        if activation:
            from .activation import get_activation
            self.activation = get_activation(activation)
    
    def _initialize_weights(self, fan_in: int, fan_out: int):
        """Initialise les poids selon la stratégie spécifiée."""
        if self.kernel_initializer_name == 'he':
            # He initialization (bon pour ReLU)
            std = np.sqrt(2.0 / fan_in)
            return np.random.randn(fan_in, fan_out) * std
        elif self.kernel_initializer_name == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif self.kernel_initializer_name == 'zeros':
            return np.zeros((fan_in, fan_out))
        elif self.kernel_initializer_name == 'random_normal':
            return np.random.randn(fan_in, fan_out) * 0.05
        else:
            raise ValueError(f"Unknown initializer: {self.kernel_initializer_name}")
    
    def _initialize_bias(self, shape: tuple):
        """Initialise les biais."""
        if self.bias_initializer_name == 'zeros':
            return np.zeros(shape)
        elif self.bias_initializer_name == 'random_normal':
            return np.random.randn(*shape) * 0.05
        else:
            raise ValueError(f"Unknown bias initializer: {self.bias_initializer_name}")
    
    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {len(input_shape)}D")
        
        batch_size, input_dim = input_shape
        
        if self.input_dim is None:
            self.input_dim = input_dim
        elif self.input_dim != input_dim:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim}, got {input_dim}")
        
        # Initialisation des poids
        self.parameters['W'] = self._initialize_weights(self.input_dim, self.units)
        self.gradients['W'] = np.zeros_like(self.parameters['W'])
        
        # Initialisation des biais
        if self.use_bias:
            self.parameters['b'] = self._initialize_bias((1, self.units))
            self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, self.units)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z = x @ self.parameters['W']
        
        if self.use_bias:
            self.z += self.parameters['b']
        
        if self.activation:
            self.output = self.activation.forward(self.z)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size = self.input.shape[0]
        
        if self.activation:
            dout = self.activation.backward(dout)
        
        # Gradient par rapport aux poids
        self.gradients['W'] = (self.input.T @ dout) / batch_size
        
        # Gradient par rapport aux biais
        if self.use_bias:
            self.gradients['b'] = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        # Gradient par rapport à l'input
        dx = dout @ self.parameters['W'].T
        
        return dx
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'activation': self.activation_name,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer_name,
            'bias_initializer': self.bias_initializer_name
        })
        return config