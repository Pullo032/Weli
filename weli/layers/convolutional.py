import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base import Layer

class Conv2D(Layer):
    """
    Couche de convolution 2D.
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 padding: str = 'valid',
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.activation_name = activation
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        self.activation = None
        
        if activation:
            from .activation import get_activation
            self.activation = get_activation(activation)
    
    def _pad_input(self, X: np.ndarray) -> np.ndarray:
        """Applique le padding à l'input."""
        if self.padding == 'valid':
            return X
        
        # Padding 'same'
        pad_h = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] // 2)
        pad_w = ((self.kernel_size[1] - 1) // 2, self.kernel_size[1] // 2)
        
        return np.pad(X, 
                     pad_width=((0, 0), (pad_h[0], pad_h[1]), 
                               (pad_w[0], pad_w[1]), (0, 0)),
                     mode='constant')
    
    def _im2col(self, X: np.ndarray) -> np.ndarray:
        """Convertit les images en colonnes pour la convolution."""
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Initialisation de la matrice de colonnes
        cols = np.zeros((batch_size, out_h, out_w, k_h, k_w, in_c))
        
        for i in range(k_h):
            for j in range(k_w):
                cols[:, :, :, i, j, :] = X[:, 
                                           i:i + out_h * self.strides[0]:self.strides[0],
                                           j:j + out_w * self.strides[1]:self.strides[1],
                                           :]
        
        return cols.reshape(batch_size * out_h * out_w, k_h * k_w * in_c)
    
    def _col2im(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en images (pour la backward pass)."""
        batch_size, in_h, in_w, in_c = X_shape
        k_h, k_w = self.kernel_size
        
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_h, out_w, k_h, k_w, in_c)
        
        for i in range(k_h):
            for j in range(k_w):
                dX[:,
                   i:i + out_h * self.strides[0]:self.strides[0],
                   j:j + out_w * self.strides[1]:self.strides[1],
                   :] += dcols_reshaped[:, :, :, i, j, :]
        
        return dX
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_h, in_w, in_c = input_shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        if self.padding == 'same':
            out_h = int(np.ceil(in_h / self.strides[0]))
            out_w = int(np.ceil(in_w / self.strides[1]))
        else:  # 'valid'
            out_h = (in_h - k_h) // self.strides[0] + 1
            out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Initialisation des kernels
        fan_in = k_h * k_w * in_c
        fan_out = k_h * k_w * self.filters
        
        if self.kernel_initializer_name == 'he':
            std = np.sqrt(2.0 / fan_in)
            kernel_shape = (k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * std
        elif self.kernel_initializer_name == 'xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            kernel_shape = (k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.uniform(-limit, limit, kernel_shape)
        else:
            kernel_shape = (k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * 0.05
        
        self.gradients['W'] = np.zeros_like(self.parameters['W'])
        
        # Initialisation des biais
        if self.bias_initializer_name == 'zeros':
            self.parameters['b'] = np.zeros((1, 1, 1, self.filters))
        else:
            self.parameters['b'] = np.random.randn(1, 1, 1, self.filters) * 0.05
        
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, out_h, out_w, self.filters)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Applique le padding
        if self.padding == 'same':
            x_padded = self._pad_input(x)
        else:
            x_padded = x
        
        batch_size, in_h, in_w, in_c = x_padded.shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Convolution utilisant im2col
        cols = self._im2col(x_padded)
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        
        # Calcul de la sortie
        self.z = cols @ W_reshaped
        self.z = self.z.reshape(batch_size, out_h, out_w, self.filters)
        self.z += self.parameters['b']
        
        # Stocke pour la backward pass
        self.cols = cols
        self.x_shape = x_padded.shape
        
        # Activation
        if self.activation:
            self.output = self.activation.forward(self.z)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size = self.input.shape[0]
        
        if self.activation:
            dout = self.activation.backward(dout)
        
        # Gradient par rapport aux biais
        self.gradients['b'] = np.sum(dout, axis=(0, 1, 2), keepdims=True) / batch_size
        
        # Reshape dout
        dout_reshaped = dout.reshape(batch_size * self.output_shape[1] * self.output_shape[2], self.filters)
        
        # Gradient par rapport aux poids
        self.gradients['W'] = (self.cols.T @ dout_reshaped).reshape(self.parameters['W'].shape) / batch_size
        
        # Gradient par rapport à l'input
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        dcols = dout_reshaped @ W_reshaped.T
        
        # Convertir les colonnes en images
        dx_padded = self._col2im(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
            # Calculer le padding appliqué
            pad_h = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] // 2)
            pad_w = ((self.kernel_size[1] - 1) // 2, self.kernel_size[1] // 2)
            
            if pad_h[0] > 0 or pad_h[1] > 0 or pad_w[0] > 0 or pad_w[1] > 0:
                dx = dx_padded[:, 
                             pad_h[0]:-pad_h[1] if pad_h[1] > 0 else None,
                             pad_w[0]:-pad_w[1] if pad_w[1] > 0 else None,
                             :]
            else:
                dx = dx_padded
        else:
            dx = dx_padded
        
        return dx

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation_name,
            'kernel_initializer': self.kernel_initializer_name,
            'bias_initializer': self.bias_initializer_name,
        })
        return config

class MaxPool2D(Layer):
    """
    Couche de pooling max 2D.
    """
    
    def __init__(self,
                 pool_size: int = 2,
                 strides: Optional[int] = None,
                 padding: str = 'valid',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides else pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        self.padding = padding.lower()
        self.trainable = False
        self.mask = None
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_h, in_w, in_c = input_shape
        p_h, p_w = self.pool_size
        
        if self.padding == 'same':
            out_h = int(np.ceil(in_h / self.strides[0]))
            out_w = int(np.ceil(in_w / self.strides[1]))
        else:
            out_h = (in_h - p_h) // self.strides[0] + 1
            out_w = (in_w - p_w) // self.strides[1] + 1
        
        self.output_shape = (batch_size, out_h, out_w, in_c)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, in_h, in_w, in_c = x.shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.strides
        
        # Calcul des dimensions de sortie
        out_h = (in_h - p_h) // s_h + 1
        out_w = (in_w - p_w) // s_w + 1
        
        # Initialisation de la sortie
        output = np.zeros((batch_size, out_h, out_w, in_c))
        
        # Stocke les indices des max pour la backward pass
        self.max_indices = np.zeros((batch_size, out_h, out_w, in_c, 2), dtype=int)
        
        # Pooling max
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                h_end = h_start + p_h
                w_start = j * s_w
                w_end = w_start + p_w
                
                window = x[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(window, axis=(1, 2))
                
                # Trouver les indices des max
                window_reshaped = window.reshape(batch_size, p_h * p_w, in_c)
                max_indices_flat = np.argmax(window_reshaped, axis=1)
                
                # Convertir les indices plats en indices 2D
                self.max_indices[:, i, j, :, 0] = h_start + (max_indices_flat // p_w)
                self.max_indices[:, i, j, :, 1] = w_start + (max_indices_flat % p_w)
        
        self.output = output
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, in_h, in_w, in_c = self.input.shape
        dx = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    for c in range(in_c):
                        h_idx, w_idx = self.max_indices[b, i, j, c]
                        dx[b, h_idx, w_idx, c] += dout[b, i, j, c]
        
        return dx

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

class Flatten(Layer):
    """
    Couche pour aplatir l'input (ex: (batch, 28, 28, 1) -> (batch, 784)).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
    
    def initialize(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        batch_size = input_shape[0]
        flattened_size = np.prod(input_shape[1:])
        self.output_shape = (batch_size, flattened_size)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size = x.shape[0]
        self.output = x.reshape(batch_size, -1)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.input.shape)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()