"""
Convolutions avancées pour Weli.

Implémentations:
- Conv1D: Convolution 1D pour séquences
- Conv3D: Convolution 3D pour vidéos/volumes
- DepthwiseConv2D: Convolution depthwise
- SeparableConv2D: Convolution séparable (depthwise + pointwise)
- TransposeConv2D: Déconvolution (transposed convolution)
- DilatedConv2D: Convolution dilatée (atrous convolution)
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base import Layer


class Conv1D(Layer):
    """
    Couche de convolution 1D pour séquences temporelles.
    
    Input shape: (batch, length, channels)
    Output shape: (batch, new_length, filters)
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
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.kernel_size = self.kernel_size[0]
        self.strides = strides if isinstance(strides, tuple) else (strides,)
        self.strides = self.strides[0]
        self.padding = padding.lower()
        self.activation_name = activation
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        self.activation = None
        
        if activation:
            from .activation import get_activation
            self.activation = get_activation(activation)
    
    def _pad_input(self, X: np.ndarray) -> np.ndarray:
        """Applique le padding à l'input 1D."""
        if self.padding == 'valid':
            return X
        
        # Padding 'same'
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        
        return np.pad(X, 
                     pad_width=((0, 0), (pad_l, pad_r), (0, 0)),
                     mode='constant')
    
    def _im2col_1d(self, X: np.ndarray) -> np.ndarray:
        """Convertit les séquences en colonnes pour la convolution 1D."""
        batch_size, in_l, in_c = X.shape
        k_l = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_l = (in_l - k_l) // self.strides + 1
        
        # Initialisation de la matrice de colonnes
        cols = np.zeros((batch_size, out_l, k_l, in_c))
        
        for i in range(k_l):
            cols[:, :, i, :] = X[:, 
                                 i:i + out_l * self.strides:self.strides,
                                 :]
        
        return cols.reshape(batch_size * out_l, k_l * in_c)
    
    def _col2im_1d(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en séquences (pour la backward pass)."""
        batch_size, in_l, in_c = X_shape
        k_l = self.kernel_size
        
        out_l = (in_l - k_l) // self.strides + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_l, k_l, in_c)
        
        for i in range(k_l):
            np.add.at(dX[:, 
                        i:i + out_l * self.strides:self.strides,
                        :],
                     dcols_reshaped[:, :, i, :])
        
        return dX
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_l, in_c = input_shape
        k_l = self.kernel_size
        
        # Calcul des dimensions de sortie
        if self.padding == 'same':
            out_l = int(np.ceil(in_l / self.strides))
        else:  # 'valid'
            out_l = (in_l - k_l) // self.strides + 1
        
        # Initialisation des kernels
        fan_in = k_l * in_c
        fan_out = k_l * self.filters
        
        if self.kernel_initializer_name == 'he':
            std = np.sqrt(2.0 / fan_in)
            kernel_shape = (k_l, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * std
        elif self.kernel_initializer_name == 'xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            kernel_shape = (k_l, in_c, self.filters)
            self.parameters['W'] = np.random.uniform(-limit, limit, kernel_shape)
        else:
            kernel_shape = (k_l, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * 0.05
        
        self.gradients['W'] = np.zeros_like(self.parameters['W'])
        
        # Initialisation des biais
        if self.bias_initializer_name == 'zeros':
            self.parameters['b'] = np.zeros((1, 1, self.filters))
        else:
            self.parameters['b'] = np.random.randn(1, 1, self.filters) * 0.05
        
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, out_l, self.filters)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Applique le padding
        if self.padding == 'same':
            x_padded = self._pad_input(x)
        else:
            x_padded = x
        
        batch_size, in_l, in_c = x_padded.shape
        k_l = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_l = (in_l - k_l) // self.strides + 1
        
        # Convolution utilisant im2col
        cols = self._im2col_1d(x_padded)
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        
        # Calcul de la sortie
        self.z = cols @ W_reshaped
        self.z = self.z.reshape(batch_size, out_l, self.filters)
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
        self.gradients['b'] = np.sum(dout, axis=(0, 1), keepdims=True) / batch_size
        
        # Reshape dout
        dout_reshaped = dout.reshape(batch_size * self.output_shape[1], self.filters)
        
        # Gradient par rapport aux poids
        self.gradients['W'] = (self.cols.T @ dout_reshaped).reshape(self.parameters['W'].shape) / batch_size
        
        # Gradient par rapport à l'input
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        dcols = dout_reshaped @ W_reshaped.T
        
        # Convertir les colonnes en séquences
        dx_padded = self._col2im_1d(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
            pad_l = (self.kernel_size - 1) // 2
            pad_r = self.kernel_size // 2
            
            if pad_l > 0 or pad_r > 0:
                dx = dx_padded[:, 
                              pad_l:-pad_r if pad_r > 0 else None,
                              :]
            else:
                dx = dx_padded
        else:
            dx = dx_padded
        
        return dx


class Conv3D(Layer):
    """
    Couche de convolution 3D pour vidéos/volumes 3D.
    
    Input shape: (batch, depth, height, width, channels)
    Output shape: (batch, new_depth, new_height, new_width, filters)
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
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) == 3:
            self.kernel_size = kernel_size
        else:
            raise ValueError("kernel_size must be int or tuple of 3 ints")
        
        if isinstance(strides, int):
            self.strides = (strides, strides, strides)
        elif len(strides) == 3:
            self.strides = strides
        else:
            raise ValueError("strides must be int or tuple of 3 ints")
        
        self.padding = padding.lower()
        self.activation_name = activation
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        self.activation = None
        
        if activation:
            from .activation import get_activation
            self.activation = get_activation(activation)
    
    def _pad_input(self, X: np.ndarray) -> np.ndarray:
        """Applique le padding à l'input 3D."""
        if self.padding == 'valid':
            return X
        
        # Padding 'same'
        pad_d = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] // 2)
        pad_h = ((self.kernel_size[1] - 1) // 2, self.kernel_size[1] // 2)
        pad_w = ((self.kernel_size[2] - 1) // 2, self.kernel_size[2] // 2)
        
        return np.pad(X, 
                     pad_width=((0, 0), (pad_d[0], pad_d[1]), 
                               (pad_h[0], pad_h[1]), 
                               (pad_w[0], pad_w[1]), (0, 0)),
                     mode='constant')
    
    def _im2col_3d(self, X: np.ndarray) -> np.ndarray:
        """Convertit les volumes en colonnes pour la convolution 3D."""
        batch_size, in_d, in_h, in_w, in_c = X.shape
        k_d, k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_d = (in_d - k_d) // self.strides[0] + 1
        out_h = (in_h - k_h) // self.strides[1] + 1
        out_w = (in_w - k_w) // self.strides[2] + 1
        
        # Initialisation de la matrice de colonnes
        cols = np.zeros((batch_size, out_d, out_h, out_w, k_d, k_h, k_w, in_c))
        
        for i in range(k_d):
            for j in range(k_h):
                for k in range(k_w):
                    cols[:, :, :, :, i, j, k, :] = X[:, 
                                                      i:i + out_d * self.strides[0]:self.strides[0],
                                                      j:j + out_h * self.strides[1]:self.strides[1],
                                                      k:k + out_w * self.strides[2]:self.strides[2],
                                                      :]
        
        return cols.reshape(batch_size * out_d * out_h * out_w, k_d * k_h * k_w * in_c)
    
    def _col2im_3d(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en volumes (pour la backward pass)."""
        batch_size, in_d, in_h, in_w, in_c = X_shape
        k_d, k_h, k_w = self.kernel_size
        
        out_d = (in_d - k_d) // self.strides[0] + 1
        out_h = (in_h - k_h) // self.strides[1] + 1
        out_w = (in_w - k_w) // self.strides[2] + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_d, out_h, out_w, k_d, k_h, k_w, in_c)
        
        for i in range(k_d):
            for j in range(k_h):
                for k in range(k_w):
                    np.add.at(dX[:, 
                               i:i + out_d * self.strides[0]:self.strides[0],
                               j:j + out_h * self.strides[1]:self.strides[1],
                               k:k + out_w * self.strides[2]:self.strides[2],
                               :],
                             dcols_reshaped[:, :, :, :, i, j, k, :])
        
        return dX
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_d, in_h, in_w, in_c = input_shape
        k_d, k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        if self.padding == 'same':
            out_d = int(np.ceil(in_d / self.strides[0]))
            out_h = int(np.ceil(in_h / self.strides[1]))
            out_w = int(np.ceil(in_w / self.strides[2]))
        else:  # 'valid'
            out_d = (in_d - k_d) // self.strides[0] + 1
            out_h = (in_h - k_h) // self.strides[1] + 1
            out_w = (in_w - k_w) // self.strides[2] + 1
        
        # Initialisation des kernels
        fan_in = k_d * k_h * k_w * in_c
        fan_out = k_d * k_h * k_w * self.filters
        
        if self.kernel_initializer_name == 'he':
            std = np.sqrt(2.0 / fan_in)
            kernel_shape = (k_d, k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * std
        elif self.kernel_initializer_name == 'xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            kernel_shape = (k_d, k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.uniform(-limit, limit, kernel_shape)
        else:
            kernel_shape = (k_d, k_h, k_w, in_c, self.filters)
            self.parameters['W'] = np.random.randn(*kernel_shape) * 0.05
        
        self.gradients['W'] = np.zeros_like(self.parameters['W'])
        
        # Initialisation des biais
        if self.bias_initializer_name == 'zeros':
            self.parameters['b'] = np.zeros((1, 1, 1, 1, self.filters))
        else:
            self.parameters['b'] = np.random.randn(1, 1, 1, 1, self.filters) * 0.05
        
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, out_d, out_h, out_w, self.filters)
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Applique le padding
        if self.padding == 'same':
            x_padded = self._pad_input(x)
        else:
            x_padded = x
        
        batch_size, in_d, in_h, in_w, in_c = x_padded.shape
        k_d, k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_d = (in_d - k_d) // self.strides[0] + 1
        out_h = (in_h - k_h) // self.strides[1] + 1
        out_w = (in_w - k_w) // self.strides[2] + 1
        
        # Convolution utilisant im2col
        cols = self._im2col_3d(x_padded)
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        
        # Calcul de la sortie
        self.z = cols @ W_reshaped
        self.z = self.z.reshape(batch_size, out_d, out_h, out_w, self.filters)
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
        self.gradients['b'] = np.sum(dout, axis=(0, 1, 2, 3), keepdims=True) / batch_size
        
        # Reshape dout
        dout_reshaped = dout.reshape(batch_size * self.output_shape[1] * 
                                     self.output_shape[2] * self.output_shape[3], 
                                     self.filters)
        
        # Gradient par rapport aux poids
        self.gradients['W'] = (self.cols.T @ dout_reshaped).reshape(self.parameters['W'].shape) / batch_size
        
        # Gradient par rapport à l'input
        W_reshaped = self.parameters['W'].reshape(-1, self.filters)
        dcols = dout_reshaped @ W_reshaped.T
        
        # Convertir les colonnes en volumes
        dx_padded = self._col2im_3d(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
            pad_d = ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] // 2)
            pad_h = ((self.kernel_size[1] - 1) // 2, self.kernel_size[1] // 2)
            pad_w = ((self.kernel_size[2] - 1) // 2, self.kernel_size[2] // 2)
            
            if pad_d[0] > 0 or pad_d[1] > 0 or pad_h[0] > 0 or pad_h[1] > 0 or pad_w[0] > 0 or pad_w[1] > 0:
                dx = dx_padded[:, 
                             pad_d[0]:-pad_d[1] if pad_d[1] > 0 else None,
                             pad_h[0]:-pad_h[1] if pad_h[1] > 0 else None,
                             pad_w[0]:-pad_w[1] if pad_w[1] > 0 else None,
                             :]
            else:
                dx = dx_padded
        else:
            dx = dx_padded
        
        return dx


class DepthwiseConv2D(Layer):
    """
    Couche de convolution depthwise 2D (MobileNet style).
    
    Chaque canal de l'input est convolué séparément.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, new_height, new_width, channels * depth_multiplier)
    """
    
    def __init__(self,
                 kernel_size: int,
                 strides: int = 1,
                 padding: str = 'valid',
                 depth_multiplier: int = 1,
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.depth_multiplier = depth_multiplier
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
    
    def _im2col_depthwise(self, X: np.ndarray) -> np.ndarray:
        """Convertit les images en colonnes pour la convolution depthwise."""
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Initialisation de la matrice de colonnes (séparée par canal)
        cols = np.zeros((batch_size, out_h, out_w, in_c, k_h, k_w))
        
        for i in range(k_h):
            for j in range(k_w):
                cols[:, :, :, :, i, j] = X[:, 
                                           i:i + out_h * self.strides[0]:self.strides[0],
                                           j:j + out_w * self.strides[1]:self.strides[1],
                                           :]
        
        return cols.reshape(batch_size * out_h * out_w, in_c, k_h * k_w)
    
    def _col2im_depthwise(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en images (pour la backward pass)."""
        batch_size, in_h, in_w, in_c = X_shape
        k_h, k_w = self.kernel_size
        
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_h, out_w, in_c, k_h, k_w)
        
        for i in range(k_h):
            for j in range(k_w):
                np.add.at(dX[:, 
                           i:i + out_h * self.strides[0]:self.strides[0],
                           j:j + out_w * self.strides[1]:self.strides[1],
                           :],
                         dcols_reshaped[:, :, :, :, i, j])
        
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
        
        # Initialisation des kernels depthwise
        # Shape: (k_h, k_w, in_c, depth_multiplier)
        fan_in = k_h * k_w
        fan_out = k_h * k_w * self.depth_multiplier
        
        if self.kernel_initializer_name == 'he':
            std = np.sqrt(2.0 / fan_in)
            kernel_shape = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W'] = np.random.randn(*kernel_shape) * std
        elif self.kernel_initializer_name == 'xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            kernel_shape = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W'] = np.random.uniform(-limit, limit, kernel_shape)
        else:
            kernel_shape = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W'] = np.random.randn(*kernel_shape) * 0.05
        
        self.gradients['W'] = np.zeros_like(self.parameters['W'])
        
        # Initialisation des biais
        if self.bias_initializer_name == 'zeros':
            self.parameters['b'] = np.zeros((1, 1, 1, in_c * self.depth_multiplier))
        else:
            self.parameters['b'] = np.random.randn(1, 1, 1, in_c * self.depth_multiplier) * 0.05
        
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, out_h, out_w, in_c * self.depth_multiplier)
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
        
        # Convolution depthwise
        cols = self._im2col_depthwise(x_padded)
        
        # Pour chaque canal, appliquer le kernel correspondant
        output = np.zeros((batch_size, out_h, out_w, in_c, self.depth_multiplier))
        
        for c in range(in_c):
            # Colonnes pour ce canal
            cols_c = cols[:, c, :]  # (batch*out_h*out_w, k_h*k_w)
            # Kernel pour ce canal
            W_c = self.parameters['W'][:, :, c, :].reshape(-1, self.depth_multiplier)  # (k_h*k_w, depth_multiplier)
            # Convolution
            output[:, :, :, c, :] = (cols_c @ W_c).reshape(batch_size, out_h, out_w, self.depth_multiplier)
        
        # Reshape: (batch, out_h, out_w, in_c * depth_multiplier)
        self.z = output.reshape(batch_size, out_h, out_w, in_c * self.depth_multiplier)
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
        in_c = self.input.shape[3]
        
        if self.activation:
            dout = self.activation.backward(dout)
        
        # Gradient par rapport aux biais
        self.gradients['b'] = np.sum(dout, axis=(0, 1, 2), keepdims=True) / batch_size
        
        # Reshape dout pour séparer les canaux
        dout_reshaped = dout.reshape(batch_size, self.output_shape[1], 
                                     self.output_shape[2], in_c, self.depth_multiplier)
        
        # Gradient par rapport aux poids et input
        dcols = np.zeros_like(self.cols)
        dW = np.zeros_like(self.parameters['W'])
        
        for c in range(in_c):
            # Gradient pour ce canal
            dout_c = dout_reshaped[:, :, :, c, :].reshape(batch_size * self.output_shape[1] * 
                                                          self.output_shape[2], 
                                                          self.depth_multiplier)
            cols_c = self.cols[:, c, :]  # (batch*out_h*out_w, k_h*k_w)
            W_c = self.parameters['W'][:, :, c, :].reshape(-1, self.depth_multiplier)
            
            # Gradient par rapport aux poids
            dW[:, :, c, :] = (cols_c.T @ dout_c).reshape(self.kernel_size[0], 
                                                          self.kernel_size[1], 
                                                          self.depth_multiplier) / batch_size
            
            # Gradient par rapport à l'input
            dcols[:, c, :] = dout_c @ W_c.T
        
        self.gradients['W'] = dW
        
        # Convertir les colonnes en images
        dx_padded = self._col2im_depthwise(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
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


class SeparableConv2D(Layer):
    """
    Couche de convolution séparable 2D (depthwise + pointwise).
    
    Plus efficace que Conv2D standard.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, new_height, new_width, filters)
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 padding: str = 'valid',
                 depth_multiplier: int = 1,
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.depth_multiplier = depth_multiplier
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
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_h, in_w, in_c = input_shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie (après depthwise)
        if self.padding == 'same':
            out_h = int(np.ceil(in_h / self.strides[0]))
            out_w = int(np.ceil(in_w / self.strides[1]))
        else:  # 'valid'
            out_h = (in_h - k_h) // self.strides[0] + 1
            out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Initialisation du kernel depthwise
        fan_in_dw = k_h * k_w
        fan_out_dw = k_h * k_w * self.depth_multiplier
        
        if self.kernel_initializer_name == 'he':
            std_dw = np.sqrt(2.0 / fan_in_dw)
            kernel_shape_dw = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W_depthwise'] = np.random.randn(*kernel_shape_dw) * std_dw
        elif self.kernel_initializer_name == 'xavier':
            limit_dw = np.sqrt(6.0 / (fan_in_dw + fan_out_dw))
            kernel_shape_dw = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W_depthwise'] = np.random.uniform(-limit_dw, limit_dw, kernel_shape_dw)
        else:
            kernel_shape_dw = (k_h, k_w, in_c, self.depth_multiplier)
            self.parameters['W_depthwise'] = np.random.randn(*kernel_shape_dw) * 0.05
        
        self.gradients['W_depthwise'] = np.zeros_like(self.parameters['W_depthwise'])
        
        # Initialisation du kernel pointwise (1x1)
        depthwise_channels = in_c * self.depth_multiplier
        fan_in_pw = depthwise_channels
        fan_out_pw = self.filters
        
        if self.kernel_initializer_name == 'he':
            std_pw = np.sqrt(2.0 / fan_in_pw)
            kernel_shape_pw = (1, 1, depthwise_channels, self.filters)
            self.parameters['W_pointwise'] = np.random.randn(*kernel_shape_pw) * std_pw
        elif self.kernel_initializer_name == 'xavier':
            limit_pw = np.sqrt(6.0 / (fan_in_pw + fan_out_pw))
            kernel_shape_pw = (1, 1, depthwise_channels, self.filters)
            self.parameters['W_pointwise'] = np.random.uniform(-limit_pw, limit_pw, kernel_shape_pw)
        else:
            kernel_shape_pw = (1, 1, depthwise_channels, self.filters)
            self.parameters['W_pointwise'] = np.random.randn(*kernel_shape_pw) * 0.05
        
        self.gradients['W_pointwise'] = np.zeros_like(self.parameters['W_pointwise'])
        
        # Initialisation des biais
        if self.bias_initializer_name == 'zeros':
            self.parameters['b'] = np.zeros((1, 1, 1, self.filters))
        else:
            self.parameters['b'] = np.random.randn(1, 1, 1, self.filters) * 0.05
        
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        self.output_shape = (batch_size, out_h, out_w, self.filters)
        return self.output_shape
    
    def _im2col_depthwise(self, X: np.ndarray) -> np.ndarray:
        """Convertit les images en colonnes pour la convolution depthwise."""
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w = self.kernel_size
        
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        cols = np.zeros((batch_size, out_h, out_w, in_c, k_h, k_w))
        
        for i in range(k_h):
            for j in range(k_w):
                cols[:, :, :, :, i, j] = X[:, 
                                           i:i + out_h * self.strides[0]:self.strides[0],
                                           j:j + out_w * self.strides[1]:self.strides[1],
                                           :]
        
        return cols.reshape(batch_size * out_h * out_w, in_c, k_h * k_w)
    
    def _col2im_depthwise(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en images (pour la backward pass)."""
        batch_size, in_h, in_w, in_c = X_shape
        k_h, k_w = self.kernel_size
        
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_h, out_w, in_c, k_h, k_w)
        
        for i in range(k_h):
            for j in range(k_w):
                np.add.at(dX[:, 
                           i:i + out_h * self.strides[0]:self.strides[0],
                           j:j + out_w * self.strides[1]:self.strides[1],
                           :],
                         dcols_reshaped[:, :, :, :, i, j])
        
        return dX
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Applique le padding
        if self.padding == 'same':
            x_padded = self._pad_input(x)
        else:
            x_padded = x
        
        batch_size, in_h, in_w, in_c = x_padded.shape
        k_h, k_w = self.kernel_size
        
        # Calcul des dimensions de sortie après depthwise
        out_h = (in_h - k_h) // self.strides[0] + 1
        out_w = (in_w - k_w) // self.strides[1] + 1
        
        # Étape 1: Convolution depthwise
        cols = self._im2col_depthwise(x_padded)
        depthwise_output = np.zeros((batch_size, out_h, out_w, in_c, self.depth_multiplier))
        
        for c in range(in_c):
            cols_c = cols[:, c, :]
            W_c = self.parameters['W_depthwise'][:, :, c, :].reshape(-1, self.depth_multiplier)
            depthwise_output[:, :, :, c, :] = (cols_c @ W_c).reshape(batch_size, out_h, out_w, self.depth_multiplier)
        
        depthwise_output = depthwise_output.reshape(batch_size, out_h, out_w, in_c * self.depth_multiplier)
        
        # Stocke pour backward
        self.depthwise_cols = cols
        self.x_shape = x_padded.shape
        
        # Étape 2: Convolution pointwise (1x1)
        # Reshape pour pointwise: (batch, out_h, out_w, depthwise_channels)
        pointwise_input = depthwise_output.reshape(batch_size * out_h * out_w, in_c * self.depth_multiplier)
        
        # Convolution pointwise
        self.z = pointwise_input @ self.parameters['W_pointwise'].reshape(in_c * self.depth_multiplier, self.filters)
        self.z = self.z.reshape(batch_size, out_h, out_w, self.filters)
        self.z += self.parameters['b']
        
        # Stocke pour backward
        self.depthwise_output = depthwise_output
        
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
        out_h, out_w = self.output_shape[1], self.output_shape[2]
        dout_reshaped = dout.reshape(batch_size * out_h * out_w, self.filters)
        
        # Étape 1: Backward de la convolution pointwise
        # Gradient par rapport aux poids pointwise
        pointwise_input = self.depthwise_output.reshape(batch_size * out_h * out_w, -1)
        self.gradients['W_pointwise'] = (pointwise_input.T @ dout_reshaped).reshape(
            self.parameters['W_pointwise'].shape) / batch_size
        
        # Gradient par rapport à la sortie depthwise
        W_pointwise_reshaped = self.parameters['W_pointwise'].reshape(-1, self.filters)
        d_depthwise = dout_reshaped @ W_pointwise_reshaped.T
        d_depthwise = d_depthwise.reshape(batch_size, out_h, out_w, -1)
        
        # Étape 2: Backward de la convolution depthwise
        in_c = self.input.shape[3]
        d_depthwise_reshaped = d_depthwise.reshape(batch_size, out_h, out_w, in_c, self.depth_multiplier)
        
        dcols = np.zeros_like(self.depthwise_cols)
        dW_depthwise = np.zeros_like(self.parameters['W_depthwise'])
        
        for c in range(in_c):
            dout_c = d_depthwise_reshaped[:, :, :, c, :].reshape(batch_size * out_h * out_w, self.depth_multiplier)
            cols_c = self.depthwise_cols[:, c, :]
            W_c = self.parameters['W_depthwise'][:, :, c, :].reshape(-1, self.depth_multiplier)
            
            # Gradient par rapport aux poids depthwise
            dW_depthwise[:, :, c, :] = (cols_c.T @ dout_c).reshape(
                self.kernel_size[0], self.kernel_size[1], self.depth_multiplier) / batch_size
            
            # Gradient par rapport à l'input
            dcols[:, c, :] = dout_c @ W_c.T
        
        self.gradients['W_depthwise'] = dW_depthwise
        
        # Convertir les colonnes en images
        dx_padded = self._col2im_depthwise(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
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


class TransposeConv2D(Layer):
    """
    Couche de déconvolution 2D (transposed convolution).
    
    Utilisée pour les réseaux génératifs et la segmentation.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, new_height, new_width, filters)
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 padding: str = 'valid',
                 output_padding: int = 0,
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.activation_name = activation
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        self.activation = None
        
        if activation:
            from .activation import get_activation
            self.activation = get_activation(activation)
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_h, in_w, in_c = input_shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        
        # Calcul des dimensions de sortie pour transpose convolution
        if self.padding == 'same':
            out_h = in_h * s_h
            out_w = in_w * s_w
        else:  # 'valid'
            out_h = (in_h - 1) * s_h + k_h + self.output_padding[0]
            out_w = (in_w - 1) * s_w + k_w + self.output_padding[1]
        
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
        batch_size, in_h, in_w, in_c = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        
        # Calcul des dimensions de sortie
        if self.padding == 'same':
            out_h = in_h * s_h
            out_w = in_w * s_w
        else:  # 'valid'
            out_h = (in_h - 1) * s_h + k_h + self.output_padding[0]
            out_w = (in_w - 1) * s_w + k_w + self.output_padding[1]
        
        # Initialiser la sortie avec zéros
        output = np.zeros((batch_size, out_h, out_w, self.filters))
        
        # Transpose convolution: appliquer le kernel sur l'input et placer dans la sortie
        for i in range(in_h):
            for j in range(in_w):
                h_start = i * s_h
                h_end = min(h_start + k_h, out_h)
                w_start = j * s_w
                w_end = min(w_start + k_w, out_w)
                
                if h_end > h_start and w_end > w_start:
                    # Extraire la fenêtre de sortie
                    h_window = h_end - h_start
                    w_window = w_end - w_start
                    
                    # Pour chaque filtre
                    for f in range(self.filters):
                        # Multiplier l'input avec le kernel (transposé)
                        # x shape: (batch, 1, 1, in_c)
                        # W shape: (k_h, k_w, in_c, filters)
                        # On prend la fenêtre du kernel qui correspond
                        kernel_window = self.parameters['W'][:h_window, :w_window, :, f]
                        # kernel_window: (h_window, w_window, in_c)
                        
                        # Calcul: sum over channels
                        output[:, h_start:h_end, w_start:w_end, f] += np.sum(
                            x[:, i:i+1, j:j+1, :, np.newaxis, np.newaxis] * 
                            kernel_window[np.newaxis, :, :, :],
                            axis=3
                        )
        
        self.z = output + self.parameters['b']
        
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
        
        # Gradient par rapport aux poids et input
        in_h, in_w = self.input.shape[1], self.input.shape[2]
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        
        dW = np.zeros_like(self.parameters['W'])
        dx = np.zeros_like(self.input)
        
        # Calculer les gradients
        for i in range(in_h):
            for j in range(in_w):
                h_start = i * s_h
                h_end = min(h_start + k_h, dout.shape[1])
                w_start = j * s_w
                w_end = min(w_start + k_w, dout.shape[2])
                
                if h_end > h_start and w_end > w_start:
                    dout_window = dout[:, h_start:h_end, w_start:w_end, :]
                    h_window = h_end - h_start
                    w_window = w_end - w_start
                    
                    # Gradient par rapport aux poids
                    for f in range(self.filters):
                        # dW[h_window, w_window, in_c, f] += sum over batch
                        dW[:h_window, :w_window, :, f] += np.sum(
                            self.input[:, i:i+1, j:j+1, :, np.newaxis, np.newaxis] * 
                            dout_window[:, :, :, f:f+1, np.newaxis],
                            axis=0
                        ) / batch_size
                    
                    # Gradient par rapport à l'input
                    # dx[:, i, j, c] += sum over (h, w, filters)
                    for c in range(self.input.shape[3]):
                        kernel_window = self.parameters['W'][:h_window, :w_window, c, :]
                        dx[:, i, j, c] += np.sum(
                            dout_window * kernel_window[np.newaxis, :, :, :],
                            axis=(1, 2, 3)
                        )
        
        self.gradients['W'] = dW
        
        return dx


class DilatedConv2D(Layer):
    """
    Couche de convolution dilatée 2D (atrous convolution).
    
    Permet d'augmenter le champ récepteur sans augmenter les paramètres.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, new_height, new_width, filters)
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 dilation_rate: int = 1,
                 strides: int = 1,
                 padding: str = 'valid',
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'he',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, tuple) else (dilation_rate, dilation_rate)
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
        """Applique le padding à l'input avec prise en compte de la dilatation."""
        if self.padding == 'valid':
            return X
        
        # Padding 'same' avec dilatation
        effective_k_h = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation_rate[0] - 1)
        effective_k_w = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation_rate[1] - 1)
        
        pad_h = ((effective_k_h - 1) // 2, effective_k_h // 2)
        pad_w = ((effective_k_w - 1) // 2, effective_k_w // 2)
        
        return np.pad(X, 
                     pad_width=((0, 0), (pad_h[0], pad_h[1]), 
                               (pad_w[0], pad_w[1]), (0, 0)),
                     mode='constant')
    
    def _im2col_dilated(self, X: np.ndarray) -> np.ndarray:
        """Convertit les images en colonnes pour la convolution dilatée."""
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w = self.kernel_size
        d_h, d_w = self.dilation_rate
        
        # Taille effective du kernel avec dilatation
        effective_k_h = k_h + (k_h - 1) * (d_h - 1)
        effective_k_w = k_w + (k_w - 1) * (d_w - 1)
        
        # Calcul des dimensions de sortie
        out_h = (in_h - effective_k_h) // self.strides[0] + 1
        out_w = (in_w - effective_k_w) // self.strides[1] + 1
        
        # Initialisation de la matrice de colonnes
        cols = np.zeros((batch_size, out_h, out_w, k_h, k_w, in_c))
        
        for i in range(k_h):
            for j in range(k_w):
                # Position dans l'image avec dilatation
                h_pos = i * d_h
                w_pos = j * d_w
                
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_idx = oh * self.strides[0] + h_pos
                        w_idx = ow * self.strides[1] + w_pos
                        
                        if 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                            cols[:, oh, ow, i, j, :] = X[:, h_idx, w_idx, :]
        
        return cols.reshape(batch_size * out_h * out_w, k_h * k_w * in_c)
    
    def _col2im_dilated(self, dcols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Convertit les colonnes en images (pour la backward pass)."""
        batch_size, in_h, in_w, in_c = X_shape
        k_h, k_w = self.kernel_size
        d_h, d_w = self.dilation_rate
        
        effective_k_h = k_h + (k_h - 1) * (d_h - 1)
        effective_k_w = k_w + (k_w - 1) * (d_w - 1)
        
        out_h = (in_h - effective_k_h) // self.strides[0] + 1
        out_w = (in_w - effective_k_w) // self.strides[1] + 1
        
        dX = np.zeros(X_shape)
        dcols_reshaped = dcols.reshape(batch_size, out_h, out_w, k_h, k_w, in_c)
        
        for i in range(k_h):
            for j in range(k_w):
                # Position dans l'image avec dilatation
                h_pos = i * d_h
                w_pos = j * d_w
                
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_idx = oh * self.strides[0] + h_pos
                        w_idx = ow * self.strides[1] + w_pos
                        
                        if 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                            np.add.at(dX[:, h_idx, w_idx, :],
                                     dcols_reshaped[:, oh, ow, i, j, :])
        
        return dX
    
    def initialize(self, input_shape: tuple) -> tuple:
        batch_size, in_h, in_w, in_c = input_shape
        k_h, k_w = self.kernel_size
        d_h, d_w = self.dilation_rate
        
        # Taille effective du kernel avec dilatation
        effective_k_h = k_h + (k_h - 1) * (d_h - 1)
        effective_k_w = k_w + (k_w - 1) * (d_w - 1)
        
        # Calcul des dimensions de sortie
        if self.padding == 'same':
            out_h = int(np.ceil(in_h / self.strides[0]))
            out_w = int(np.ceil(in_w / self.strides[1]))
        else:  # 'valid'
            out_h = (in_h - effective_k_h) // self.strides[0] + 1
            out_w = (in_w - effective_k_w) // self.strides[1] + 1
        
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
        d_h, d_w = self.dilation_rate
        
        effective_k_h = k_h + (k_h - 1) * (d_h - 1)
        effective_k_w = k_w + (k_w - 1) * (d_w - 1)
        
        # Calcul des dimensions de sortie
        out_h = (in_h - effective_k_h) // self.strides[0] + 1
        out_w = (in_w - effective_k_w) // self.strides[1] + 1
        
        # Convolution utilisant im2col avec dilatation
        cols = self._im2col_dilated(x_padded)
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
        dx_padded = self._col2im_dilated(dcols, self.x_shape)
        
        # Enlever le padding si nécessaire
        if self.padding == 'same':
            k_h, k_w = self.kernel_size
            d_h, d_w = self.dilation_rate
            effective_k_h = k_h + (k_h - 1) * (d_h - 1)
            effective_k_w = k_w + (k_w - 1) * (d_w - 1)
            
            pad_h = ((effective_k_h - 1) // 2, effective_k_h // 2)
            pad_w = ((effective_k_w - 1) // 2, effective_k_w // 2)
            
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