import numpy as np
from typing import Optional, Dict, Any
from .base import Layer


class LayerNorm(Layer):
    """
    Normalisation par couche: chaque échantillon est normalisé sur son dernier axe (features).
    """

    def __init__(self, epsilon: float = 1e-5, name: Optional[str] = None):
        super().__init__(name)
        self.epsilon = epsilon
        self._x_hat = None
        self._std = None

    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) < 2:
            raise ValueError(f"LayerNorm expects at least 2D input, got {len(input_shape)}D")

        param_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.parameters['gamma'] = np.ones(param_shape)
        self.parameters['beta'] = np.zeros(param_shape)
        self.gradients['gamma'] = np.zeros(param_shape)
        self.gradients['beta'] = np.zeros(param_shape)

        self.input_shape = input_shape
        self.output_shape = input_shape
        return input_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self._std = np.sqrt(var + self.epsilon)
        self._x_hat = (x - mean) / self._std
        self.output = self.parameters['gamma'] * self._x_hat + self.parameters['beta']
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_axes = tuple(range(dout.ndim - 1))
        self.gradients['gamma'] = np.sum(dout * self._x_hat, axis=batch_axes, keepdims=True)
        self.gradients['beta'] = np.sum(dout, axis=batch_axes, keepdims=True)

        dx_hat = dout * self.parameters['gamma']
        dx = (dx_hat
              - np.mean(dx_hat, axis=-1, keepdims=True)
              - self._x_hat * np.mean(dx_hat * self._x_hat, axis=-1, keepdims=True)) / self._std
        return dx

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['epsilon'] = self.epsilon
        return config


class _ChannelGroupNorm(Layer):
    """
    Base commune pour les normalisations par groupes de canaux (entrée channels-last).
    """

    def __init__(self, epsilon: float, name: Optional[str]):
        super().__init__(name)
        self.epsilon = epsilon
        self._groups = None
        self._x_hat = None
        self._std = None
        self._reduce_axes = None

    def _num_groups(self, channels: int) -> int:
        raise NotImplementedError

    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) < 2:
            raise ValueError(f"{self.__class__.__name__} expects at least 2D input, got {len(input_shape)}D")

        channels = input_shape[-1]
        groups = self._num_groups(channels)
        if groups < 1 or channels % groups != 0:
            raise ValueError(
                f"{self.__class__.__name__}: {channels} canaux ne sont pas divisibles en {groups} groupes"
            )
        self._groups = groups

        param_shape = (1,) * (len(input_shape) - 1) + (channels,)
        self.parameters['gamma'] = np.ones(param_shape)
        self.parameters['beta'] = np.zeros(param_shape)
        self.gradients['gamma'] = np.zeros(param_shape)
        self.gradients['beta'] = np.zeros(param_shape)

        self.input_shape = input_shape
        self.output_shape = input_shape
        return input_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        ndim = x.ndim
        grouped = x.reshape(x.shape[:-1] + (self._groups, x.shape[-1] // self._groups))

        # Moyenne et variance sur les axes spatiaux et les canaux du groupe
        self._reduce_axes = tuple(range(1, ndim - 1)) + (ndim,)
        mean = np.mean(grouped, axis=self._reduce_axes, keepdims=True)
        var = np.var(grouped, axis=self._reduce_axes, keepdims=True)
        self._std = np.sqrt(var + self.epsilon)
        self._x_hat = (grouped - mean) / self._std

        x_hat = self._x_hat.reshape(x.shape)
        self.output = self.parameters['gamma'] * x_hat + self.parameters['beta']
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_axes = tuple(range(dout.ndim - 1))
        x_hat = self._x_hat.reshape(dout.shape)
        self.gradients['gamma'] = np.sum(dout * x_hat, axis=batch_axes, keepdims=True)
        self.gradients['beta'] = np.sum(dout, axis=batch_axes, keepdims=True)

        dx_hat = (dout * self.parameters['gamma']).reshape(self._x_hat.shape)
        axes = self._reduce_axes
        dx = (dx_hat
              - np.mean(dx_hat, axis=axes, keepdims=True)
              - self._x_hat * np.mean(dx_hat * self._x_hat, axis=axes, keepdims=True)) / self._std
        return dx.reshape(dout.shape)


class GroupNorm(_ChannelGroupNorm):
    """
    Normalisation par groupes de canaux (entrée channels-last: (batch, ..., canaux)).
    """

    def __init__(self, groups: int = 32, epsilon: float = 1e-5, name: Optional[str] = None):
        super().__init__(epsilon, name)
        self.groups = groups

    def _num_groups(self, channels: int) -> int:
        return self.groups

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'groups': self.groups, 'epsilon': self.epsilon})
        return config


class InstanceNorm(_ChannelGroupNorm):
    """
    Normalisation par instance: chaque canal de chaque échantillon est normalisé séparément.
    """

    def __init__(self, epsilon: float = 1e-5, name: Optional[str] = None):
        super().__init__(epsilon, name)

    def _num_groups(self, channels: int) -> int:
        return channels

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['epsilon'] = self.epsilon
        return config


class WeightNorm(Layer):
    """
    Réparamétrage des poids: W = g * v / ||v||, avec une norme par unité de sortie.

    Enveloppe une couche possédant un poids 'W' (Dense, Conv2D, ...). Les paramètres
    entraînables sont g, v et, si présent, le biais b.
    """

    def __init__(self, layer: Any, name: Optional[str] = None):
        if isinstance(layer, dict):
            from ..models.registry import registry
            layer = registry.deserialize_layer(layer)
        super().__init__(name)
        self.layer = layer
        self._norm = None

    @staticmethod
    def _reduce_axes(weight: np.ndarray) -> tuple:
        return tuple(range(weight.ndim - 1))

    def initialize(self, input_shape: tuple) -> tuple:
        output_shape = self.layer.initialize(input_shape)
        if 'W' not in self.layer.parameters:
            raise ValueError("WeightNorm requiert une couche possédant un poids 'W'")

        v = self.layer.parameters.pop('W')
        self.parameters['v'] = v
        self.parameters['g'] = np.sqrt(np.sum(v * v, axis=self._reduce_axes(v), keepdims=True))
        self.gradients['v'] = np.zeros_like(v)
        self.gradients['g'] = np.zeros_like(self.parameters['g'])

        if 'b' in self.layer.parameters:
            self.parameters['b'] = self.layer.parameters.pop('b')
            self.gradients['b'] = np.zeros_like(self.parameters['b'])

        self.input_shape = input_shape
        self.output_shape = output_shape
        return output_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        v, g = self.parameters['v'], self.parameters['g']
        self._norm = np.sqrt(np.sum(v * v, axis=self._reduce_axes(v), keepdims=True))
        self.layer.parameters['W'] = g * v / self._norm
        if 'b' in self.parameters:
            self.layer.parameters['b'] = self.parameters['b']
        self.layer.training = self.training
        self.output = self.layer.forward(x)
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = self.layer.backward(dout)
        dW = self.layer.gradients['W']
        v, g, norm = self.parameters['v'], self.parameters['g'], self._norm
        axes = self._reduce_axes(v)

        self.gradients['g'] = np.sum(dW * v / norm, axis=axes, keepdims=True)
        self.gradients['v'] = (g / norm) * (dW - v * np.sum(dW * v, axis=axes, keepdims=True) / norm ** 2)
        if 'b' in self.parameters:
            self.gradients['b'] = self.layer.gradients['b']
        return dx

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['layer'] = self.layer.get_config()
        return config
