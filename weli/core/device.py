"""Sélection des devices CPU/GPU et transfert des tableaux."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


_default_device = "auto"
_active_device = ContextVar("weli_active_device", default=None)
DeviceLike = Union["Device", str, None]


def cuda_available() -> bool:
    """Indique si CuPy peut accéder à au moins un périphérique CUDA."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def available_devices():
    """Retourne les devices disponibles, en commençant toujours par le CPU."""
    devices = [Device("cpu")]
    if cuda_available():
        devices.extend(Device("cuda:{}".format(index))
                       for index in range(cp.cuda.runtime.getDeviceCount()))
    return devices


@dataclass(frozen=True)
class Device:
    """Représente un device CPU ou un périphérique CUDA."""

    name: str = "cpu"

    def __post_init__(self):
        normalized = str(self.name).lower()
        if normalized == "gpu":
            normalized = "cuda:0"
        if normalized == "cuda":
            normalized = "cuda:0"
        if normalized != "cpu":
            if not normalized.startswith("cuda:"):
                raise ValueError("Device must be 'cpu' or 'cuda[:index]'")
            try:
                index = int(normalized.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("CUDA device index must be an integer") from exc
            if index < 0:
                raise ValueError("CUDA device index must be non-negative")
            if not cuda_available():
                raise RuntimeError(
                    "CUDA was requested, but CuPy and an available CUDA device "
                    "are required"
                )
            if index >= cp.cuda.runtime.getDeviceCount():
                raise ValueError("CUDA device index {} is not available".format(index))
            normalized = "cuda:{}".format(index)
        object.__setattr__(self, "name", normalized)

    @property
    def type(self):
        return "cpu" if self.name == "cpu" else "cuda"

    @property
    def index(self):
        return None if self.type == "cpu" else int(self.name.split(":", 1)[1])

    @property
    def array_module(self):
        return np if self.type == "cpu" else cp

    def __str__(self):
        return self.name


def get_device(device: DeviceLike = None) -> Device:
    """Résout un device explicite, le contexte actif ou le défaut global."""
    if isinstance(device, Device):
        return device
    if device is None:
        device = _active_device.get() or _default_device
    normalized = str(device).lower()
    if normalized == "auto":
        normalized = "cuda:0" if cuda_available() else "cpu"
    return Device(normalized)


def set_default_device(device: DeviceLike = "auto") -> Device:
    """Configure le device utilisé par défaut et le retourne."""
    global _default_device
    resolved = get_device(device)
    _default_device = resolved.name
    return resolved


@contextmanager
def device_scope(device: DeviceLike):
    """Sélectionne temporairement un device dans le contexte courant."""
    resolved = get_device(device)
    token = _active_device.set(resolved.name)
    try:
        yield resolved
    finally:
        _active_device.reset(token)


def is_gpu_array(array) -> bool:
    """Indique si un tableau est un tableau CuPy."""
    return cp is not None and isinstance(array, cp.ndarray)


def array_module(array):
    """Retourne NumPy ou CuPy selon le type du tableau."""
    return cp if is_gpu_array(array) else np


def to_device(array, device: DeviceLike = None, copy: bool = False):
    """Transfère un tableau vers le device demandé."""
    resolved = get_device(device)
    if resolved.type == "cuda":
        with cp.cuda.Device(resolved.index):
            return cp.array(array, copy=True) if copy else cp.asarray(array)
    if is_gpu_array(array):
        array = cp.asnumpy(array)
    return np.array(array, copy=True) if copy else np.asarray(array)


def to_cpu(array, copy: bool = False):
    """Transfère un tableau vers le CPU."""
    return to_device(array, "cpu", copy=copy)


__all__ = [
    "Device",
    "DeviceLike",
    "array_module",
    "available_devices",
    "cuda_available",
    "device_scope",
    "get_device",
    "is_gpu_array",
    "set_default_device",
    "to_cpu",
    "to_device",
]
