"""Infrastructure principale du framework Weli."""

from .debugging import (
    check_gradients,
    check_numerics,
    debug_model,
    tensor_summary,
    visualize_graph,
)
from .device import (
    Device,
    available_devices,
    cuda_available,
    device_scope,
    get_device,
    set_default_device,
    to_cpu,
    to_device,
)
from .memory_management import (
    MemoryManager,
    cleanup_memory,
    get_memory_usage,
    tensor_memory_bytes,
)
from .profiling import (
    ProfileRecord,
    Profiler,
    get_profiler,
    profile,
    profile_operation,
)

__all__ = [
    "Device",
    "MemoryManager",
    "ProfileRecord",
    "Profiler",
    "available_devices",
    "check_gradients",
    "check_numerics",
    "cleanup_memory",
    "cuda_available",
    "debug_model",
    "device_scope",
    "get_device",
    "get_memory_usage",
    "get_profiler",
    "profile",
    "profile_operation",
    "set_default_device",
    "tensor_memory_bytes",
    "tensor_summary",
    "to_cpu",
    "to_device",
    "visualize_graph",
]
