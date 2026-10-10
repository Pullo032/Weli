"""Outils de mesure et de libération de mémoire CPU/GPU."""

import gc
import sys

try:
    import psutil
except ImportError:
    psutil = None

from .device import cp, get_device


def _process_rss_bytes():
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("Kernel32.dll")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if not get_info(process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def tensor_memory_bytes(tensor) -> int:
    """Retourne la taille en octets d'un tableau ou tenseur."""
    size = getattr(tensor, "nbytes", None)
    if size is None:
        raise TypeError("tensor must expose an 'nbytes' attribute")
    return int(size)


def get_memory_usage(device=None):
    """Retourne les mesures mémoire disponibles pour un device."""
    resolved = get_device(device)
    if resolved.type == "cpu":
        rss = _process_rss_bytes()
        return {
            "device": str(resolved),
            "process_rss_bytes": rss,
            "allocated_bytes": rss,
        }

    with cp.cuda.Device(resolved.index):
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        return {
            "device": str(resolved),
            "allocated_bytes": int(pool.used_bytes()),
            "reserved_bytes": int(pool.total_bytes()),
            "free_bytes": int(pool.free_bytes()),
            "pinned_free_blocks": int(pinned_pool.n_free_blocks()),
        }


def cleanup_memory(device=None):
    """Libère les objets Python inutilisés et le cache mémoire CUDA demandé."""
    resolved = get_device(device)
    collected = gc.collect()
    freed_gpu_bytes = 0
    if resolved.type == "cuda":
        with cp.cuda.Device(resolved.index):
            pool = cp.get_default_memory_pool()
            freed_gpu_bytes = int(pool.total_bytes() - pool.used_bytes())
            pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    return {
        "device": str(resolved),
        "collected_objects": collected,
        "freed_gpu_bytes": freed_gpu_bytes,
    }


class MemoryManager:
    """Façade pratique pour mesurer et nettoyer la mémoire."""

    def __init__(self, device=None):
        self.device = get_device(device)

    def usage(self):
        return get_memory_usage(self.device)

    def cleanup(self):
        return cleanup_memory(self.device)

    def tensor_size(self, tensor):
        return tensor_memory_bytes(tensor)


__all__ = [
    "MemoryManager",
    "cleanup_memory",
    "get_memory_usage",
    "tensor_memory_bytes",
]
