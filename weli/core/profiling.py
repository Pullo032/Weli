"""Profilage temporel et mémoire des opérations Weli."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from time import perf_counter

from .memory_management import get_memory_usage


@dataclass
class ProfileRecord:
    """Statistiques agrégées d'une opération profilée."""

    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    memory_delta_bytes: int = 0

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls else 0.0

    def as_dict(self):
        return {
            "name": self.name,
            "calls": self.calls,
            "total_time": self.total_time,
            "average_time": self.average_time,
            "min_time": self.min_time if self.calls else 0.0,
            "max_time": self.max_time,
            "memory_delta_bytes": self.memory_delta_bytes,
        }


class Profiler:
    """Collecteur thread-safe de durées et variations mémoire."""

    def __init__(self, device=None):
        self.device = device
        self._records = {}
        self._lock = RLock()

    @contextmanager
    def measure(self, name):
        """Mesure une opération avec un bloc ``with``."""
        operation_name = str(name)
        started_at = perf_counter()
        before = get_memory_usage(self.device).get("allocated_bytes")
        try:
            yield
        finally:
            elapsed = perf_counter() - started_at
            after = get_memory_usage(self.device).get("allocated_bytes")
            memory_delta = (
                int(after - before) if before is not None and after is not None else 0
            )
            with self._lock:
                record = self._records.setdefault(
                    operation_name, ProfileRecord(operation_name)
                )
                record.calls += 1
                record.total_time += elapsed
                record.min_time = min(record.min_time, elapsed)
                record.max_time = max(record.max_time, elapsed)
                record.memory_delta_bytes += memory_delta

    def profile(self, function=None, *, name=None):
        """Décore une fonction pour ajouter sa durée aux statistiques."""
        if function is None:
            return lambda wrapped: self.profile(wrapped, name=name)

        operation_name = name or function.__qualname__

        @wraps(function)
        def measured(*args, **kwargs):
            with self.measure(operation_name):
                return function(*args, **kwargs)

        return measured

    def summary(self):
        """Retourne les statistiques sous forme de dictionnaires triés."""
        with self._lock:
            records = [record.as_dict() for record in self._records.values()]
        return sorted(records, key=lambda record: record["total_time"], reverse=True)

    def format_summary(self):
        """Formate les statistiques pour affichage ou journalisation."""
        rows = self.summary()
        if not rows:
            return "Aucune opération profilée."
        header = "{:<40} {:>8} {:>12} {:>12} {:>14}".format(
            "Opération", "Appels", "Total (s)", "Moyenne (s)", "Mémoire (octets)"
        )
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(
                "{:<40} {:>8d} {:>12.6f} {:>12.6f} {:>14d}".format(
                    row["name"][:40],
                    row["calls"],
                    row["total_time"],
                    row["average_time"],
                    row["memory_delta_bytes"],
                )
            )
        return "\n".join(lines)

    def reset(self):
        """Efface les mesures enregistrées."""
        with self._lock:
            self._records.clear()


_default_profiler = Profiler()


def profile(function=None, *, name=None, profiler=None):
    """Décorateur de profilage utilisant le profiler global par défaut."""
    selected_profiler = profiler or _default_profiler
    return selected_profiler.profile(function, name=name)


def profile_operation(name, profiler=None):
    """Retourne un contexte de mesure pour une opération."""
    return (profiler or _default_profiler).measure(name)


def get_profiler():
    """Retourne le profiler global par défaut."""
    return _default_profiler


__all__ = [
    "ProfileRecord",
    "Profiler",
    "get_profiler",
    "profile",
    "profile_operation",
]
