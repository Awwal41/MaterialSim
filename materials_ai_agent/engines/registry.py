"""Runtime registry of MD engine adapters."""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import EngineAdapter

_ENGINES: Dict[str, Callable[[], EngineAdapter]] = {}


def register_engine(name: str, factory: Callable[[], EngineAdapter]) -> None:
    """Register an engine factory under a lowercase name."""
    _ENGINES[name.lower()] = factory


def get_engine(name: str) -> EngineAdapter:
    """Instantiate a registered engine by name."""
    key = (name or "").lower()
    if key not in _ENGINES:
        raise KeyError(
            f"Unknown engine '{name}'. Registered: {sorted(_ENGINES)}"
        )
    return _ENGINES[key]()


def list_engines() -> List[str]:
    """All registered engine names (installed or not)."""
    return sorted(_ENGINES)


def available_engines() -> List[str]:
    """Engine names whose backend dependency is actually importable/usable."""
    out: List[str] = []
    for name in list_engines():
        try:
            if get_engine(name).available():
                out.append(name)
        except Exception:
            continue
    return out
