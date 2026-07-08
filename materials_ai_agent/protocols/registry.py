"""Runtime registry of simulation protocols."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .base import Protocol

_PROTOCOLS: Dict[str, Callable[[Dict[str, Any]], Protocol]] = {}


def register_protocol(name: str, factory: Callable[[Dict[str, Any]], Protocol]) -> None:
    _PROTOCOLS[name.lower()] = factory


def get_protocol(name: str, params: Dict[str, Any] | None = None) -> Protocol:
    key = (name or "equilibrium").lower()
    if key not in _PROTOCOLS:
        raise KeyError(
            f"Unknown protocol '{name}'. Registered: {sorted(_PROTOCOLS)}"
        )
    return _PROTOCOLS[key](params or {})


def list_protocols() -> List[str]:
    return sorted(_PROTOCOLS)
