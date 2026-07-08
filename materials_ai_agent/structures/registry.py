"""Runtime registry that dispatches a SystemSpec to the right builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from .base import BuiltSystem, StructureBuilder

if TYPE_CHECKING:
    from ..spec import SystemSpec

# Ordered: most specific first. ``can_build`` decides applicability.
_BUILDERS: List[Callable[[], StructureBuilder]] = []
_BY_NAME: dict = {}


def register_builder(name: str, factory: Callable[[], StructureBuilder], *, priority: int = 100) -> None:
    _BY_NAME[name] = (priority, factory)
    ordered = sorted(_BY_NAME.values(), key=lambda pf: pf[0])
    _BUILDERS.clear()
    _BUILDERS.extend(f for _, f in ordered)


def list_builders() -> List[str]:
    return sorted(_BY_NAME)


def build_system(system: "SystemSpec", *, mp_api_key: Optional[str] = None) -> BuiltSystem:
    """Find the first applicable builder and build the system."""
    for factory in _BUILDERS:
        builder = factory()
        try:
            applicable = builder.can_build(system)
        except Exception:
            applicable = False
        if applicable:
            return builder.build(system, mp_api_key=mp_api_key)
    raise ValueError(
        "No structure builder could handle this system. Provide a formula, "
        "a SMILES string, a structure file, or a Materials Project id."
    )
