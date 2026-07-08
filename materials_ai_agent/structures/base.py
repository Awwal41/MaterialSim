"""Structure builder interface + result container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

from .topology import Topology

if TYPE_CHECKING:
    from ..spec import SystemSpec


@dataclass
class BuiltSystem:
    """A prepared system ready to hand to an engine."""

    atoms: Any  # ase.Atoms
    topology: Optional[Topology] = None
    label: str = "system"
    warnings: List[str] = field(default_factory=list)


class StructureBuilder(ABC):
    """Base class for structure/topology builders."""

    name: str = "builder"

    @abstractmethod
    def can_build(self, system: "SystemSpec") -> bool:
        """Whether this builder applies to the given system spec."""

    @abstractmethod
    def build(self, system: "SystemSpec", *, mp_api_key: Optional[str] = None) -> BuiltSystem:
        """Construct the system, raising a clear error if it cannot."""
