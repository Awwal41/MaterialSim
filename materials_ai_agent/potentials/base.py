"""Interface every interatomic potential provider implements.

A provider declares which engines it targets and which chemistry it supports,
and implements only the engine hooks that make sense for it. This is what makes
the platform force-field-agnostic: adding a new potential means registering a
new provider, touching no engine or core code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..engines.base import ResolvedJob
    from ..spec import PotentialSpec


class PotentialNotSupported(Exception):
    """Raised when a provider cannot service a given engine or system."""


@dataclass
class LammpsPotential:
    """Engine-neutral description of a LAMMPS pair style for a run."""

    units: str = "metal"
    atom_style: str = "atomic"
    pair_style: str = "lj/cut 10.0"
    pair_coeff: List[str] = field(default_factory=list)
    masses: Dict[int, float] = field(default_factory=dict)  # type-id -> amu
    extra_commands: List[str] = field(default_factory=list)
    requires_charges: bool = False


class PotentialProvider(ABC):
    """Base class for all potentials."""

    kind: str = "custom"
    engines: Set[str] = set()
    #: human description shown in capability listings
    description: str = ""

    def __init__(self, spec: Optional["PotentialSpec"] = None):
        self.spec = spec

    # -- availability & applicability ---------------------------------
    def available(self) -> bool:
        """Whether the backing library/data is importable/present."""
        return True

    @abstractmethod
    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        """Whether this potential can model the given chemistry."""

    # -- engine hooks (implement only what applies) -------------------
    def ase_calculator(self, atoms: Any) -> Any:
        raise PotentialNotSupported(f"{self.kind} has no ASE calculator.")

    def lammps_potential(self, job: "ResolvedJob") -> LammpsPotential:
        raise PotentialNotSupported(f"{self.kind} has no LAMMPS mapping.")

    def openmm_system(self, job: "ResolvedJob") -> Any:
        raise PotentialNotSupported(f"{self.kind} has no OpenMM mapping.")

    # -- metadata -----------------------------------------------------
    def warnings_for(self, elements: Set[str]) -> List[str]:
        return []
