"""Engine adapter interface shared by every MD backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:  # avoid import cycles / heavy imports at module load
    from ..potentials.base import PotentialProvider
    from ..protocols.base import Protocol
    from ..runners.base import Runner
    from ..spec import SimulationSpec
    from ..structures.topology import Topology


@dataclass
class EngineCapabilities:
    """Declares what an engine can do so the validator can reason about it."""

    name: str
    available: bool
    ensembles: Set[str] = field(default_factory=set)
    thermostats: Set[str] = field(default_factory=set)
    barostats: Set[str] = field(default_factory=set)
    protocols: Set[str] = field(default_factory=set)
    potential_kinds: Set[str] = field(default_factory=set)
    notes: str = ""


@dataclass
class ResolvedJob:
    """Everything an engine needs to run, produced by the orchestrator."""

    spec: "SimulationSpec"
    atoms: Any  # ase.Atoms
    topology: Optional["Topology"]
    potential: "PotentialProvider"
    protocol: "Protocol"
    workdir: Path
    runner: "Runner"
    material_label: str = "system"
    warnings: List[str] = field(default_factory=list)


@dataclass
class EngineResult:
    """Normalized result contract returned by every engine adapter."""

    success: bool
    engine: str
    workdir: str
    message: str = ""
    n_atoms: int = 0
    n_frames: int = 0
    production_start_step: Optional[int] = None
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "success": self.success,
            "engine": self.engine,
            "simulation_directory": self.workdir,
            "message": self.message,
            "n_atoms": self.n_atoms,
            "n_frames": self.n_frames,
            "production_start_step": self.production_start_step,
            "output_files": self.output_files,
            "warnings": self.warnings,
        }
        d.update(self.extra)
        if self.error:
            d["error"] = self.error
        return d


class EngineAdapter(ABC):
    """Base class for all MD engines."""

    name: str = "engine"

    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return static + availability info for this engine."""

    def available(self) -> bool:
        return self.capabilities().available

    @abstractmethod
    def run(self, job: ResolvedJob) -> EngineResult:
        """Prepare inputs, execute, and return a normalized result."""
