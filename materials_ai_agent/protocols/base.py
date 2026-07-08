"""Protocol interface: a simulation *method* independent of chemistry.

Protocols declare which engines can run them and provide engine-specific hooks.
The ASE hook drives ASE integrators directly; the LAMMPS hook emits fix/compute
lines for the generated input script. Each protocol owns its post-processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from ..engines.base import ResolvedJob


class ProtocolNotSupported(Exception):
    """Raised when a protocol cannot run on the requested engine."""


class Protocol(ABC):
    name: str = "protocol"
    engines: Set[str] = set()
    description: str = ""

    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = dict(params or {})

    # -- ASE hook -----------------------------------------------------
    def ase_run(self, job: "ResolvedJob", context: Any) -> Dict[str, Any]:
        """Drive an ASE run. ``context`` exposes integrator/logging helpers."""
        raise ProtocolNotSupported(f"{self.name} is not implemented for ASE.")

    # -- LAMMPS hook --------------------------------------------------
    def lammps_blocks(self, job: "ResolvedJob") -> List[str]:
        """Return LAMMPS command blocks (fixes/computes/runs) for this protocol."""
        raise ProtocolNotSupported(f"{self.name} is not implemented for LAMMPS.")

    # -- OpenMM hook --------------------------------------------------
    def openmm_run(self, job: "ResolvedJob", context: Any) -> Dict[str, Any]:
        raise ProtocolNotSupported(f"{self.name} is not implemented for OpenMM.")

    # -- shared post-processing --------------------------------------
    def postprocess(self, workdir: Path) -> Dict[str, Any]:
        """Optional protocol-specific analysis. Default: nothing extra."""
        return {}
