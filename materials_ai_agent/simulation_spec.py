"""Structured specification for molecular dynamics runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimulationSpec:
    """Full parameter set for a (possibly complex) MD simulation."""

    material: str
    temperature: float = 300.0
    pressure: float = 1.0  # atm
    n_steps: int = 10_000
    timestep: float = 0.001  # ps
    ensemble: str = "NVT"
    thermostat: str = "Langevin"
    force_field: Optional[str] = None
    structure_source: str = "generate"  # generate | file | upload | material_project
    structure_file: Optional[str] = None
    mp_material_id: Optional[str] = None
    supercell_reps: Optional[tuple[int, int, int]] = None
    target_atoms: int = 64
    output_frequency: int = 100
    alloy_elements: Optional[List[str]] = None
    alloy_fractions: Optional[List[float]] = None
    description: str = ""

    def to_run_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for :func:`run_simple_simulation`."""
        d = asdict(self)
        d.pop("description", None)
        return d

    def summary(self) -> str:
        parts = [
            f"{self.material}",
            f"{self.ensemble} @ {self.temperature:g} K",
            f"{self.n_steps:,} steps",
            f"dt={self.timestep} ps",
        ]
        if self.ensemble.upper() == "NPT":
            parts.append(f"P={self.pressure:g} atm")
        if self.alloy_elements:
            parts.append(f"alloy {''.join(self.alloy_elements)}")
        if self.structure_file:
            parts.append(f"structure={self.structure_file}")
        if self.mp_material_id:
            parts.append(f"mp-id={self.mp_material_id}")
        if self.structure_source not in ("generate", ""):
            parts.append(f"source={self.structure_source}")
        if self.supercell_reps:
            parts.append(f"supercell={self.supercell_reps}")
        return ", ".join(parts)
