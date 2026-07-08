"""Declarative, engine-agnostic simulation specification (v2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SystemSpec:
    """What to simulate. ``kind='auto'`` lets the builder registry decide."""

    kind: str = "auto"  # auto|crystal|molecule|solvent|polymer|alloy|file|material_project
    material: Optional[str] = None
    structure_file: Optional[str] = None
    mp_material_id: Optional[str] = None
    smiles: Optional[str] = None
    elements: Optional[List[str]] = None
    fractions: Optional[List[float]] = None
    supercell: Optional[Tuple[int, int, int]] = None
    target_atoms: int = 0
    n_molecules: int = 0
    density: Optional[float] = None  # g/cm^3, for packed molecular boxes
    monomer: Optional[str] = None  # SMILES (all-atom) or bead label (CG)
    chain_length: int = 0
    n_chains: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PotentialSpec:
    """Interatomic model. ``kind='auto'`` picks the best available provider."""

    kind: str = "auto"  # auto|emt|lj|eam|tersoff|meam|reaxff|mace|chgnet|m3gnet|opls|gaff|openff|custom
    name: Optional[str] = None
    file: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleSpec:
    """Thermodynamic ensemble and coupling. ``'auto'`` defers to the engine."""

    name: str = "NVT"  # NVE|NVT|NPT
    temperature: float = 300.0  # K
    pressure: float = 1.0  # atm
    thermostat: str = "auto"  # auto|langevin|berendsen|nose-hoover|none
    barostat: str = "auto"  # auto|berendsen|parrinello-rahman|nose-hoover|none
    tdamp: Optional[float] = None  # ps
    pdamp: Optional[float] = None  # ps


@dataclass
class ProtocolSpec:
    """Simulation method. ``params`` carry method-specific knobs."""

    name: str = "equilibrium"  # equilibrium|nemd|msst|deformation
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec:
    """Integration and output controls."""

    n_steps: int = 10_000
    timestep: float = 0.001  # ps
    equilibration_steps: Optional[int] = None
    output_frequency: int = 100
    seed: int = 12345


@dataclass
class SimulationSpec:
    """Full, engine-agnostic description of a molecular dynamics run."""

    system: SystemSpec = field(default_factory=SystemSpec)
    potential: PotentialSpec = field(default_factory=PotentialSpec)
    ensemble: EnsembleSpec = field(default_factory=EnsembleSpec)
    protocol: ProtocolSpec = field(default_factory=ProtocolSpec)
    run: RunSpec = field(default_factory=RunSpec)
    engine: str = "auto"  # auto|ase|lammps|openmm
    description: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        s, e, p, r = self.system, self.ensemble, self.protocol, self.run
        label = s.material or s.smiles or s.structure_file or s.mp_material_id or "system"
        parts = [
            label,
            f"{e.name} @ {e.temperature:g} K",
            f"protocol={p.name}",
            f"{r.n_steps:,} steps",
            f"dt={r.timestep} ps",
            f"engine={self.engine}",
            f"potential={self.potential.kind}",
        ]
        if e.name.upper() == "NPT":
            parts.insert(2, f"P={e.pressure:g} atm")
        if s.supercell:
            parts.append(f"supercell={tuple(s.supercell)}")
        if s.elements and s.fractions:
            comp = ",".join(f"{el}{fr:g}" for el, fr in zip(s.elements, s.fractions))
            parts.append(f"alloy[{comp}]")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_kwargs(cls, **kw: Any) -> "SimulationSpec":
        """Build a v2 spec from the historical ``run_simple_simulation`` kwargs.

        Keeps the old CLI/GUI/parser paths working while everything routes
        through the new orchestrator.
        """
        elements = kw.get("alloy_elements")
        fractions = kw.get("alloy_fractions")
        supercell = kw.get("supercell_reps")
        system = SystemSpec(
            kind="auto",
            material=kw.get("material"),
            structure_file=kw.get("structure_file"),
            mp_material_id=kw.get("mp_material_id"),
            elements=list(elements) if elements else None,
            fractions=list(fractions) if fractions else None,
            supercell=tuple(supercell) if supercell else None,
            target_atoms=int(kw.get("target_atoms") or 0),
        )
        source = (kw.get("structure_source") or "generate").lower()
        if kw.get("structure_file") or source in {"file", "upload", "user", "custom"}:
            system.kind = "file"
        elif kw.get("mp_material_id") or source in {"material_project", "materials_project", "mp"}:
            system.kind = "material_project"
        elif elements and len(elements) >= 2:
            system.kind = "alloy"

        ff = kw.get("force_field")
        potential = PotentialSpec(kind=(ff or "auto").lower() if ff else "auto", name=ff)

        ensemble = EnsembleSpec(
            name=(kw.get("ensemble") or "NVT").upper(),
            temperature=float(kw.get("temperature") or 300.0),
            pressure=float(kw.get("pressure") or 1.0),
            thermostat=(kw.get("thermostat") or "auto").lower(),
        )

        run = RunSpec(
            n_steps=int(kw.get("n_steps") or 10_000),
            timestep=float(kw["timestep"]) if kw.get("timestep") else 0.001,
            output_frequency=int(kw.get("output_frequency") or 100),
        )

        protocol = ProtocolSpec(name=(kw.get("protocol") or "equilibrium"))
        if isinstance(kw.get("protocol_params"), dict):
            protocol.params = dict(kw["protocol_params"])

        return cls(
            system=system,
            potential=potential,
            ensemble=ensemble,
            protocol=protocol,
            run=run,
            engine=(kw.get("engine") or "auto").lower(),
            description=str(kw.get("description") or ""),
        )
