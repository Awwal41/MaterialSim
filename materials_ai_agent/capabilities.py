"""Capability matrix and spec validation.

Single arbiter of what can actually run. Given a :class:`SimulationSpec`, it
resolves an engine, checks the ensemble/thermostat/protocol/potential against
that engine's declared capabilities, and returns explicit problems instead of
letting the system silently do the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import bootstrap
from .engines.registry import available_engines, get_engine, list_engines
from .protocols.registry import list_protocols
from .spec import SimulationSpec


@dataclass
class ValidationResult:
    ok: bool
    engine: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)


_UNRESOLVED = {"", "unresolved", "unknown", "custom", "user", "uploaded", None}


def _system_has_structure(spec: SimulationSpec) -> bool:
    s = spec.system
    if s.structure_file or s.mp_material_id or s.smiles or s.monomer:
        return True
    if s.elements and len(s.elements) >= 1:
        return True
    return s.material not in _UNRESOLVED


def choose_engine(spec: SimulationSpec) -> Optional[str]:
    """Pick an engine for ``engine='auto'`` based on availability + needs."""
    bootstrap.ensure()
    if spec.engine and spec.engine != "auto":
        return spec.engine

    avail = available_engines()
    protocol = spec.protocol.name.lower()
    pot = spec.potential.kind.lower()

    # Method/potential-driven routing among available engines.
    preferred: List[str] = []
    if protocol in {"msst", "nemd", "deformation"}:
        preferred = ["lammps"]
    if pot in {"opls", "gaff", "openff"}:
        preferred = ["openmm", *preferred]
    if pot in {"eam", "tersoff", "meam", "reaxff"}:
        preferred = ["lammps", *preferred]
    if pot in {"mace", "chgnet", "m3gnet", "emt", "lj", "auto"} and not preferred:
        preferred = ["ase", "lammps", "openmm"]

    for name in preferred:
        if name in avail:
            return name
    # Fall back to any available engine, else ASE (may still be importable).
    if avail:
        return avail[0]
    return "ase" if "ase" in list_engines() else None


def validate(spec: SimulationSpec) -> ValidationResult:
    """Validate a spec against the capability matrix."""
    bootstrap.ensure()
    errors: List[str] = []
    warnings: List[str] = []
    clarifications: List[str] = []

    if not _system_has_structure(spec):
        clarifications.append(
            "I could not determine which material/system to simulate. "
            "Please specify a formula (e.g. 'Cu', 'Al2O3'), a SMILES string, "
            "a structure file, or a Materials Project id (mp-1234)."
        )

    engine_name = choose_engine(spec)
    if engine_name is None:
        errors.append("No MD engine is available. Install ASE, LAMMPS, or OpenMM.")
        return ValidationResult(False, None, errors, warnings, clarifications)

    try:
        engine = get_engine(engine_name)
        caps = engine.capabilities()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Engine '{engine_name}' could not be loaded: {exc}")
        return ValidationResult(False, engine_name, errors, warnings, clarifications)

    if not caps.available:
        errors.append(
            f"Engine '{engine_name}' is selected but not installed/available. "
            f"{caps.notes}".strip()
        )

    ensemble = spec.ensemble.name.upper()
    if caps.ensembles and ensemble not in caps.ensembles:
        errors.append(
            f"Engine '{engine_name}' does not support the {ensemble} ensemble "
            f"(supported: {sorted(caps.ensembles)})."
        )

    thermostat = (spec.ensemble.thermostat or "auto").lower()
    if thermostat not in {"auto", "none"} and caps.thermostats and thermostat not in caps.thermostats:
        warnings.append(
            f"Engine '{engine_name}' does not implement the '{thermostat}' thermostat "
            f"(has {sorted(caps.thermostats)}); it will report an error rather than "
            "silently substituting a different one."
        )

    protocol = spec.protocol.name.lower()
    if protocol not in list_protocols():
        errors.append(
            f"Unknown protocol '{protocol}'. Available: {list_protocols()}."
        )
    elif caps.protocols and protocol not in caps.protocols:
        errors.append(
            f"Protocol '{protocol}' is not supported by engine '{engine_name}' "
            f"(supported: {sorted(caps.protocols)}). Try a different engine."
        )

    ok = not errors and not clarifications
    return ValidationResult(ok, engine_name, errors, warnings, clarifications)
