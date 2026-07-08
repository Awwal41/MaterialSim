"""Turn a natural-language instruction into a :class:`SimulationSpec` (v2).

Deterministic and offline: it reuses the legacy field parser for the common
knobs (material, T, P, steps, ensemble, ...) and layers on detection of engine,
potential kind, protocol (NEMD/MSST/deformation), and molecular/polymer systems.
It never invents a material; an unresolved system flows through to the validator
which asks the user to clarify.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .simulation_spec import SimulationSpec

_POTENTIAL_KEYWORDS = {
    "mace": "mace", "chgnet": "chgnet", "m3gnet": "m3gnet",
    "eam": "eam", "tersoff": "tersoff", "meam": "meam",
    "reaxff": "reaxff", "reax": "reaxff",
    "opls": "opls", "gaff": "gaff", "openff": "openff", "sage": "openff",
    "lennard-jones": "lj", "lennard jones": "lj", " lj ": "lj",
    "emt": "emt", "mlip": "mace", "machine-learned": "mace", "machine learned": "mace",
}
_ENGINES = {"lammps": "lammps", "openmm": "openmm", "ase": "ase"}


def _detect_potential(text: str) -> Optional[str]:
    low = f" {text.lower()} "
    for kw, kind in _POTENTIAL_KEYWORDS.items():
        if kw in low:
            return kind
    return None


def _detect_engine(text: str) -> Optional[str]:
    low = text.lower()
    for kw, eng in _ENGINES.items():
        if kw in low:
            return eng
    return None


def _detect_protocol(text: str) -> Dict[str, Any]:
    low = text.lower()
    if any(k in low for k in ("msst", "shock", "multi-scale shock", "hugoniot")):
        params: Dict[str, Any] = {}
        m = re.search(r"(\d+(?:\.\d+)?)\s*km/s", low)
        if m:
            params["shock_velocity_kms"] = float(m.group(1))
        m = re.search(r"\b(shock\s+)?(?:along|in)\s+([xyz])\b", low)
        if m:
            params["direction"] = m.group(2)
        return {"name": "msst", "params": params}

    if any(k in low for k in ("nemd", "non-equilibrium", "thermal conductivity", "viscosity", "heat flux")):
        params = {"mode": "shear" if "viscosity" in low or "shear" in low else "thermal"}
        m = re.search(r"shear\s*rate\s*(\d+(?:\.\d+e?-?\d*)?)", low)
        if m:
            params["shear_rate"] = float(m.group(1))
        return {"name": "nemd", "params": params}

    if any(k in low for k in ("deformation", "tensile", "compress", "strain", "stress-strain", "stress strain")):
        params = {"mode": "compressive" if "compress" in low else "tensile"}
        m = re.search(r"strain\s*rate\s*(\d+(?:\.\d+e?-?\d*)?)", low)
        if m:
            params["strain_rate"] = float(m.group(1))
        m = re.search(r"\balong\s+([xyz])\b", low)
        if m:
            params["axis"] = m.group(1)
        return {"name": "deformation", "params": params}

    return {"name": "equilibrium", "params": {}}


def _detect_smiles(text: str) -> Optional[str]:
    m = re.search(r"smiles[:\s]+([^\s,;]+)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"monomer[:\s]+([^\s,;]+)", text, re.I)
    if m:
        return m.group(1)
    return None


def _detect_polymer(text: str) -> Dict[str, Any]:
    low = text.lower()
    out: Dict[str, Any] = {}
    if "polymer" in low or "chain" in low or "bead-spring" in low or "bead spring" in low:
        out["is_polymer"] = True
    m = re.search(r"(?:chain length|degree of polymerization|dp|n\s*=)\s*(\d+)", low)
    if m:
        out["chain_length"] = int(m.group(1))
    m = re.search(r"(\d+)\s*chains", low)
    if m:
        out["n_chains"] = int(m.group(1))
    if "coarse-grained" in low or "coarse grained" in low or "bead" in low:
        out["coarse_grained"] = True
    return out


def extract_spec(instruction: str, config=None) -> SimulationSpec:
    """Build a v2 SimulationSpec from free text."""
    from ..simulation_parser import parse_simulation_instruction

    legacy = parse_simulation_instruction(instruction, config)
    spec = SimulationSpec.from_legacy_kwargs(**legacy.to_run_kwargs())
    spec.description = instruction.strip()[:500]

    # Potential / engine overrides.
    pot = _detect_potential(instruction)
    if pot:
        spec.potential.kind = pot
    eng = _detect_engine(instruction)
    if eng:
        spec.engine = eng

    # Protocol.
    proto = _detect_protocol(instruction)
    spec.protocol.name = proto["name"]
    spec.protocol.params = proto["params"]

    # Molecular / polymer systems.
    smiles = _detect_smiles(instruction)
    poly = _detect_polymer(instruction)
    if poly.get("is_polymer"):
        spec.system.kind = "polymer"
        if smiles:
            spec.system.monomer = smiles
        if poly.get("chain_length"):
            spec.system.chain_length = poly["chain_length"]
        if poly.get("n_chains"):
            spec.system.n_chains = poly["n_chains"]
        if poly.get("coarse_grained"):
            spec.system.extras["coarse_grained"] = True
    elif smiles:
        spec.system.kind = "molecule"
        spec.system.smiles = smiles

    return spec
