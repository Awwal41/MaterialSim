"""Interatomic potential selection and parameter tables for ASE MD."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ase import Atoms
from ase.calculators.emt import EMT, parameters as EMT_PARAMETERS
from ase.calculators.lj import LennardJones

logger = logging.getLogger(__name__)

EMT_ELEMENTS = set(EMT_PARAMETERS.keys())

# Elements where generic LJ is a poor model (covalent / directional bonding).
COVALENT_ELEMENTS = frozenset({"Si", "C", "Ge", "Sn"})

# Lennard-Jones (epsilon eV, sigma Å) tuned toward equilibrium spacing where known.
_LJ_BY_ELEMENT = {
    "Si": (0.05, 2.35),
    "C": (0.05, 2.0),
    "Ge": (0.05, 2.45),
    "Cu": (0.015, 2.55),
    "Al": (0.012, 2.65),
    "Fe": (0.015, 2.45),
    "Ni": (0.015, 2.50),
    "Au": (0.015, 2.90),
    "Ag": (0.015, 2.90),
    "Pt": (0.015, 2.80),
    "Pd": (0.015, 2.75),
}
_LJ_DEFAULT = (0.0103, 3.4)
_LJ_RC = 8.5


def lj_parameters_for_material(material: str, symbols: Optional[set[str]] = None) -> Tuple[float, float]:
    """Return (epsilon, sigma) for *material* or the dominant element in *symbols*."""
    sym = material.strip().title() if len(material) <= 2 else material
    if sym in _LJ_BY_ELEMENT:
        return _LJ_BY_ELEMENT[sym]
    if symbols:
        for element in sorted(symbols):
            if element in _LJ_BY_ELEMENT:
                return _LJ_BY_ELEMENT[element]
    return _LJ_DEFAULT


def select_calculator(
    atoms: Atoms,
    force_field: str,
    material: str,
) -> Tuple[object, str, List[str]]:
    """Pick an ASE calculator and collect any reliability warnings."""
    warnings: List[str] = []
    symbols = set(atoms.get_chemical_symbols())
    ff = (force_field or "").lower()
    mat = material.strip() or (next(iter(symbols), "") if symbols else "")

    if ff in {"lj", "lennard-jones", "lennardjones"}:
        eps, sig = lj_parameters_for_material(mat, symbols)
        return LennardJones(epsilon=eps, sigma=sig, rc=_LJ_RC), "lennard-jones", warnings

    if symbols.issubset(EMT_ELEMENTS):
        return EMT(), "emt", warnings

    eps, sig = lj_parameters_for_material(mat, symbols)
    logger.warning(
        "No dedicated ASE potential for %s; using element-tuned Lennard-Jones.",
        "".join(sorted(symbols)),
    )
    warnings.append(
        f"The requested potential ({force_field or 'auto'}) is not available in ASE "
        f"for {mat}. A simplified Lennard-Jones model was used instead."
    )
    if mat in COVALENT_ELEMENTS or symbols & COVALENT_ELEMENTS:
        warnings.append(
            f"{mat} has directional covalent bonding; Lennard-Jones cannot capture "
            "diamond/zinc-blende physics. Temperature and pressure may not equilibrate "
            "reliably. Prefer EMT-supported metals (Cu, Al, Ni, …) for production-quality runs."
        )
    return LennardJones(epsilon=eps, sigma=sig, rc=_LJ_RC), "lennard-jones (fallback)", warnings


def recommended_timestep_ps(material: str, force_field_label: str) -> float:
    """Conservative timestep when the potential may be stiff or approximate."""
    ff = force_field_label.lower()
    if "lj" in ff and material in COVALENT_ELEMENTS:
        return 0.0005  # 0.5 fs
    return 0.001


def recommended_equilibration_fraction(force_field_label: str, material: str) -> float:
    """Fraction of total steps to spend equilibrating before production."""
    ff = force_field_label.lower()
    if "lj" in ff and material in COVALENT_ELEMENTS:
        return 0.35
    if "lj" in ff:
        return 0.25
    return 0.15
