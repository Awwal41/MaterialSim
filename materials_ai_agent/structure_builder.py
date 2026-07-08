"""Build atomic structures for simple and complex MD setups."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read

from .core.materials_database import MaterialsDatabase

logger = logging.getLogger(__name__)

# Common chemical formulas not in the JSON database.
_COMPOUND_LATTICE = {
    "Al2O3": ("corundum", 4.76),
    "SiO2": ("quartz", 4.91),
    "TiO2": ("rutile", 4.59),
    "MgO": ("rocksalt", 4.21),
    "Fe2O3": ("corundum", 5.03),
}


def load_structure_file(path: str | Path) -> Atoms:
    """Load a structure from XYZ, CIF, POSCAR, etc."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    atoms = read(str(p))
    if not isinstance(atoms, Atoms):
        raise ValueError(f"Could not read a single structure from {path}")
    if atoms.cell.rank == 3:
        atoms.pbc = True
    return atoms


def build_atoms(
    material: str,
    *,
    structure_source: str = "generate",
    structure_file: Optional[str] = None,
    supercell_reps: Optional[Tuple[int, int, int]] = None,
    target_atoms: int = 64,
    alloy_elements: Optional[List[str]] = None,
    alloy_fractions: Optional[List[float]] = None,
) -> Atoms:
    """Build or load an :class:`ase.Atoms` object for *material*."""
    if structure_source == "file" or structure_file:
        if not structure_file:
            raise ValueError("structure_file is required when structure_source='file'.")
        atoms = load_structure_file(structure_file)
        if supercell_reps:
            atoms = atoms * supercell_reps
        return atoms

    if alloy_elements and len(alloy_elements) >= 2:
        return _build_alloy(
            alloy_elements,
            alloy_fractions or _equal_fractions(len(alloy_elements)),
            supercell_reps=supercell_reps,
            target_atoms=target_atoms,
        )

    formula = material.strip()
    db = MaterialsDatabase()
    props = db.get_material(formula)

    if props is not None:
        atoms = _from_database(formula, props)
    elif formula in _COMPOUND_LATTICE:
        atoms = _build_compound(formula)
    else:
        atoms = _build_fallback(formula)

    if supercell_reps:
        atoms = atoms * supercell_reps
    else:
        atoms = _resize_supercell(atoms, target_atoms)
    return atoms


def _equal_fractions(n: int) -> List[float]:
    return [1.0 / n] * n


def _build_alloy(
    elements: List[str],
    fractions: List[float],
    *,
    supercell_reps: Optional[Tuple[int, int, int]] = None,
    target_atoms: int = 128,
) -> Atoms:
    """Random substitutional alloy on an FCC lattice (metals)."""
    primary = elements[0]
    try:
        cell = bulk(primary, "fcc", a=_guess_lattice(primary), cubic=True)
    except Exception:
        cell = bulk(primary, "sc", a=3.6)

    reps = supercell_reps or _reps_for_target(len(cell), target_atoms)
    supercell = cell * reps
    n_atoms = len(supercell)

    fracs = np.array(fractions, dtype=float)
    fracs = fracs / fracs.sum()
    cum = np.concatenate([[0.0], np.cumsum(fracs)])
    rng = np.random.default_rng(42)
    u = rng.random(n_atoms)
    idx = np.searchsorted(cum, u, side="right") - 1
    idx = np.clip(idx, 0, len(elements) - 1)
    symbols = [elements[i] for i in idx]
    supercell.set_chemical_symbols(symbols)
    supercell.pbc = True
    return supercell


def _guess_lattice(symbol: str) -> float:
    guesses = {"Cu": 3.61, "Al": 4.05, "Ni": 3.52, "Fe": 2.87, "Au": 4.08, "Ag": 4.09}
    return guesses.get(symbol, 3.6)


def _from_database(formula: str, props) -> Atoms:
    if props.lattice_type == "molecular":
        return _molecular_box(formula)
    lattice = props.lattice_type
    a = props.lattice_parameter
    builders = {
        "diamond": lambda: bulk(formula, "diamond", a=a),
        "fcc": lambda: bulk(formula, "fcc", a=a),
        "bcc": lambda: bulk(formula, "bcc", a=a),
        "hcp": lambda: bulk(formula, "hcp", a=a),
        "zincblende": lambda: bulk(formula, "zincblende", a=a),
    }
    try:
        cell = builders.get(lattice, lambda: bulk(formula, "sc", a=a or 3.0))()
    except Exception:
        cell = bulk(formula, "sc", a=a or 3.0)
    return cell


def _build_compound(formula: str) -> Atoms:
    struct, a = _COMPOUND_LATTICE[formula]
    return bulk(formula, struct, a=a)


def _build_fallback(formula: str) -> Atoms:
    if formula.upper() == "H2O":
        return _molecular_box("H2O")
    # Try pymatgen-style formula parsing via ASE bulk.
    for crystal in ("fcc", "bcc", "diamond", "sc"):
        try:
            return bulk(formula, crystal, a=3.6)
        except Exception:
            continue
    raise ValueError(
        f"Could not build a structure for '{formula}'. "
        "Provide a structure file or a supported formula (e.g. Cu, Al2O3, CuNi alloy)."
    )


def _resize_supercell(cell: Atoms, target_atoms: int) -> Atoms:
    n_unit = max(1, len(cell))
    reps = max(1, int(round((target_atoms / n_unit) ** (1.0 / 3.0))))
    out = cell * (reps, reps, reps)
    out.pbc = True
    return out


def _reps_for_target(n_unit: int, target_atoms: int) -> Tuple[int, int, int]:
    reps = max(1, int(round((target_atoms / n_unit) ** (1.0 / 3.0))))
    return (reps, reps, reps)


def _molecular_box(formula: str, box: float = 12.0) -> Atoms:
    mol = molecule(formula)
    mol.set_cell([box, box, box])
    mol.center()
    mol.pbc = False
    return mol


def parse_alloy_notation(text: str) -> Optional[Tuple[List[str], List[float]]]:
    """Parse 'CuNi 50-50', 'Cu0.8Ni0.2', 'Fe-Cr alloy'."""
    lower = text.lower()
    t = text.replace(" ", "")

    # Decimal fractions only — avoids misreading stoichiometric formulas (Al2O3).
    m = re.search(r"([A-Z][a-z]?)(0?\.\d+|[01]\.\d+)([A-Z][a-z]?)(0?\.\d+|[01]\.\d+)", t)
    if m:
        e1, f1, e2, f2 = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
        return [e1, e2], [f1, f2]

    m = re.search(r"([A-Z][a-z]?)\s*[-/]\s*([A-Z][a-z]?)", text, re.I)
    if m and ("alloy" in lower or "solid solution" in lower):
        return [m.group(1).title(), m.group(2).title()], [0.5, 0.5]

    m = re.search(r"\b([A-Z][a-z]?)([A-Z][a-z]?)\b", text)
    if m and m.group(1) != m.group(2) and "alloy" in lower:
        return [m.group(1), m.group(2)], [0.5, 0.5]
    return None
