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
    "GaN": ("wurtzite", 3.19),
    "ZnO": ("wurtzite", 3.25),
    "NaCl": ("rocksalt", 5.64),
    "CaF2": ("fluorite", 5.46),
}

_FILE_SOURCE_ALIASES = frozenset({"file", "upload", "user", "custom"})
_MP_SOURCE_ALIASES = frozenset({"material_project", "materials_project", "mp", "materials project"})


def normalize_structure_source(source: str) -> str:
    """Map user-facing structure source labels to internal values."""
    key = (source or "generate").strip().lower().replace("-", "_")
    if key in _FILE_SOURCE_ALIASES:
        return "file"
    if key in _MP_SOURCE_ALIASES:
        return "material_project"
    return key or "generate"


def infer_material_label(atoms: Atoms) -> str:
    """Derive a human-readable formula from an :class:`ase.Atoms` object."""
    symbols = atoms.get_chemical_symbols()
    if not symbols:
        return "custom"
    counts: dict[str, int] = {}
    for sym in symbols:
        counts[sym] = counts.get(sym, 0) + 1
    try:
        from pymatgen.core import Composition

        return Composition(counts).reduced_formula
    except Exception:
        parts = [f"{sym}{counts[sym]}" if counts[sym] > 1 else sym for sym in sorted(counts)]
        return "".join(parts)


def load_structure_file(path: str | Path) -> Atoms:
    """Load a structure from XYZ, CIF, POSCAR, PDB, etc."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    atoms = read(str(p))
    if not isinstance(atoms, Atoms):
        raise ValueError(f"Could not read a single structure from {path}")
    if atoms.cell.rank == 3:
        atoms.pbc = True
    atoms.info.setdefault("structure_source", "file")
    atoms.info.setdefault("structure_file", str(p.resolve()))
    return atoms


def build_atoms(
    material: str,
    *,
    structure_source: str = "generate",
    structure_file: Optional[str] = None,
    mp_material_id: Optional[str] = None,
    supercell_reps: Optional[Tuple[int, int, int]] = None,
    target_atoms: int = 64,
    alloy_elements: Optional[List[str]] = None,
    alloy_fractions: Optional[List[float]] = None,
    mp_api_key: Optional[str] = None,
) -> Atoms:
    """Build or load an :class:`ase.Atoms` object for *material*."""
    source = normalize_structure_source(structure_source)

    if source == "file" or structure_file:
        if not structure_file:
            raise ValueError(
                "structure_file is required when using an uploaded or user-provided structure."
            )
        atoms = load_structure_file(structure_file)
        if supercell_reps:
            atoms = atoms * supercell_reps
        return atoms

    if source == "material_project" or mp_material_id:
        from .mp_structure import fetch_mp_structure

        atoms = fetch_mp_structure(
            mp_material_id or material,
            api_key=mp_api_key,
            material_id=mp_material_id,
        )
        if supercell_reps:
            atoms = atoms * supercell_reps
        elif target_atoms and len(atoms) < target_atoms:
            atoms = _resize_supercell(atoms, target_atoms)
        return atoms

    if alloy_elements and len(alloy_elements) >= 2:
        return _build_alloy(
            alloy_elements,
            alloy_fractions or _equal_fractions(len(alloy_elements)),
            supercell_reps=supercell_reps,
            target_atoms=target_atoms,
        )

    formula = (material or "").strip() or "custom"
    db = MaterialsDatabase()
    props = db.get_material(formula)

    if props is not None:
        atoms = _from_database(formula, props)
    elif formula in _COMPOUND_LATTICE:
        atoms = _build_compound(formula)
    else:
        atoms = _build_fallback(formula, mp_api_key=mp_api_key)

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
    """Best-effort cubic lattice constant (A) from ASE reference data.

    Uses ASE's tabulated experimental reference states when available, and
    otherwise estimates a nearest-neighbour spacing from covalent radii. This
    replaces the previous hardcoded 6-element table + ``3.6`` magic default so
    arbitrary elements get physically reasonable cells.
    """
    from ase.data import atomic_numbers, covalent_radii, reference_states

    z = atomic_numbers.get(symbol)
    if z is not None:
        ref = reference_states[z] if z < len(reference_states) else None
        if ref and ref.get("a"):
            return float(ref["a"])
        r = covalent_radii[z] if z < len(covalent_radii) else 0.0
        if r:
            # FCC nearest-neighbour distance d = a/sqrt(2); d ~= 2 * r.
            return float(2.0 * r * (2.0 ** 0.5))
    return 3.6


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


def _build_fallback(formula: str, *, mp_api_key: Optional[str] = None) -> Atoms:
    if formula.upper() == "H2O":
        return _molecular_box("H2O")

    if formula.lower() in {"custom", "user", "uploaded"}:
        raise ValueError(
            "No structure file or Materials Project ID was provided. "
            "Upload a structure file (XYZ, CIF, POSCAR) or specify an mp-id."
        )

    # Try pymatgen-informed ASE builders for arbitrary stoichiometries.
    atoms = _try_pymatgen_bulk(formula)
    if atoms is not None:
        return atoms

    # Single-element fallback: use the element's real reference structure and
    # lattice constant rather than a fixed a=3.6 A cell.
    from ase.data import atomic_numbers, reference_states

    if formula in atomic_numbers:
        z = atomic_numbers[formula]
        ref = reference_states[z] if z < len(reference_states) else None
        sym = (ref or {}).get("symmetry")
        a = _guess_lattice(formula)
        prototypes = [sym] if sym in {"fcc", "bcc", "hcp", "diamond", "sc"} else []
        prototypes += ["fcc", "bcc", "sc"]
        for crystal in prototypes:
            try:
                return bulk(formula, crystal, a=a, cubic=(crystal != "hcp"))
            except Exception:
                continue

    # Last resort: fetch from Materials Project when configured.
    from .mp_structure import fetch_mp_structure, mp_available

    if mp_available(mp_api_key):
        try:
            return fetch_mp_structure(formula, api_key=mp_api_key)
        except Exception as exc:
            logger.warning("MP fallback failed for %s: %s", formula, exc)

    raise ValueError(
        f"Could not build a structure for '{formula}'. "
        "Provide a structure file, an mp-id (e.g. mp-1234), or a supported formula."
    )


def _try_pymatgen_bulk(formula: str) -> Optional[Atoms]:
    """Use pymatgen composition hints to pick a reasonable ASE bulk prototype."""
    try:
        from pymatgen.core import Composition
    except ImportError:
        return None

    try:
        comp = Composition(formula)
    except Exception:
        return None

    elements = [str(el) for el in comp.elements]
    if len(elements) == 1:
        el = elements[0]
        for crystal in ("fcc", "bcc", "diamond", "hcp", "sc"):
            try:
                return bulk(el, crystal, a=_guess_lattice(el), cubic=True)
            except Exception:
                continue
        return None

    if len(elements) == 2:
        a_el, b_el = elements[0], elements[1]
        ratio = comp.get_atomic_fraction(a_el) / max(comp.get_atomic_fraction(b_el), 1e-9)
        prototypes = (
            ["zincblende", "rocksalt", "wurtzite", "cesiumchloride", "fluorite"]
            if ratio <= 1.5
            else ["rocksalt", "fluorite", "cesiumchloride"]
        )
        a_guess = _estimate_binary_lattice(a_el, b_el)
        for crystal in prototypes:
            try:
                return bulk(f"{a_el}{b_el}", crystal, a=a_guess, cubic=True)
            except Exception:
                try:
                    return bulk(comp.reduced_formula, crystal, a=a_guess, cubic=True)
                except Exception:
                    continue
    return None


def _estimate_binary_lattice(a_el: str, b_el: str) -> float:
    """Estimate a cubic lattice constant for a binary from covalent radii."""
    from ase.data import atomic_numbers, covalent_radii

    try:
        ra = covalent_radii[atomic_numbers[a_el]]
        rb = covalent_radii[atomic_numbers[b_el]]
    except (KeyError, IndexError):
        return 4.5
    # Rocksalt-like: a ~= 2 * (r_a + r_b); clamp to a sane range.
    return float(max(3.0, min(2.0 * (ra + rb), 8.0)))


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
