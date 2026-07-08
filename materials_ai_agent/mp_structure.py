"""Fetch crystal structures from the Materials Project (optional dependency)."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from ase import Atoms

logger = logging.getLogger(__name__)

_MP_ID_RE = re.compile(r"^mp-\d+$", re.I)


def get_mp_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return an MP API key from *explicit* or the environment."""
    return explicit or os.getenv("MP_API_KEY") or os.getenv("MATERIALS_PROJECT_API_KEY")


def fetch_mp_structure(
    formula_or_id: str,
    *,
    api_key: Optional[str] = None,
    material_id: Optional[str] = None,
) -> Atoms:
    """Load the lowest-energy-per-atom MP structure for a formula or mp-id.

    Requires ``mp-api`` and ``pymatgen``. Raises :class:`ImportError` or
    :class:`RuntimeError` when MP is unavailable or no match is found.
    """
    key = get_mp_api_key(api_key)
    if not key:
        raise RuntimeError(
            "Materials Project API key not configured. Set MP_API_KEY in the environment."
        )

    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError(
            "Materials Project structure fetch requires 'mp-api' and 'pymatgen'."
        ) from exc

    target_id = material_id or (
        formula_or_id if _MP_ID_RE.match(formula_or_id.strip()) else None
    )

    with MPRester(key) as mpr:
        if target_id:
            structure = mpr.get_structure_by_material_id(target_id)
            mp_id = target_id
        else:
            docs = mpr.materials.summary.search(
                formula=formula_or_id.strip(),
                fields=["material_id", "structure", "energy_per_atom"],
            )
            if not docs:
                raise RuntimeError(
                    f"No Materials Project entry found for formula '{formula_or_id}'."
                )
            docs = sorted(
                docs,
                key=lambda d: getattr(d, "energy_per_atom", float("inf")) or float("inf"),
            )
            doc = docs[0]
            structure = doc.structure
            mp_id = doc.material_id

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.info["mp_material_id"] = mp_id
    atoms.info["structure_source"] = "material_project"
    logger.info("Loaded MP structure %s (%d atoms)", mp_id, len(atoms))
    return atoms


def mp_available(api_key: Optional[str] = None) -> bool:
    """Return True when MP packages and an API key are available."""
    if not get_mp_api_key(api_key):
        return False
    try:
        import mp_api  # noqa: F401
        import pymatgen  # noqa: F401
    except ImportError:
        return False
    return True
