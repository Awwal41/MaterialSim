"""Crystalline / metallic / alloy / Materials-Project systems.

Reuses the (de-hardcoded) builders in ``structure_builder`` which now derive
lattice constants from ASE/pymatgen reference data rather than magic numbers.
"""

from __future__ import annotations

from typing import Optional

from .base import BuiltSystem, StructureBuilder

_UNRESOLVED = {"", "unresolved", "unknown", "custom", "user", "uploaded", None}


class CrystalBuilder(StructureBuilder):
    name = "crystal"

    def can_build(self, system) -> bool:
        if system.kind in {"crystal", "alloy", "material_project"}:
            return True
        if system.mp_material_id:
            return True
        if system.elements and len(system.elements) >= 1:
            return True
        if system.kind == "auto" and system.material not in _UNRESOLVED and not system.smiles:
            return True
        return False

    def build(self, system, *, mp_api_key: Optional[str] = None) -> BuiltSystem:
        from ..structure_builder import build_atoms, infer_material_label

        source = "generate"
        if system.kind == "material_project" or system.mp_material_id:
            source = "material_project"

        atoms = build_atoms(
            system.material or (system.mp_material_id or ""),
            structure_source=source,
            mp_material_id=system.mp_material_id,
            supercell_reps=tuple(system.supercell) if system.supercell else None,
            target_atoms=system.target_atoms or 64,
            alloy_elements=system.elements,
            alloy_fractions=system.fractions,
            mp_api_key=mp_api_key,
        )
        label = system.material or infer_material_label(atoms)
        return BuiltSystem(atoms=atoms, topology=None, label=label)
