"""Load a system from a user-provided structure file (XYZ/CIF/POSCAR/PDB/...)."""

from __future__ import annotations

from typing import Optional

from .base import BuiltSystem, StructureBuilder


class FileLoaderBuilder(StructureBuilder):
    name = "file"

    def can_build(self, system) -> bool:
        return bool(system.structure_file) or system.kind == "file"

    def build(self, system, *, mp_api_key: Optional[str] = None) -> BuiltSystem:
        from ..structure_builder import infer_material_label, load_structure_file

        if not system.structure_file:
            raise ValueError(
                "structure_source is 'file' but no structure_file was provided. "
                "Upload an XYZ/CIF/POSCAR/PDB, or choose a different source."
            )
        atoms = load_structure_file(system.structure_file)
        if system.supercell:
            atoms = atoms * tuple(system.supercell)
        return BuiltSystem(atoms=atoms, topology=None, label=infer_material_label(atoms))
