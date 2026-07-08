"""Molecular systems from SMILES (RDKit) or named molecules (ASE).

Produces an ASE ``Atoms`` plus a bonded :class:`Topology` when built from
SMILES, so downstream bonded force fields (OpenMM) have connectivity.
"""

from __future__ import annotations

from typing import Optional

from .base import BuiltSystem, StructureBuilder
from .topology import Topology

_NAMED = {
    "water": "H2O", "methane": "CH4", "ammonia": "NH3", "benzene": "C6H6",
    "ethanol": "C2H6O", "methanol": "CH4O", "co2": "CO2", "acetone": "C3H6O",
}


class MoleculeBuilder(StructureBuilder):
    name = "molecule"

    def can_build(self, system) -> bool:
        if system.kind == "molecule":
            return True
        if system.smiles and not system.monomer and system.chain_length <= 1:
            return True
        if system.kind == "auto" and system.material and system.material.lower() in _NAMED:
            return True
        return False

    def build(self, system, *, mp_api_key: Optional[str] = None) -> BuiltSystem:
        warnings = []
        if system.smiles:
            atoms, topo = _from_smiles(system.smiles)
            label = system.smiles
        else:
            name = (system.material or "").lower()
            formula = _NAMED.get(name, system.material)
            atoms, topo = _from_named(formula)
            label = formula

        box = float(system.extras.get("box", 15.0))
        atoms.set_cell([box, box, box])
        atoms.center()
        atoms.pbc = system.extras.get("pbc", False)

        if system.n_molecules and system.n_molecules > 1:
            warnings.append(
                "Multiple-molecule packing (n_molecules>1) needs packmol; built a "
                "single molecule. Install packmol for solvated/bulk molecular boxes."
            )
        return BuiltSystem(atoms=atoms, topology=topo, label=label, warnings=warnings)


def _from_smiles(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Building from SMILES requires RDKit (`pip install rdkit`). ({exc})"
        )
    import numpy as np
    from ase import Atoms

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES '{smiles}'.")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise ValueError(f"Could not generate 3D coordinates for '{smiles}'.")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atoms = Atoms(symbols=symbols, positions=positions)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return atoms, Topology(bonds=bonds, atom_types=symbols)


def _from_named(formula: str):
    from ase.build import molecule

    try:
        atoms = molecule(formula)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Unknown molecule '{formula}'. Provide a SMILES string instead."
        ) from exc
    return atoms, None
