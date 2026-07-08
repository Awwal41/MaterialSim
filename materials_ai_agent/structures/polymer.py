"""Polymer builders: coarse-grained bead-spring and all-atom from a monomer.

- Coarse-grained (default when ``monomer`` is a bead label or missing SMILES):
  a Kremer-Grest style linear chain of beads with harmonic bonds. Engine-neutral
  positions + :class:`Topology`.
- All-atom: replicate a monomer SMILES ``chain_length`` times into a linear chain
  (approximate geometry) with connectivity for a bonded force field (OpenMM).
"""

from __future__ import annotations

from typing import Optional

from .base import BuiltSystem, StructureBuilder
from .topology import Topology


class PolymerBuilder(StructureBuilder):
    name = "polymer"

    def can_build(self, system) -> bool:
        if system.kind == "polymer":
            return True
        return bool(system.monomer) or system.chain_length > 1

    def build(self, system, *, mp_api_key: Optional[str] = None) -> BuiltSystem:
        chain_length = max(2, int(system.chain_length or 10))
        n_chains = max(1, int(system.n_chains or 1))
        is_cg = bool(system.extras.get("coarse_grained")) or not _looks_like_smiles(system.monomer)

        if is_cg:
            return _build_cg(chain_length, n_chains, system)
        return _build_all_atom(system.monomer, chain_length, n_chains, system)


def _looks_like_smiles(monomer: Optional[str]) -> bool:
    if not monomer:
        return False
    return any(c in monomer for c in "()=#[]") or monomer.isalpha() and len(monomer) > 2


def _build_cg(chain_length: int, n_chains: int, system) -> BuiltSystem:
    import numpy as np
    from ase import Atoms

    bond_len = float(system.extras.get("bond_length", 0.97))  # sigma units-ish (A)
    spacing = float(system.extras.get("chain_spacing", 2.0))
    positions = []
    bonds = []
    idx = 0
    for c in range(n_chains):
        y = c * spacing
        start = idx
        for i in range(chain_length):
            positions.append([i * bond_len, y, 0.0])
            if i > 0:
                bonds.append((idx - 1, idx))
            idx += 1
        _ = start
    positions = np.array(positions)
    # Bead type: use carbon as a neutral placeholder mass/type.
    atoms = Atoms(symbols=["C"] * len(positions), positions=positions)
    span = positions.max(axis=0) - positions.min(axis=0) + 10.0
    atoms.set_cell(span)
    atoms.center()
    atoms.pbc = True
    topo = Topology(bonds=bonds, atom_types=["bead"] * len(positions))
    topo.extras["coarse_grained"] = True
    warn = [
        "Coarse-grained bead-spring polymer built (Kremer-Grest style). Running it "
        "requires a bonded engine path (LAMMPS bond_style / OpenMM custom); the "
        "current ASE potentials treat beads as non-bonded."
    ]
    label = f"CG-polymer(N={chain_length}x{n_chains})"
    return BuiltSystem(atoms=atoms, topology=topo, label=label, warnings=warn)


def _build_all_atom(monomer_smiles: str, chain_length: int, n_chains: int, system) -> BuiltSystem:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"All-atom polymer construction requires RDKit. ({exc}) "
            "Use a coarse-grained polymer (set extras.coarse_grained=true) instead."
        )
    import numpy as np
    from ase import Atoms

    # Build a homopolymer SMILES by repetition; user may supply a polymer SMILES
    # directly (e.g. with [*] connection points) for better geometry.
    poly_smiles = monomer_smiles * 1
    repeat = monomer_smiles.replace("[*]", "").replace("*", "")
    poly_smiles = repeat * chain_length

    mol = Chem.MolFromSmiles(poly_smiles)
    if mol is None:
        raise ValueError(
            f"Could not build a polymer from monomer '{monomer_smiles}'. Provide a "
            "valid repeat-unit SMILES."
        )
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:  # noqa: BLE001
        pass
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atoms = Atoms(symbols=symbols, positions=positions)
    box = positions.max(axis=0) - positions.min(axis=0) + 15.0
    atoms.set_cell(box)
    atoms.center()
    atoms.pbc = False
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    topo = Topology(bonds=bonds, atom_types=symbols)
    warnings = []
    if n_chains > 1:
        warnings.append("n_chains>1 needs packmol packing; built a single chain.")
    label = f"polymer({repeat})x{chain_length}"
    return BuiltSystem(atoms=atoms, topology=topo, label=label, warnings=warnings)
