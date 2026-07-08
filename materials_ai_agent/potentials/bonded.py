"""Bonded / all-atom force fields for molecular & polymer systems (OpenMM).

These build an OpenMM ``System`` for organic/soft-matter systems using the
OpenFF/AMBER-GAFF toolchains (via ``openmmforcefields`` + ``openff-toolkit``).
They target the OpenMM engine and require a SMILES (small molecule / monomer)
so a parameterizable molecule can be constructed.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from .base import PotentialProvider

_ORGANIC = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si"}


class _BondedProvider(PotentialProvider):
    engines = {"openmm"}
    _small_molecule_ff = "gaff-2.11"

    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        # Bonded FFs are meant for molecular systems; require organic chemistry.
        return set(elements).issubset(_ORGANIC | {"Na", "K", "Ca", "Mg", "Cl"})

    def available(self) -> bool:
        try:
            import openff.toolkit  # noqa: F401
            import openmmforcefields  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _smiles(self, job) -> str:
        s = job.spec.system
        smiles = s.smiles or s.monomer
        if not smiles:
            raise ValueError(
                f"The {self.kind} force field needs a SMILES string (system.smiles "
                "or system.monomer) to parameterize the molecule."
            )
        return smiles

    def openmm_system(self, job) -> Dict[str, Any]:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SystemGenerator

        try:
            from openmm import unit
            from openmm.app import ForceField, Modeller, PDBFile  # noqa: F401
        except Exception:  # noqa: BLE001
            from simtk import unit  # type: ignore
            from simtk.openmm.app import ForceField  # type: ignore  # noqa: F401

        mol = Molecule.from_smiles(self._smiles(job))
        mol.generate_conformers(n_conformers=1)

        gen = SystemGenerator(
            small_molecule_forcefield=self._small_molecule_ff,
            molecules=[mol],
        )
        off_top = mol.to_topology()
        omm_top = off_top.to_openmm()
        system = gen.create_system(omm_top)
        positions = mol.conformers[0].to_openmm()
        return {"system": system, "topology": omm_top, "positions": positions}


class GAFFPotential(_BondedProvider):
    kind = "gaff"
    description = "AMBER GAFF 2 small-molecule force field (via openmmforcefields)."
    _small_molecule_ff = "gaff-2.11"


class OpenFFPotential(_BondedProvider):
    kind = "openff"
    description = "Open Force Field (Sage) small-molecule force field."
    _small_molecule_ff = "openff-2.1.0"


class OPLSPotential(_BondedProvider):
    kind = "opls"
    description = "OPLS-style all-atom parameters (approximated via GAFF toolchain)."
    _small_molecule_ff = "gaff-2.11"

    def warnings_for(self, elements: Set[str]):
        return [
            "True OPLS-AA typing requires an OPLS parameter source; this run uses "
            "the GAFF toolchain as a practical stand-in. Supply LigParGen/foyer "
            "parameters for strict OPLS."
        ]
