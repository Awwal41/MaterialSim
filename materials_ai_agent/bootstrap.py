"""Register all built-in plugins into the runtime registries.

Import side effects are avoided elsewhere: nothing registers itself at import
time except through :func:`ensure`, which is idempotent and cheap. Concrete
plugins do their heavy/optional imports lazily inside their methods, so calling
``ensure()`` never requires LAMMPS/OpenMM/MACE/RDKit to be installed.
"""

from __future__ import annotations

_DONE = False


def ensure() -> None:
    """Populate engine/potential/protocol/structure registries once."""
    global _DONE
    if _DONE:
        return

    # -- engines ------------------------------------------------------
    from .engines.registry import register_engine
    from .engines.ase_adapter import ASEAdapter
    from .engines.lammps_adapter import LAMMPSAdapter
    from .engines.openmm_adapter import OpenMMAdapter

    register_engine("ase", ASEAdapter)
    register_engine("lammps", LAMMPSAdapter)
    register_engine("openmm", OpenMMAdapter)

    # -- potentials ---------------------------------------------------
    from .potentials.registry import register_potential
    from .potentials.emt import EMTPotential
    from .potentials.lennard_jones import LennardJonesPotential
    from .potentials.classical_lammps import (
        EAMPotential,
        MEAMPotential,
        ReaxFFPotential,
        TersoffPotential,
    )
    from .potentials.mlip import CHGNetPotential, M3GNetPotential, MACEPotential
    from .potentials.bonded import GAFFPotential, OpenFFPotential, OPLSPotential

    register_potential("emt", EMTPotential)
    register_potential("lj", LennardJonesPotential)
    register_potential("eam", EAMPotential)
    register_potential("tersoff", TersoffPotential)
    register_potential("meam", MEAMPotential)
    register_potential("reaxff", ReaxFFPotential)
    register_potential("mace", MACEPotential)
    register_potential("chgnet", CHGNetPotential)
    register_potential("m3gnet", M3GNetPotential)
    register_potential("opls", OPLSPotential)
    register_potential("gaff", GAFFPotential)
    register_potential("openff", OpenFFPotential)

    # -- protocols ----------------------------------------------------
    from .protocols.registry import register_protocol
    from .protocols.equilibrium import EquilibriumProtocol
    from .protocols.nemd import NEMDProtocol
    from .protocols.msst import MSSTProtocol
    from .protocols.deformation import DeformationProtocol

    register_protocol("equilibrium", lambda p: EquilibriumProtocol(p))
    register_protocol("nemd", lambda p: NEMDProtocol(p))
    register_protocol("msst", lambda p: MSSTProtocol(p))
    register_protocol("deformation", lambda p: DeformationProtocol(p))

    # -- structure builders ------------------------------------------
    from .structures.registry import register_builder
    from .structures.file_loader import FileLoaderBuilder
    from .structures.crystal import CrystalBuilder
    from .structures.molecule import MoleculeBuilder
    from .structures.polymer import PolymerBuilder

    register_builder("file", FileLoaderBuilder, priority=10)
    register_builder("material_project", CrystalBuilder, priority=20)
    register_builder("polymer", PolymerBuilder, priority=30)
    register_builder("molecule", MoleculeBuilder, priority=40)
    register_builder("crystal", CrystalBuilder, priority=90)

    _DONE = True
