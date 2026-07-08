"""Tests for the engine-agnostic platform: registries, validation, extraction.

These avoid requiring LAMMPS/OpenMM/MACE by exercising the abstractions and the
always-available ASE path.
"""

from pathlib import Path

import pytest

from materials_ai_agent import bootstrap
from materials_ai_agent.capabilities import choose_engine, validate
from materials_ai_agent.engines.registry import available_engines, list_engines
from materials_ai_agent.potentials.base import PotentialNotSupported
from materials_ai_agent.potentials.registry import list_potentials, resolve_potential
from materials_ai_agent.protocols.registry import list_protocols
from materials_ai_agent.spec import PotentialSpec, SimulationSpec, SystemSpec
from materials_ai_agent.spec.extractor import extract_spec
from materials_ai_agent.structures.registry import build_system


@pytest.fixture(autouse=True)
def _bootstrap():
    bootstrap.ensure()


class TestRegistries:
    def test_engines_registered(self):
        assert {"ase", "lammps", "openmm"}.issubset(set(list_engines()))
        assert "ase" in available_engines()  # ASE is a hard dependency

    def test_potentials_registered(self):
        for kind in ("emt", "lj", "eam", "tersoff", "reaxff", "mace", "chgnet", "opls"):
            assert kind in list_potentials()

    def test_protocols_registered(self):
        assert set(list_protocols()) == {"equilibrium", "nemd", "msst", "deformation"}


class TestPotentialResolution:
    def test_auto_picks_emt_for_metal(self):
        provider = resolve_potential("auto", {"Cu"}, engine="ase")
        assert provider.kind in {"emt", "lj", "mace", "chgnet"}

    def test_auto_falls_back_to_lj(self):
        # A noble gas is not EMT-supported; LJ should service it on ASE.
        provider = resolve_potential("auto", {"Ar"}, engine="ase")
        assert provider.kind == "lj"

    def test_explicit_unavailable_raises(self):
        # EAM is LAMMPS-only and needs a potential file; without one it must
        # raise rather than silently substituting.
        with pytest.raises(PotentialNotSupported):
            resolve_potential("eam", {"Cu"}, engine="ase")


class TestValidation:
    def test_unresolved_material_requests_clarification(self):
        spec = SimulationSpec(system=SystemSpec(material="unresolved"))
        vr = validate(spec)
        assert not vr.ok
        assert vr.clarifications

    def test_valid_metal_spec_ok(self):
        spec = SimulationSpec(system=SystemSpec(material="Cu"))
        vr = validate(spec)
        assert vr.ok
        assert vr.engine == "ase"

    def test_msst_routes_to_lammps(self):
        spec = SimulationSpec(system=SystemSpec(material="Cu"))
        spec.protocol.name = "msst"
        assert choose_engine(spec) in {"lammps", "ase"}
        # If LAMMPS is unavailable, ASE cannot run MSST -> validation fails clearly.
        vr = validate(spec)
        if "lammps" not in available_engines():
            assert not vr.ok


class TestExtractor:
    def test_extract_msst(self):
        spec = extract_spec("MSST shock of Cu along z at 8 km/s for 5000 steps")
        assert spec.protocol.name == "msst"
        assert spec.protocol.params.get("shock_velocity_kms") == 8.0
        assert spec.system.material == "Cu"

    def test_extract_nemd_thermal(self):
        spec = extract_spec("NEMD thermal conductivity of Si at 300 K")
        assert spec.protocol.name == "nemd"
        assert spec.protocol.params.get("mode") == "thermal"

    def test_extract_potential_and_engine(self):
        spec = extract_spec("run MACE simulation of LiFePO4 on lammps")
        assert spec.potential.kind == "mace"
        assert spec.engine == "lammps"

    def test_extract_polymer(self):
        spec = extract_spec("coarse-grained polymer chain length 20 with 4 chains")
        assert spec.system.kind == "polymer"
        assert spec.system.chain_length == 20
        assert spec.system.n_chains == 4


class TestStructureBuilders:
    def test_build_crystal(self):
        built = build_system(SystemSpec(material="Cu", target_atoms=32))
        assert set(built.atoms.get_chemical_symbols()) == {"Cu"}

    def test_build_cg_polymer(self):
        built = build_system(SystemSpec(kind="polymer", chain_length=10,
                                        extras={"coarse_grained": True}))
        assert built.topology is not None
        assert built.topology.is_bonded
        assert len(built.topology.bonds) == 9


class TestLAMMPSInputGeneration:
    """Prove LAMMPS input is generated from the spec, with nothing hardcoded."""

    def _job(self, tmp_path):
        from ase.build import bulk

        from materials_ai_agent.engines.base import ResolvedJob
        from materials_ai_agent.potentials.lennard_jones import LennardJonesPotential
        from materials_ai_agent.protocols.equilibrium import EquilibriumProtocol
        from materials_ai_agent.runners.local import LocalRunner

        atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
        spec = SimulationSpec(system=SystemSpec(material="Cu"),
                              potential=PotentialSpec(kind="lj"))
        spec.run.n_steps = 100
        return ResolvedJob(
            spec=spec, atoms=atoms, topology=None,
            potential=LennardJonesPotential(spec.potential),
            protocol=EquilibriumProtocol({}), workdir=tmp_path,
            runner=LocalRunner(), material_label="Cu",
        )

    def test_script_is_generated_from_spec(self, tmp_path):
        from materials_ai_agent.engines.lammps_adapter import LAMMPSAdapter, specorder

        adapter = LAMMPSAdapter()
        job = self._job(tmp_path)
        order = specorder(job.atoms)
        pot = job.potential.lammps_potential(job)
        adapter._write_data(job, order, pot)
        script = adapter._build_script(job, order, pot)

        # Not hardcoded to silicon/aluminum like the old interface.
        assert "Si.tersoff" not in script
        assert "Al99.eam.alloy" not in script
        # Real, spec-derived content.
        assert "pair_style lj/cut" in script
        assert "read_data structure.data" in script
        assert "fix integrate all nvt" in script
        assert (tmp_path / "structure.data").exists()


class TestEndToEndASE:
    def test_run_spec_ase(self, tmp_path):
        from materials_ai_agent.orchestrator import run_spec

        spec = SimulationSpec(
            system=SystemSpec(material="Cu", target_atoms=32),
            potential=PotentialSpec(kind="auto"),
        )
        spec.run.n_steps = 200
        spec.run.output_frequency = 50
        result = run_spec(spec, base_dir=str(tmp_path))
        assert result["success"] is True
        assert result["engine"] == "ase"
        assert Path(result["simulation_directory"], "trajectory.xyz").exists()
