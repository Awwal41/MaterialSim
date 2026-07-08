"""Tests for complex simulation parsing and structure building."""

import pytest

from materials_ai_agent import Config
from materials_ai_agent.simulation_parser import parse_simulation_instruction
from materials_ai_agent.structure_builder import (
    build_atoms,
    infer_material_label,
    load_structure_file,
    normalize_structure_source,
    parse_alloy_notation,
)


@pytest.fixture
def config():
    return Config(openai_api_key="")


class TestSimulationParser:
    def test_basic_copper(self, config):
        spec = parse_simulation_instruction("simulate copper at 250 K for 5000 steps", config)
        assert spec.material == "Cu"
        assert spec.temperature == 250.0
        assert spec.n_steps == 5000

    def test_npt_pressure_bar(self, config):
        spec = parse_simulation_instruction(
            "NPT simulation of Al at 600 K and 10 bar for 2000 steps", config
        )
        assert spec.material == "Al"
        assert spec.ensemble == "NPT"
        assert spec.temperature == 600.0
        assert abs(spec.pressure - 10 * 0.986923) < 0.1

    def test_duration_ps(self, config):
        spec = parse_simulation_instruction("Cu NVT at 300 K for 10 ps", config)
        assert spec.material == "Cu"
        assert spec.n_steps == 10_000  # 10 ps / 0.001 ps timestep

    def test_supercell_and_atoms(self, config):
        spec = parse_simulation_instruction(
            "256 atom Cu supercell 4x4x4 NVT at 300 K for 1000 steps", config
        )
        assert spec.target_atoms == 256
        assert spec.supercell_reps == (4, 4, 4)

    def test_compound(self, config):
        spec = parse_simulation_instruction("Al2O3 NVT at 800 K for 3000 steps", config)
        assert spec.material == "Al2O3"

    def test_alloy(self, config):
        spec = parse_simulation_instruction("CuNi alloy NVT at 500 K for 2000 steps", config)
        assert spec.alloy_elements == ["Cu", "Ni"]
        assert spec.material == "CuNi"

    def test_decimal_alloy(self, config):
        spec = parse_simulation_instruction("Cu0.8Ni0.2 NVT at 400 K for 1000 steps", config)
        assert spec.alloy_elements == ["Cu", "Ni"]
        assert spec.alloy_fractions == [0.8, 0.2]

    def test_thermostat_and_timestep(self, config):
        spec = parse_simulation_instruction(
            "Cu Berendsen NVT at 300 K timestep 0.002 ps for 5000 steps", config
        )
        assert spec.thermostat == "Berendsen"
        assert spec.timestep == 0.002

    def test_mp_id_sets_material_project_source(self, config):
        spec = parse_simulation_instruction(
            "NVT simulation using mp-134 for 2000 steps at 300 K", config
        )
        assert spec.mp_material_id == "mp-134"
        assert spec.structure_source == "material_project"
        assert spec.material == "mp-134"

    def test_upload_keyword_sets_file_source(self, config):
        spec = parse_simulation_instruction(
            "NVT simulation of my structure file at 300 K for 1000 steps", config
        )
        assert spec.structure_source == "file"
        assert spec.material == "custom"


class TestStructureBuilder:
    def test_build_copper_supercell(self):
        atoms = build_atoms("Cu", target_atoms=64)
        assert len(atoms) >= 32
        assert set(atoms.get_chemical_symbols()) == {"Cu"}

    def test_build_alloy(self):
        atoms = build_atoms(
            "CuNi",
            alloy_elements=["Cu", "Ni"],
            alloy_fractions=[0.5, 0.5],
            target_atoms=32,
        )
        symbols = set(atoms.get_chemical_symbols())
        assert symbols <= {"Cu", "Ni"}
        assert len(symbols) == 2

    def test_parse_alloy_notation(self):
        assert parse_alloy_notation("Cu0.8Ni0.2") == (["Cu", "Ni"], [0.8, 0.2])
        assert parse_alloy_notation("Fe-Cr alloy")[0] == ["Fe", "Cr"]
        assert parse_alloy_notation("Al2O3 NVT") is None

    def test_normalize_structure_source(self):
        assert normalize_structure_source("upload") == "file"
        assert normalize_structure_source("material_project") == "material_project"
        assert normalize_structure_source("mp") == "material_project"

    def test_load_structure_file_xyz(self, tmp_path):
        xyz = tmp_path / "cu2.xyz"
        xyz.write_text(
            "2\nCu dimer\n"
            "Cu 0 0 0\n"
            "Cu 2 0 0\n",
            encoding="utf-8",
        )
        atoms = load_structure_file(xyz)
        assert len(atoms) == 2
        assert set(atoms.get_chemical_symbols()) == {"Cu"}

    def test_build_from_file(self, tmp_path):
        xyz = tmp_path / "al.xyz"
        xyz.write_text(
            "1\nAl\n"
            "Al 0 0 0\n",
            encoding="utf-8",
        )
        atoms = build_atoms("custom", structure_source="file", structure_file=str(xyz))
        assert len(atoms) == 1
        assert atoms.get_chemical_symbols() == ["Al"]
