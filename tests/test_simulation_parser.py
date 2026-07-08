"""Tests for complex simulation parsing and structure building."""

import pytest

from materials_ai_agent import Config
from materials_ai_agent.simulation_parser import parse_simulation_instruction
from materials_ai_agent.structure_builder import build_atoms, parse_alloy_notation


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
