"""Tests for the Materials AI Agent.

These are integration tests: the simulation tests actually run short
molecular dynamics simulations with the ASE-backed engine.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from materials_ai_agent import Config, MaterialsAgent


class TestMaterialsAgent:
    """Test cases for the MaterialsAgent class."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            openai_api_key="",  # no key -> chat disabled, sim/analysis still work
            simulation_output_dir=Path(self.temp_dir) / "simulations",
            analysis_output_dir=Path(self.temp_dir) / "analysis",
            visualization_output_dir=Path(self.temp_dir) / "visualizations",
        )
        self.config.create_directories()
        self.agent = MaterialsAgent(self.config)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_initialization(self):
        assert self.agent.config == self.config
        assert len(self.agent.tools) >= 1
        # No API key -> conversational agent is disabled but object is usable.
        assert self.agent.agent is None

    def test_parse_instruction(self):
        params = self.agent._parse_simulation_instruction(
            "simulate copper at 250 K for 5000 steps"
        )
        assert params["material"] == "Cu"
        assert params["temperature"] == 250.0
        assert params["n_steps"] == 5000

    def test_run_simulation(self):
        result = self.agent.run_simulation("simulate copper at 300 K for 1000 steps")
        assert result["success"] is True
        assert "message" in result
        assert Path(result["simulation_directory"]).exists()

    def test_analyze_results(self):
        sim = self.agent.run_simulation("simulate copper at 300 K for 1000 steps")
        assert sim["success"] is True

        result = self.agent.analyze_results(sim["simulation_directory"])
        assert result["success"] is True
        assert result["rdf"]["success"] is True
        assert result["thermodynamics"]["success"] is True

    def test_chat_without_key_is_graceful(self):
        response = self.agent.chat("Hello")
        assert "unavailable" in response.lower()

    def test_run_simulation_error_handling(self):
        # An unbuildable material should fail cleanly, not raise.
        result = self.agent.run_simulation("simulate Xx99 at 300 K")
        assert isinstance(result, dict)
        assert "success" in result


class TestConfig:
    """Test cases for the Config class."""

    def test_config_creation(self):
        config = Config(
            openai_api_key="test_key",
            simulation_output_dir=Path("/tmp/simulations"),
        )
        assert config.openai_api_key == "test_key"
        assert config.simulation_output_dir == Path("/tmp/simulations")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env_key")
        monkeypatch.setenv("MP_API_KEY", "mp_key")
        monkeypatch.setenv("LAMMPS_EXECUTABLE", "lmp_test")
        config = Config.from_env()
        assert config.openai_api_key == "env_key"
        assert config.mp_api_key == "mp_key"
        assert config.lammps_executable == "lmp_test"

    def test_from_file_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_path = Path(temp_dir) / "cfg.yaml"
            cfg_path.write_text(
                "openai_api_key: file_key\n"
                "default_temperature: 500.0\n"
                "model_name: gpt-4\n"
            )
            config = Config.from_file(str(cfg_path))
            assert config.openai_api_key == "file_key"
            assert config.default_temperature == 500.0

    def test_create_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                openai_api_key="test_key",
                simulation_output_dir=Path(temp_dir) / "simulations",
                analysis_output_dir=Path(temp_dir) / "analysis",
                visualization_output_dir=Path(temp_dir) / "visualizations",
            )
            config.create_directories()
            assert config.simulation_output_dir.exists()
            assert config.analysis_output_dir.exists()
            assert config.visualization_output_dir.exists()


class TestSimulationEngine:
    """Direct tests of the real MD engine."""

    def test_real_simulation_and_output(self):
        from materials_ai_agent.simple_simulation import run_simple_simulation

        # force_field="auto" resolves to a runnable potential (EMT for Cu).
        # Requesting a LAMMPS-only potential like "eam" without LAMMPS installed
        # now fails loudly by design rather than silently downgrading.
        result = run_simple_simulation(
            material="Cu",
            temperature=300,
            n_steps=500,
            force_field="auto",
            ensemble="NVT",
            output_frequency=100,
        )
        assert result["success"] is True
        sim_dir = Path(result["simulation_directory"])
        assert (sim_dir / "trajectory.xyz").exists()
        assert (sim_dir / "output.log").exists()
        assert result["n_frames"] >= 2

    def test_rdf_matches_copper_spacing(self):
        from materials_ai_agent.analysis_engine import compute_rdf
        from materials_ai_agent.simple_simulation import run_simple_simulation

        result = run_simple_simulation(
            material="Cu", temperature=100, n_steps=200, force_field="auto"
        )
        rdf = compute_rdf(Path(result["simulation_directory"]))
        assert rdf["success"] is True
        # Copper nearest-neighbor distance is ~2.55 Angstrom.
        assert 2.0 < rdf["first_peak"] < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
