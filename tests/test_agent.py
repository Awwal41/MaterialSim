"""Tests for Materials AI Agent."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from materials_ai_agent import MaterialsAgent, Config
from materials_ai_agent.core.exceptions import MaterialsAgentError


class TestMaterialsAgent:
    """Test cases for MaterialsAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            openai_api_key="test_key",
            simulation_output_dir=Path(self.temp_dir) / "simulations",
            analysis_output_dir=Path(self.temp_dir) / "analysis",
            visualization_output_dir=Path(self.temp_dir) / "visualizations"
        )
        self.config.create_directories()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_agent_initialization(self, mock_llm):
        """Test agent initialization."""
        agent = MaterialsAgent(self.config)
        
        assert agent.config == self.config
        assert len(agent.tools) == 5  # 5 tools
        assert agent.llm is not None
        assert agent.memory is not None
        assert agent.agent is not None
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_run_simulation(self, mock_llm):
        """Test running a simulation."""
        # Mock the agent's invoke method
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Simulation completed successfully"
        }
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        result = agent.run_simulation("Test simulation")
        
        assert result["success"] is True
        assert "Simulation completed successfully" in result["result"]
        mock_agent.invoke.assert_called_once()
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_analyze_results(self, mock_llm):
        """Test analyzing simulation results."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Analysis completed"
        }
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        result = agent.analyze_results("/path/to/simulation")
        
        assert result["success"] is True
        assert "Analysis completed" in result["analysis"]
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_query_database(self, mock_llm):
        """Test querying database."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Database query results"
        }
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        result = agent.query_database("Test query")
        
        assert result["success"] is True
        assert "Database query results" in result["results"]
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_predict_properties(self, mock_llm):
        """Test predicting properties."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Property predictions"
        }
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        result = agent.predict_properties("Si", ["thermal_conductivity"])
        
        assert result["success"] is True
        assert "Property predictions" in result["predictions"]
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_chat(self, mock_llm):
        """Test chat functionality."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Hello! How can I help you?"
        }
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        response = agent.chat("Hello")
        
        assert response == "Hello! How can I help you?"
    
    @patch('materials_ai_agent.core.agent.ChatOpenAI')
    def test_error_handling(self, mock_llm):
        """Test error handling."""
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Test error")
        
        agent = MaterialsAgent(self.config)
        agent.agent = mock_agent
        
        result = agent.run_simulation("Test simulation")
        
        assert result["success"] is False
        assert "Test error" in result["error"]


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = Config(
            openai_api_key="test_key",
            simulation_output_dir=Path("/tmp/simulations"),
            analysis_output_dir=Path("/tmp/analysis"),
            visualization_output_dir=Path("/tmp/visualizations")
        )
        
        assert config.openai_api_key == "test_key"
        assert config.simulation_output_dir == Path("/tmp/simulations")
        assert config.analysis_output_dir == Path("/tmp/analysis")
        assert config.visualization_output_dir == Path("/tmp/visualizations")
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'env_key',
        'MP_API_KEY': 'mp_key',
        'LAMMPS_EXECUTABLE': 'lmp_test'
    })
    def test_from_env(self):
        """Test loading config from environment."""
        config = Config.from_env()
        
        assert config.openai_api_key == "env_key"
        assert config.mp_api_key == "mp_key"
        assert config.lammps_executable == "lmp_test"
    
    def test_create_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                openai_api_key="test_key",
                simulation_output_dir=Path(temp_dir) / "simulations",
                analysis_output_dir=Path(temp_dir) / "analysis",
                visualization_output_dir=Path(temp_dir) / "visualizations"
            )
            
            config.create_directories()
            
            assert config.simulation_output_dir.exists()
            assert config.analysis_output_dir.exists()
            assert config.visualization_output_dir.exists()


class TestSimulationTool:
    """Test cases for SimulationTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            openai_api_key="test_key",
            simulation_output_dir=Path(self.temp_dir) / "simulations",
            analysis_output_dir=Path(self.temp_dir) / "analysis",
            visualization_output_dir=Path(self.temp_dir) / "visualizations"
        )
        self.config.create_directories()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('materials_ai_agent.tools.simulation.LAMMPSInterface')
    def test_setup_simulation(self, mock_lammps):
        """Test simulation setup."""
        from materials_ai_agent.tools import SimulationTool
        
        mock_interface = Mock()
        mock_interface.generate_input_file.return_value = Path("test_input.lammps")
        mock_lammps.return_value = mock_interface
        
        tool = SimulationTool(self.config)
        
        result = tool.setup_simulation(
            material="Si",
            temperature=300,
            n_steps=1000
        )
        
        assert result["success"] is True
        assert "Si" in result["parameters"]["material"]
        assert result["parameters"]["temperature"] == 300
        assert result["parameters"]["n_steps"] == 1000
    
    def test_list_available_materials(self):
        """Test listing available materials."""
        from materials_ai_agent.tools import SimulationTool
        
        tool = SimulationTool(self.config)
        result = tool.list_available_materials()
        
        assert result["success"] is True
        assert "materials" in result
        assert "elements" in result["materials"]
        assert "compounds" in result["materials"]
    
    def test_get_force_fields(self):
        """Test getting available force fields."""
        from materials_ai_agent.tools import SimulationTool
        
        tool = SimulationTool(self.config)
        result = tool.get_force_fields()
        
        assert result["success"] is True
        assert "force_fields" in result
        assert "tersoff" in result["force_fields"]
        assert "lj" in result["force_fields"]


if __name__ == "__main__":
    pytest.main([__file__])
