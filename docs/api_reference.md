# Materials AI Agent - API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Tools](#tools)
3. [Configuration](#configuration)
4. [Exceptions](#exceptions)

## Core Classes

### MaterialsAgent

Main agent class for running simulations and analysis.

```python
from materials_ai_agent import MaterialsAgent

agent = MaterialsAgent(config=None)
```

#### Methods

##### `run_simulation(instruction: str) -> Dict[str, Any]`

Run a simulation based on natural language instruction.

**Parameters:**
- `instruction` (str): Natural language description of the simulation

**Returns:**
- `Dict[str, Any]`: Dictionary containing simulation results

**Example:**
```python
result = agent.run_simulation(
    "Simulate silicon at 300 K using Tersoff potential"
)
```

##### `analyze_results(simulation_path: str) -> Dict[str, Any]`

Analyze simulation results.

**Parameters:**
- `simulation_path` (str): Path to simulation output files

**Returns:**
- `Dict[str, Any]`: Dictionary containing analysis results

##### `query_database(query: str) -> Dict[str, Any]`

Query materials databases.

**Parameters:**
- `query` (str): Natural language query about materials properties

**Returns:**
- `Dict[str, Any]`: Dictionary containing query results

##### `predict_properties(material: str, properties: List[str]) -> Dict[str, Any]`

Predict material properties using ML models.

**Parameters:**
- `material` (str): Material formula or structure
- `properties` (List[str]): List of properties to predict

**Returns:**
- `Dict[str, Any]`: Dictionary containing predictions

##### `chat(message: str) -> str`

Chat with the agent.

**Parameters:**
- `message` (str): User message

**Returns:**
- `str`: Agent response

## Tools

### SimulationTool

Tool for setting up and running molecular dynamics simulations.

```python
from materials_ai_agent.tools import SimulationTool

sim_tool = SimulationTool(config)
```

#### Methods

##### `setup_simulation(material: str, temperature: float = 300.0, pressure: float = 1.0, timestep: float = 0.001, n_steps: int = 10000, ensemble: str = "NVT", force_field: str = "tersoff", output_frequency: int = 100) -> Dict[str, Any]`

Set up a molecular dynamics simulation.

**Parameters:**
- `material` (str): Material formula (e.g., 'Si', 'Al2O3', 'H2O')
- `temperature` (float): Temperature in K
- `pressure` (float): Pressure in atm
- `timestep` (float): Timestep in ps
- `n_steps` (int): Number of simulation steps
- `ensemble` (str): Thermodynamic ensemble (NVT, NPT, NVE)
- `force_field` (str): Force field to use (tersoff, lj, eam, etc.)
- `output_frequency` (int): How often to output data

**Returns:**
- `Dict[str, Any]`: Dictionary containing simulation setup information

##### `run_simulation(simulation_directory: str, input_file: str = "in.lammps") -> Dict[str, Any]`

Run a molecular dynamics simulation.

**Parameters:**
- `simulation_directory` (str): Path to simulation directory
- `input_file` (str): Name of LAMMPS input file

**Returns:**
- `Dict[str, Any]`: Dictionary containing simulation results

##### `monitor_simulation(simulation_directory: str) -> Dict[str, Any]`

Monitor a running simulation.

**Parameters:**
- `simulation_directory` (str): Path to simulation directory

**Returns:**
- `Dict[str, Any]`: Dictionary containing simulation status and progress

### AnalysisTool

Tool for analyzing simulation results and computing materials properties.

```python
from materials_ai_agent.tools import AnalysisTool

analysis_tool = AnalysisTool(config)
```

#### Methods

##### `compute_radial_distribution_function(trajectory_file: str, atom_types: List[str] = None, r_max: float = 10.0, n_bins: int = 200) -> Dict[str, Any]`

Compute radial distribution function (RDF).

**Parameters:**
- `trajectory_file` (str): Path to trajectory dump file
- `atom_types` (List[str]): List of atom types to include (None for all)
- `r_max` (float): Maximum distance for RDF calculation
- `n_bins` (int): Number of bins for RDF

**Returns:**
- `Dict[str, Any]`: Dictionary containing RDF data and plot

##### `compute_mean_squared_displacement(trajectory_file: str, atom_types: List[str] = None, max_time: float = None) -> Dict[str, Any]`

Compute mean squared displacement (MSD).

**Parameters:**
- `trajectory_file` (str): Path to trajectory dump file
- `atom_types` (List[str]): List of atom types to include (None for all)
- `max_time` (float): Maximum time for MSD calculation (None for all)

**Returns:**
- `Dict[str, Any]`: Dictionary containing MSD data and plot

##### `compute_elastic_constants(trajectory_file: str, strain_range: float = 0.01, n_strains: int = 10) -> Dict[str, Any]`

Compute elastic constants from stress-strain analysis.

**Parameters:**
- `trajectory_file` (str): Path to trajectory dump file
- `strain_range` (float): Maximum strain for elastic constant calculation
- `n_strains` (int): Number of strain values

**Returns:**
- `Dict[str, Any]`: Dictionary containing elastic constants

##### `compute_thermal_conductivity(trajectory_file: str, method: str = "green_kubo") -> Dict[str, Any]`

Compute thermal conductivity.

**Parameters:**
- `trajectory_file` (str): Path to trajectory dump file
- `method` (str): Method for thermal conductivity calculation

**Returns:**
- `Dict[str, Any]`: Dictionary containing thermal conductivity data

##### `analyze_thermodynamic_properties(log_file: str) -> Dict[str, Any]`

Analyze thermodynamic properties from log file.

**Parameters:**
- `log_file` (str): Path to LAMMPS log file

**Returns:**
- `Dict[str, Any]`: Dictionary containing thermodynamic analysis

### DatabaseTool

Tool for querying materials databases.

```python
from materials_ai_agent.tools import DatabaseTool

db_tool = DatabaseTool(config)
```

#### Methods

##### `query_materials_project(formula: str, properties: List[str] = None) -> Dict[str, Any]`

Query Materials Project database.

**Parameters:**
- `formula` (str): Chemical formula (e.g., 'Si', 'Al2O3')
- `properties` (List[str]): List of properties to retrieve

**Returns:**
- `Dict[str, Any]`: Dictionary containing Materials Project data

##### `get_elastic_properties(material_id: str) -> Dict[str, Any]`

Get elastic properties from Materials Project.

**Parameters:**
- `material_id` (str): Materials Project material ID

**Returns:**
- `Dict[str, Any]`: Dictionary containing elastic properties

##### `search_by_structure(structure: Dict[str, Any], tolerance: float = 0.1) -> Dict[str, Any]`

Search Materials Project by structure similarity.

**Parameters:**
- `structure` (Dict[str, Any]): Structure dictionary (from ASE or pymatgen)
- `tolerance` (float): Structure matching tolerance

**Returns:**
- `Dict[str, Any]`: Dictionary containing similar structures

##### `query_nomad(query: str, max_results: int = 10) -> Dict[str, Any]`

Query NOMAD database.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of results

**Returns:**
- `Dict[str, Any]`: Dictionary containing NOMAD search results

### MLTool

Tool for machine learning-based property prediction.

```python
from materials_ai_agent.tools import MLTool

ml_tool = MLTool(config)
```

#### Methods

##### `train_property_predictor(training_data: str, target_property: str, model_type: str = "random_forest", test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]`

Train a machine learning model for property prediction.

**Parameters:**
- `training_data` (str): Path to training data CSV file
- `target_property` (str): Name of target property column
- `model_type` (str): Type of model ('random_forest', 'gradient_boosting', 'neural_network')
- `test_size` (float): Fraction of data to use for testing
- `random_state` (int): Random state for reproducibility

**Returns:**
- `Dict[str, Any]`: Dictionary containing training results

##### `predict_property(model_name: str, features: List[float]) -> Dict[str, Any]`

Predict property using trained model.

**Parameters:**
- `model_name` (str): Name of trained model
- `features` (List[float]): Feature values for prediction

**Returns:**
- `Dict[str, Any]`: Dictionary containing prediction and uncertainty

##### `train_neural_network(training_data: str, target_property: str, hidden_layers: List[int] = [100, 50], learning_rate: float = 0.001, epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]`

Train a neural network for property prediction.

**Parameters:**
- `training_data` (str): Path to training data CSV file
- `target_property` (str): Name of target property column
- `hidden_layers` (List[int]): List of hidden layer sizes
- `learning_rate` (float): Learning rate for optimizer
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training

**Returns:**
- `Dict[str, Any]`: Dictionary containing training results

##### `predict_with_uncertainty(model_name: str, features: List[float], n_samples: int = 100) -> Dict[str, Any]`

Predict property with uncertainty quantification.

**Parameters:**
- `model_name` (str): Name of trained model
- `features` (List[float]): Feature values for prediction
- `n_samples` (int): Number of samples for uncertainty estimation

**Returns:**
- `Dict[str, Any]`: Dictionary containing prediction with uncertainty

### VisualizationTool

Tool for creating visualizations and reports.

```python
from materials_ai_agent.tools import VisualizationTool

viz_tool = VisualizationTool(config)
```

#### Methods

##### `create_property_dashboard(simulation_data: Dict[str, Any], output_file: str = "property_dashboard.html") -> Dict[str, Any]`

Create an interactive dashboard of material properties.

**Parameters:**
- `simulation_data` (Dict[str, Any]): Dictionary containing simulation results
- `output_file` (str): Output HTML file name

**Returns:**
- `Dict[str, Any]`: Dictionary containing dashboard information

##### `plot_structure_3d(structure: Dict[str, Any], output_file: str = "structure_3d.html") -> Dict[str, Any]`

Create 3D visualization of atomic structure.

**Parameters:**
- `structure` (Dict[str, Any]): Structure dictionary or ASE Atoms object
- `output_file` (str): Output HTML file name

**Returns:**
- `Dict[str, Any]`: Dictionary containing visualization information

##### `create_comparison_plot(data_sets: List[Dict[str, Any]], x_column: str, y_column: str, title: str = "Comparison Plot", output_file: str = "comparison_plot.png") -> Dict[str, Any]`

Create comparison plot of multiple data sets.

**Parameters:**
- `data_sets` (List[Dict[str, Any]]): List of data dictionaries with 'data' and 'label' keys
- `x_column` (str): Name of x-axis column
- `y_column` (str): Name of y-axis column
- `title` (str): Plot title
- `output_file` (str): Output file name

**Returns:**
- `Dict[str, Any]`: Dictionary containing plot information

##### `create_heatmap(data: Union[np.ndarray, pd.DataFrame], x_labels: List[str] = None, y_labels: List[str] = None, title: str = "Heatmap", output_file: str = "heatmap.png") -> Dict[str, Any]`

Create heatmap visualization.

**Parameters:**
- `data` (Union[np.ndarray, pd.DataFrame]): 2D data array or DataFrame
- `x_labels` (List[str]): X-axis labels
- `y_labels` (List[str]): Y-axis labels
- `title` (str): Plot title
- `output_file` (str): Output file name

**Returns:**
- `Dict[str, Any]`: Dictionary containing plot information

##### `generate_report(simulation_results: Dict[str, Any], analysis_results: Dict[str, Any] = None, output_file: str = "simulation_report.html") -> Dict[str, Any]`

Generate comprehensive HTML report.

**Parameters:**
- `simulation_results` (Dict[str, Any]): Simulation results dictionary
- `analysis_results` (Dict[str, Any]): Analysis results dictionary
- `output_file` (str): Output HTML file name

**Returns:**
- `Dict[str, Any]`: Dictionary containing report information

## Configuration

### Config

Configuration class for the Materials AI Agent.

```python
from materials_ai_agent.core.config import Config

config = Config.from_env()
```

#### Class Variables

- `openai_api_key` (str): OpenAI API key
- `mp_api_key` (Optional[str]): Materials Project API key
- `nomad_api_key` (Optional[str]): NOMAD API key
- `lammps_executable` (str): LAMMPS executable path
- `default_temperature` (float): Default temperature in K
- `default_pressure` (float): Default pressure in atm
- `default_timestep` (float): Default timestep in ps
- `simulation_output_dir` (Path): Simulation output directory
- `analysis_output_dir` (Path): Analysis output directory
- `visualization_output_dir` (Path): Visualization output directory
- `log_level` (str): Logging level
- `model_name` (str): OpenAI model name
- `max_tokens` (int): Maximum tokens for LLM responses
- `temperature` (float): LLM temperature

#### Methods

##### `from_env() -> Config`

Load configuration from environment variables.

**Returns:**
- `Config`: Configuration object

##### `create_directories() -> None`

Create necessary output directories.

## Exceptions

### MaterialsAgentError

Base exception for Materials AI Agent.

```python
from materials_ai_agent.core.exceptions import MaterialsAgentError
```

### SimulationError

Exception raised during simulation setup or execution.

```python
from materials_ai_agent.core.exceptions import SimulationError
```

### AnalysisError

Exception raised during data analysis.

```python
from materials_ai_agent.core.exceptions import AnalysisError
```

### DatabaseError

Exception raised during database operations.

```python
from materials_ai_agent.core.exceptions import DatabaseError
```

### MLModelError

Exception raised during ML model operations.

```python
from materials_ai_agent.core.exceptions import MLModelError
```

### ConfigurationError

Exception raised due to configuration issues.

```python
from materials_ai_agent.core.exceptions import ConfigurationError
```
