# Materials AI Agent - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Core Features](#core-features)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Introduction

The Materials AI Agent is an autonomous system that combines large language models with computational materials science tools to automate molecular dynamics simulations, property calculations, and materials discovery workflows.

### Key Features

- **Automated MD Simulations**: Set up and run molecular dynamics simulations using LAMMPS
- **Property Analysis**: Compute materials properties from simulation data
- **Machine Learning Integration**: Train and use ML models for property prediction
- **Database Integration**: Query materials databases for comparison and benchmarking
- **Natural Language Interface**: Control the system using natural language instructions
- **Comprehensive Visualization**: Generate plots and reports automatically

## Installation

### Prerequisites

- Python 3.8 or higher
- LAMMPS (for molecular dynamics simulations)
- Git

### Step-by-Step Installation

1. **Clone the repository**:
```bash
git clone https://github.com/materials-ai-agent/materials-ai-agent.git
cd materials-ai-agent
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

5. **Install LAMMPS** (if not already installed):
```bash
# On Ubuntu/Debian
sudo apt-get install lammps

# On macOS with Homebrew
brew install lammps

# Or compile from source
# See LAMMPS documentation for details
```

### Verify Installation

```bash
materials-agent version
```

## Quick Start

### Basic Usage

```python
from materials_ai_agent import MaterialsAgent

# Initialize the agent
agent = MaterialsAgent()

# Run a simulation
result = agent.run_simulation(
    "Simulate silicon at 300 K using Tersoff potential"
)

print(result)
```

### Command Line Interface

```bash
# Run a simulation
materials-agent run "Simulate silicon thermal conductivity at 300 K"

# Analyze results
materials-agent analyze ./simulations/silicon_300K/

# Query database
materials-agent query "silicon band gap"

# Interactive mode
materials-agent interactive
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
MP_API_KEY=your_materials_project_api_key
NOMAD_API_KEY=your_nomad_api_key
LAMMPS_EXECUTABLE=lmp

# Default simulation parameters
DEFAULT_TEMPERATURE=300
DEFAULT_PRESSURE=1.0
DEFAULT_TIMESTEP=0.001

# Output directories
SIMULATION_OUTPUT_DIR=./simulations
ANALYSIS_OUTPUT_DIR=./analysis
VISUALIZATION_OUTPUT_DIR=./visualizations
```

### Configuration File

You can also use a YAML configuration file:

```yaml
# config.yaml
openai_api_key: "your_api_key"
mp_api_key: "your_mp_key"
lammps_executable: "lmp"
default_temperature: 300.0
default_pressure: 1.0
simulation_output_dir: "./simulations"
```

## Core Features

### 1. Simulation Management

The agent can set up and run molecular dynamics simulations:

```python
# Set up simulation
result = agent.tools[0].setup_simulation(
    material="Si",
    temperature=300,
    pressure=1.0,
    n_steps=10000,
    ensemble="NVT",
    force_field="tersoff"
)

# Run simulation
if result["success"]:
    sim_result = agent.tools[0].run_simulation(
        result["simulation_directory"]
    )
```

### 2. Property Analysis

Compute materials properties from simulation data:

```python
# Compute radial distribution function
rdf_result = agent.tools[1].compute_radial_distribution_function(
    trajectory_file="trajectory.dump",
    r_max=10.0,
    n_bins=200
)

# Compute mean squared displacement
msd_result = agent.tools[1].compute_mean_squared_displacement(
    trajectory_file="trajectory.dump"
)

# Analyze thermodynamic properties
thermo_result = agent.tools[1].analyze_thermodynamic_properties(
    log_file="log.lammps"
)
```

### 3. Machine Learning

Train and use ML models for property prediction:

```python
# Train a model
train_result = agent.tools[3].train_property_predictor(
    training_data="data.csv",
    target_property="thermal_conductivity",
    model_type="random_forest"
)

# Make predictions
pred_result = agent.tools[3].predict_property(
    model_name="thermal_conductivity_random_forest",
    features=[0.8, 0.1, 0.1, 300, 1.0, 100]
)
```

### 4. Database Integration

Query materials databases:

```python
# Query Materials Project
mp_result = agent.tools[2].query_materials_project(
    formula="Si",
    properties=["band_gap", "formation_energy_per_atom"]
)

# Search by structure
structure_result = agent.tools[2].search_by_structure(
    structure=structure_dict,
    tolerance=0.1
)
```

### 5. Visualization

Create plots and reports:

```python
# Create property dashboard
dashboard_result = agent.tools[4].create_property_dashboard(
    simulation_data=sim_data,
    output_file="dashboard.html"
)

# Plot 3D structure
structure_plot = agent.tools[4].plot_structure_3d(
    structure=atoms,
    output_file="structure.html"
)
```

## Examples

### Example 1: Basic Silicon Simulation

```python
from materials_ai_agent import MaterialsAgent

agent = MaterialsAgent()

# Run simulation
result = agent.run_simulation(
    "Simulate silicon at 300 K using Tersoff potential for 10000 steps"
)

# Analyze results
if result["success"]:
    analysis = agent.analyze_results(result["result"]["simulation_directory"])
    print(analysis)
```

### Example 2: Thermal Conductivity Study

```python
# Study multiple materials
materials = ["Si", "Al2O3", "Fe"]
results = {}

for material in materials:
    # Run simulation
    sim_result = agent.run_simulation(
        f"Simulate {material} at 300 K for thermal conductivity"
    )
    
    if sim_result["success"]:
        # Analyze results
        analysis = agent.analyze_results(sim_result["result"]["simulation_directory"])
        results[material] = analysis
```

### Example 3: ML Property Prediction

```python
# Train model
train_result = agent.tools[3].train_property_predictor(
    training_data="training_data.csv",
    target_property="elastic_modulus",
    model_type="neural_network"
)

# Predict for new materials
predictions = []
for material in ["Si", "Ge", "C"]:
    pred = agent.tools[3].predict_property(
        model_name=train_result["model_name"],
        features=get_features(material)
    )
    predictions.append(pred)
```

## Troubleshooting

### Common Issues

1. **LAMMPS not found**:
   - Ensure LAMMPS is installed and in your PATH
   - Set `LAMMPS_EXECUTABLE` in your configuration

2. **API key errors**:
   - Verify your API keys are correct
   - Check that the keys have the necessary permissions

3. **Memory issues**:
   - Reduce simulation size or number of steps
   - Use smaller batch sizes for ML training

4. **Simulation failures**:
   - Check force field compatibility with your material
   - Verify simulation parameters are reasonable

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = MaterialsAgent()
```

### Getting Help

- Check the logs in the output directories
- Use the `--verbose` flag with CLI commands
- Submit issues on GitHub with detailed error messages

## Advanced Usage

### Custom Force Fields

Add custom force field files:

```python
# Place force field files in the simulation directory
# The agent will automatically detect and use them
```

### Parallel Simulations

Run multiple simulations in parallel:

```python
import concurrent.futures

def run_simulation(material):
    return agent.run_simulation(f"Simulate {material} at 300 K")

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(run_simulation, ["Si", "Ge", "C"]))
```

### Custom Analysis

Extend the analysis capabilities:

```python
# Add custom analysis functions
def custom_analysis(trajectory_data):
    # Your custom analysis code
    return results

# Use in your workflow
analysis_result = custom_analysis(trajectory_data)
```

### HPC Integration

For high-performance computing environments:

```python
# Configure for HPC
config = Config(
    lammps_executable="/path/to/lammps",
    simulation_output_dir="/scratch/simulations",
    # ... other settings
)

agent = MaterialsAgent(config)
```

## Best Practices

1. **Start Small**: Begin with simple systems and short simulations
2. **Validate Results**: Compare with known values from literature
3. **Monitor Resources**: Keep track of computational requirements
4. **Save Progress**: Regularly save intermediate results
5. **Document Workflows**: Keep records of successful parameter combinations


