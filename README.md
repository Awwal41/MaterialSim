# MaterialSim AI Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](build.py)
[![Documentation](https://img.shields.io/badge/docs-available-orange.svg)](docs/)

An autonomous LLM agent for computational materials science and molecular dynamics simulations.

## 🚀 Overview

The MaterialSim AI Agent is a sophisticated system that combines large language models with computational materials science tools to automate molecular dynamics simulations, property calculations, and materials discovery workflows. It enables researchers to perform complex materials simulations through natural language interfaces.

## ✨ Key Features

- **🌐 Modern Web GUI**: Intuitive web-based interface with interactive 3D visualization and real-time chat
- **🧬 Simulation Management**: Automated setup and execution of MD simulations using LAMMPS
- **📊 Property Calculation**: Automated computation of materials properties (RDF, MSD, elastic constants, thermal conductivity)
- **🤖 ML Integration**: Integration with machine learning models for accelerated property prediction
- **🗄️ Database Integration**: Query external databases (Materials Project, NOMAD, Open Catalyst Project)
- **💬 Natural Language Interface**: Accept high-level instructions in natural language
- **📈 Visualization**: Generate comprehensive plots and reports
- **⚡ HPC Ready**: Scalable from laptops to high-performance computing clusters

## 🛠️ Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/Awwal41/MaterialSim.git
cd MaterialSim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Set up configuration (add YOUR OWN keys - none are bundled)
cp env.example .env
# then edit .env and set OPENAI_API_KEY / MP_API_KEY as needed

# (Optional) verify the environment
python build.py
```

> Note: The core molecular dynamics engine runs locally with [ASE](https://wiki.fysik.dtu.dk/ase/)
> and does **not** require an API key or an external LAMMPS install. An
> `OPENAI_API_KEY` is only needed for the optional conversational chat.

### Detailed Installation
See [INSTALL.md](docs/INSTALL.md) for comprehensive installation instructions including LAMMPS setup and system requirements.

## 🎙️ JARVIS Voice Mode

Speak to the agent like Tony Stark talks to JARVIS:

```bash
pip install -r requirements_gui.txt
python launch_gui.py
```

1. Click **Activate JARVIS** in the sidebar
2. Use the microphone or type a command
3. Say things like:
   - *"Hey JARVIS, simulate copper at 300 kelvin for 2000 steps"*
   - *"Analyze the latest simulation results"*
   - *"Research the best model for property prediction"*
   - *"What's your status?"*

The agent automatically:
- **Selects the best LLM** for each task (searches arXiv + web benchmarks)
- **Runs real MD simulations** and speaks the results
- **Shows a live Jarvis HUD** animation while listening/thinking/speaking

## 🚀 Quick Start

### 🌐 Web GUI (Recommended for Beginners)
```bash
python launch_gui.py

# Or run Streamlit directly
streamlit run gui_app.py
```
The conversational GUI will guide you step-by-step through simulations with natural language interaction!

### Python API
```python
from materials_ai_agent import MaterialsAgent

# Initialize the agent (no API key needed for simulation/analysis)
agent = MaterialsAgent()

# Run a real MD simulation
result = agent.run_simulation(
    "Simulate copper at 300 K for 5000 steps"
)
print(result["success"], result["message"])

# Analyze the real output (RDF, MSD, thermodynamics)
analysis = agent.analyze_results(result["simulation_directory"])
print(analysis["rdf"]["first_peak"], "Angstrom nearest-neighbor peak")
```

### Command Line Interface
```bash
# Run a simulation
materials-agent run "Simulate silicon thermal conductivity at 300 K"

# Analyze results
materials-agent analyze ./simulations/silicon_300K/

# Interactive mode
materials-agent interactive
```

### Example Workflows
```bash
# Run basic example
python examples/basic_simulation.py

# Run ML training example (requires optional extras)
python examples/ml_training_example.py
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Quick start guide for new users
- **[User Guide](docs/user_guide.md)**: Complete usage instructions
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Examples](docs/examples.md)**: Comprehensive example workflows
- **[Installation Guide](docs/INSTALL.md)**: Step-by-step setup
- **[Project Summary](docs/PROJECT_SUMMARY.md)**: Complete project overview

## 🔬 Example Use Cases

### 1. Multi-material Study
```python
# Study multiple materials (EMT-supported metals run with real physics)
for material in ["Cu", "Al", "Au", "Ni"]:
    result = agent.run_simulation(f"Simulate {material} at 300 K for 2000 steps")
    if result["success"]:
        analysis = agent.analyze_results(result["simulation_directory"])
        print(material, analysis["rdf"]["first_peak"])
```

### 2. Machine Learning Property Prediction (optional extras)
```python
# Requires: pip install -r requirements-optional.txt
from materials_ai_agent.tools import MLTool

ml = MLTool(agent.config)
train_result = ml.train_property_predictor(
    training_data="data.csv",
    target_property="elastic_modulus",
)
```

### 3. Database Integration (optional extras)
```python
# Requires pymatgen + mp-api and a Materials Project API key
result = agent.query_database("Si")
print(result["success"], result.get("results"))
```

## 🏗️ Architecture

The system is built with a modular architecture:

- **Core Agent**: LangChain-based LLM orchestration
- **Tool System**: Specialized tools for different tasks
- **MD Interface**: LAMMPS integration for simulations
- **ML Pipeline**: Property prediction and training
- **Database Layer**: External API integration
- **Visualization**: Interactive plots and reports

## 🧪 Testing

```bash
# Run test suite
make test

# Run with coverage
python -m pytest tests/ --cov=materials_ai_agent

# Run examples
make examples
```


## 📊 Performance

- **Simulation Speed**: Optimized LAMMPS integration
- **Memory Efficiency**: Streaming data processing
- **Scalability**: HPC cluster support
- **Parallel Processing**: Multi-core simulation support

## 🔧 Configuration

The agent can be configured through environment variables or YAML files:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key
MP_API_KEY=your_materials_project_key
LAMMPS_EXECUTABLE=lmp
```

## 🐛 Troubleshooting

Common issues and solutions:

1. **LAMMPS not found**: Install LAMMPS and set `LAMMPS_EXECUTABLE`
2. **API key errors**: Verify your API keys in `.env`
3. **Memory issues**: Reduce simulation size or use smaller batches
4. **Import errors**: Ensure virtual environment is activated

See [INSTALL.md](docs/INSTALL.md) for detailed troubleshooting.


## 🙏 Acknowledgments

- **LAMMPS**: Molecular dynamics engine
- **ASE**: Atomic simulation environment
- **PyMatGen**: Materials informatics
- **LangChain**: LLM orchestration
- **OpenAI**: GPT-4 integration

## 📈 Roadmap

- [x] Real local MD engine (ASE) with RDF/MSD/thermodynamics analysis
- [x] Web interface (Streamlit)
- [ ] Optional LAMMPS backend for large-scale runs
- [ ] Additional MD engines (GROMACS, HOOMD-blue)
- [ ] Advanced ML models (Graph neural networks)
- [ ] Cloud deployment

---

**Made with ❤️ for the materials science community**
