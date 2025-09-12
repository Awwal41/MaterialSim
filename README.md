# MaterialSim AI Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
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
git clone https://github.com/materials-ai-agent/materials-ai-agent.git
cd materials-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Set up configuration (API keys are pre-configured!)
cp config_with_keys.env .env

# Run build script
python build.py

# Or run quick setup for immediate use
python quick_setup.py
```

### Detailed Installation
See [INSTALL.md](docs/INSTALL.md) for comprehensive installation instructions including LAMMPS setup and system requirements.

## 🚀 Quick Start

### 🌐 Web GUI (Recommended for Beginners)
```bash
# Launch the pure non-hardcoded interface (NEW!)
python launch_gui.py

# Or launch the integrated GUI
python launch_integrated_gui.py
```
The conversational GUI will guide you step-by-step through simulations with natural language interaction!

### Python API
```python
from materials_ai_agent import MaterialsAgent

# Initialize the agent
agent = MaterialsAgent()

# Run a simulation
result = agent.run_simulation(
    "Simulate the thermal conductivity of silicon at 300 K using a Tersoff potential"
)

print(result)
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

# Run ML training example
python examples/ml_training_example.py

# Run advanced workflow
python examples/advanced_workflow.py
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Quick start guide for new users
- **[User Guide](docs/user_guide.md)**: Complete usage instructions
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Examples](docs/examples.md)**: Comprehensive example workflows
- **[Installation Guide](docs/INSTALL.md)**: Step-by-step setup
- **[Project Summary](docs/PROJECT_SUMMARY.md)**: Complete project overview

## 🔬 Example Use Cases

### 1. Thermal Conductivity Study
```python
# Study multiple materials
materials = ["Si", "Al2O3", "Fe"]
for material in materials:
    result = agent.run_simulation(
        f"Simulate {material} thermal conductivity at 300 K"
    )
    analysis = agent.analyze_results(result["simulation_directory"])
```

### 2. Machine Learning Property Prediction
```python
# Train ML model
train_result = agent.tools[3].train_property_predictor(
    training_data="data.csv",
    target_property="elastic_modulus"
)

# Make predictions
prediction = agent.tools[3].predict_property(
    model_name=train_result["model_name"],
    features=[0.8, 0.1, 0.1, 300, 1.0, 100]
)
```

### 3. Database Integration
```python
# Query Materials Project
mp_data = agent.tools[2].query_materials_project("Si")

# Search by structure
similar = agent.tools[2].search_by_structure(structure_dict)
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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

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


## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/materials-ai-agent/materials-ai-agent/issues)
- **Documentation**: [Complete guides and API reference](docs/)
- **Community**: [GitHub Discussions](https://github.com/materials-ai-agent/materials-ai-agent/discussions)

## 🙏 Acknowledgments

- **LAMMPS**: Molecular dynamics engine
- **ASE**: Atomic simulation environment
- **PyMatGen**: Materials informatics
- **LangChain**: LLM orchestration
- **OpenAI**: GPT-4 integration

## 📈 Roadmap

- [ ] Additional MD engines (GROMACS, HOOMD-blue)
- [ ] Advanced ML models (Graph neural networks)
- [ ] Web interface
- [ ] Cloud deployment
- [ ] Real-time collaboration

---

**Made with ❤️ for the materials science community**
