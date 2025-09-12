# Materials AI Agent - Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM (8 GB recommended)
- **Storage**: 2 GB free space
- **Internet**: Required for API access and package installation

### Recommended Requirements
- **Operating System**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: 3.9 or higher
- **Memory**: 16 GB RAM or more
- **Storage**: 10 GB free space
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU with CUDA support (optional, for ML acceleration)

### Required Software
- **LAMMPS**: For molecular dynamics simulations
- **Git**: For version control
- **pip**: Python package manager

## Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/materials-ai-agent/materials-ai-agent.git
cd materials-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Set up configuration
cp env.example .env
# Edit .env with your API keys

# Run build script
python build.py
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n materials-ai-agent python=3.9
conda activate materials-ai-agent

# Clone and install
git clone https://github.com/materials-ai-agent/materials-ai-agent.git
cd materials-ai-agent
pip install -e .

# Install LAMMPS
conda install -c conda-forge lammps

# Set up configuration
cp env.example .env
# Edit .env with your API keys
```

## Detailed Installation

### Step 1: Install LAMMPS

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install lammps
```

#### CentOS/RHEL
```bash
sudo yum install lammps
# or for newer versions
sudo dnf install lammps
```

#### macOS with Homebrew
```bash
brew install lammps
```

#### Windows
1. Download LAMMPS from [lammps.org](https://lammps.org/)
2. Extract to a directory (e.g., `C:\lammps`)
3. Add `C:\lammps\bin` to your PATH environment variable

#### Compile from Source
```bash
# Download LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps

# Configure and compile
mkdir build
cd build
cmake ../cmake -DCMAKE_INSTALL_PREFIX=/usr/local
make -j4
sudo make install
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 3: Install Optional Dependencies

#### For GPU acceleration (PyTorch with CUDA)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For advanced visualization
```bash
# Install additional visualization packages
pip install plotly dash jupyter
```

#### For HPC environments
```bash
# Install MPI support
pip install mpi4py
```

### Step 4: Set Up Configuration

```bash
# Copy example configuration
cp env.example .env

# Edit configuration file
nano .env 
```

Required configuration:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended
MP_API_KEY=your_materials_project_api_key
NOMAD_API_KEY=your_nomad_api_key

# LAMMPS configuration
LAMMPS_EXECUTABLE=lmp  # or lammps, lmp_serial, etc.

# Output directories
SIMULATION_OUTPUT_DIR=./simulations
ANALYSIS_OUTPUT_DIR=./analysis
VISUALIZATION_OUTPUT_DIR=./visualizations
```

### Step 5: Verify Installation

```bash
# Run build script
python build.py

# Test installation
materials-agent version

# Run basic test
python examples/basic_simulation.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-api-key
MP_API_KEY=your-materials-project-api-key
NOMAD_API_KEY=your-nomad-api-key

# LAMMPS Configuration
LAMMPS_EXECUTABLE=lmp
LAMMPS_MPI_EXECUTABLE=lmp_mpi

# Default Simulation Parameters
DEFAULT_TEMPERATURE=300
DEFAULT_PRESSURE=1.0
DEFAULT_TIMESTEP=0.001

# Output Directories
SIMULATION_OUTPUT_DIR=./simulations
ANALYSIS_OUTPUT_DIR=./analysis
VISUALIZATION_OUTPUT_DIR=./visualizations
MODEL_OUTPUT_DIR=./models

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/materials_agent.log

# LLM Configuration
MODEL_NAME=gpt-4
MAX_TOKENS=4000
TEMPERATURE=0.1

# ML Configuration
ML_MODEL_CACHE_DIR=./models/cache
ML_BATCH_SIZE=32
ML_LEARNING_RATE=0.001
```

### Configuration File (YAML)

Alternatively, create a `config.yaml` file:

```yaml
openai_api_key: "sk-your-openai-api-key"
mp_api_key: "your-materials-project-api-key"
lammps_executable: "lmp"
default_temperature: 300.0
default_pressure: 1.0
simulation_output_dir: "./simulations"
analysis_output_dir: "./analysis"
visualization_output_dir: "./visualizations"
log_level: "INFO"
model_name: "gpt-4"
max_tokens: 4000
temperature: 0.1
```

## Verification

### Test 1: Basic Functionality
```bash
# Test agent initialization
python -c "from materials_ai_agent import MaterialsAgent; agent = MaterialsAgent(); print('✓ Agent initialized successfully')"
```

### Test 2: LAMMPS Integration
```bash
# Test LAMMPS availability
python -c "from materials_ai_agent.tools import SimulationTool; tool = SimulationTool(); print('✓ LAMMPS integration working')"
```

### Test 3: Database Access
```bash
# Test database connectivity
python -c "from materials_ai_agent.tools import DatabaseTool; tool = DatabaseTool(); print('✓ Database tools initialized')"
```

### Test 4: Full Workflow
```bash
# Run example simulation
python examples/basic_simulation.py
```

### Test 5: CLI Interface
```bash
# Test command line interface
materials-agent --help
materials-agent interactive
```

## Troubleshooting

### Common Issues

#### 1. LAMMPS Not Found
**Error**: `LAMMPS executable not found`

**Solutions**:
- Check if LAMMPS is installed: `which lmp` or `which lammps`
- Set `LAMMPS_EXECUTABLE` in your `.env` file
- Add LAMMPS to your PATH: `export PATH=$PATH:/path/to/lammps/bin`

#### 2. API Key Errors
**Error**: `OpenAI API key not configured`

**Solutions**:
- Verify your API key in `.env` file
- Check if the key has proper permissions
- Ensure no extra spaces or quotes in the key

#### 3. Import Errors
**Error**: `ModuleNotFoundError: No module named 'materials_ai_agent'`

**Solutions**:
- Ensure virtual environment is activated
- Reinstall package: `pip install -e .`
- Check Python path: `python -c "import sys; print(sys.path)"`

#### 4. Memory Issues
**Error**: `MemoryError` or simulation crashes

**Solutions**:
- Reduce simulation size or number of steps
- Use smaller batch sizes for ML training
- Increase system memory or use swap

#### 5. Permission Errors
**Error**: `Permission denied` when creating directories

**Solutions**:
- Check directory permissions: `ls -la`
- Create directories manually: `mkdir -p simulations analysis visualizations`
- Run with appropriate permissions

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from materials_ai_agent import MaterialsAgent
agent = MaterialsAgent()
```

### Getting Help

1. **Check Logs**: Look in the `logs/` directory for error messages
2. **Run Tests**: `python -m pytest tests/ -v`
3. **Check Dependencies**: `pip list | grep -E "(numpy|pandas|ase|pymatgen)"`
4. **Verify LAMMPS**: `lmp -help` or `lammps -help`
5. **GitHub Issues**: Submit issues with detailed error messages

### Performance Optimization

#### For Large Simulations
```python
# Use MPI for parallel simulations
config = Config(
    lammps_executable="lmp_mpi",
    # ... other settings
)
```

#### For ML Training
```python
# Use GPU acceleration
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

#### For HPC Environments
```python
# Configure for HPC
config = Config(
    simulation_output_dir="/scratch/simulations",
    analysis_output_dir="/scratch/analysis",
    # ... other settings
)
```

## Next Steps

After successful installation:

1. **Read the User Guide**: `docs/user_guide.md`
2. **Try Examples**: Run scripts in `examples/` directory
3. **Explore API**: Check `docs/api_reference.md`
4. **Join Community**: GitHub discussions and issues
5. **Contribute**: Fork the repository and submit pull requests



