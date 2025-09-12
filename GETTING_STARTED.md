# Getting Started with Materials AI Agent

## 🚀 Quick Start (5 minutes)

Your Materials AI Agent is pre-configured with API keys and ready to use!

### Option 1: Super Quick Setup
```bash
# Run the quick setup script
python quick_setup.py

# Test that everything works
python test_api_keys.py

# Start using the agent
materials-agent interactive
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Copy pre-configured settings
cp config_with_keys.env .env

# Create directories
mkdir -p simulations analysis visualizations models data logs

# Test the installation
python test_api_keys.py
```

## 🧪 Test Your Installation

Run the API key test to verify everything is working:

```bash
python test_api_keys.py
```

You should see:
```
🔑 Materials AI Agent - API Key Test
==================================================

==================== OpenAI API ====================
✓ OpenAI API working! Response: Hello! Yes, I'm working and ready to help...

==================== Materials Project API ====================
✓ Materials Project API working! Found 1 silicon materials
  Example: mp-149 - Si

==================== Materials AI Agent ====================
✓ Materials AI Agent initialized successfully!
✓ 5 tools loaded:
  1. simulation: Set up and run molecular dynamics simulations using LAMMPS
  2. analysis: Analyze simulation results to compute materials properties
  3. database: Query materials databases for properties and structures
  4. ml: Machine learning tools for property prediction and model training
  5. visualization: Create visualizations and reports for simulation results

==================================================
📊 Test Results Summary
==================================================
OpenAI API: ✓ PASS
Materials Project API: ✓ PASS
Materials AI Agent: ✓ PASS

Overall: 3/3 tests passed

🎉 All tests passed! Your Materials AI Agent is ready to use!
```

## 🎯 Your First Simulation

### Interactive Mode
```bash
materials-agent interactive
```

Then try:
```
> Simulate silicon at 300 K using Tersoff potential
> Analyze the results
> Query the database for silicon properties
```

### Python API
```python
from materials_ai_agent import MaterialsAgent

# Initialize the agent
agent = MaterialsAgent()

# Run a simulation
result = agent.run_simulation(
    "Simulate silicon at 300 K using Tersoff potential for 10000 steps"
)

print(result)
```

### Command Line
```bash
# Run a simulation
materials-agent run "Simulate silicon thermal conductivity at 300 K"

# Analyze results
materials-agent analyze ./simulations/silicon_300K/

# Query database
materials-agent query "silicon band gap"
```

## 📚 Example Workflows

### 1. Basic Silicon Simulation
```bash
python examples/basic_simulation.py
```

### 2. Machine Learning Training
```bash
python examples/ml_training_example.py
```

### 3. Advanced Multi-Material Study
```bash
python examples/advanced_workflow.py
```

## 🔧 Configuration

Your API keys are pre-configured in:
- `config_with_keys.env` - Environment variables
- `config_with_keys.yaml` - YAML configuration

To modify settings, edit the `.env` file:
```bash
nano .env
```

## 🐛 Troubleshooting

### Common Issues

1. **"LAMMPS not found"**
   ```bash
   # Install LAMMPS
   sudo apt-get install lammps  # Ubuntu/Debian
   brew install lammps          # macOS
   
   # Or set the path in .env
   echo "LAMMPS_EXECUTABLE=/path/to/lammps" >> .env
   ```

2. **"API key errors"**
   ```bash
   # Check your .env file
   cat .env
   
   # Re-copy the configuration
   cp config_with_keys.env .env
   ```

3. **"Import errors"**
   ```bash
   # Make sure you're in the right environment
   pip list | grep materials-ai-agent
   
   # Reinstall if needed
   pip install -e .
   ```

### Get Help

- **Test API Keys**: `python test_api_keys.py`
- **Check Logs**: Look in `logs/` directory
- **Run Examples**: Try the example scripts
- **Documentation**: See `docs/user_guide.md`

## 🎉 You're Ready!

Your Materials AI Agent is configured and ready to use. The system includes:

✅ **Pre-configured API keys** (OpenAI + Materials Project)  
✅ **Complete tool suite** (Simulation, Analysis, ML, Database, Visualization)  
✅ **Example workflows** ready to run  
✅ **Comprehensive documentation**  
✅ **Test suite** to verify functionality  

### Next Steps

1. **Try the interactive mode**: `materials-agent interactive`
2. **Run an example**: `python examples/basic_simulation.py`
3. **Read the docs**: `docs/user_guide.md`
4. **Explore the API**: `docs/api_reference.md`

Happy simulating! 🧬⚗️🔬
