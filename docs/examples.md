# Materials AI Agent - Examples

## Table of Contents
1. [Basic Simulation](#basic-simulation)
2. [Property Analysis](#property-analysis)
3. [Machine Learning](#machine-learning)
4. [Database Integration](#database-integration)
5. [Advanced Workflows](#advanced-workflows)
6. [HPC Integration](#hpc-integration)

## Basic Simulation

### Simple Silicon Simulation

```python
from materials_ai_agent import MaterialsAgent

# Initialize agent
agent = MaterialsAgent()

# Run simulation
result = agent.run_simulation(
    "Simulate silicon at 300 K using Tersoff potential for 10000 steps"
)

if result["success"]:
    print("Simulation completed successfully!")
    print(f"Results: {result['result']}")
else:
    print(f"Simulation failed: {result['error']}")
```

### Custom Simulation Parameters

```python
# Set up simulation with custom parameters
sim_result = agent.tools[0].setup_simulation(
    material="Al2O3",
    temperature=500,
    pressure=5.0,
    n_steps=50000,
    ensemble="NPT",
    force_field="eam"
)

if sim_result["success"]:
    # Run the simulation
    run_result = agent.tools[0].run_simulation(
        sim_result["simulation_directory"]
    )
    
    if run_result["success"]:
        print("Custom simulation completed!")
```

### Multiple Materials

```python
materials = ["Si", "Ge", "C"]
results = {}

for material in materials:
    print(f"Simulating {material}...")
    
    result = agent.run_simulation(
        f"Simulate {material} at 300 K using appropriate force field"
    )
    
    if result["success"]:
        results[material] = result["result"]
        print(f"✓ {material} simulation completed")
    else:
        print(f"✗ {material} simulation failed: {result['error']}")

print(f"Completed {len(results)} simulations")
```

## Property Analysis

### Radial Distribution Function

```python
# Compute RDF
rdf_result = agent.tools[1].compute_radial_distribution_function(
    trajectory_file="trajectory.dump",
    atom_types=["Si"],  # Only Si-Si correlations
    r_max=15.0,
    n_bins=300
)

if rdf_result["success"]:
    print("RDF computed successfully!")
    print(f"Plot saved to: {rdf_result['plot_file']}")
    
    # Access RDF data
    r = rdf_result["rdf_data"]["r"]
    g_r = rdf_result["rdf_data"]["g_r"]
    
    # Find first peak
    first_peak_idx = np.argmax(g_r[g_r > 0])
    first_peak_r = r[first_peak_idx]
    print(f"First peak at r = {first_peak_r:.2f} Å")
```

### Mean Squared Displacement

```python
# Compute MSD
msd_result = agent.tools[1].compute_mean_squared_displacement(
    trajectory_file="trajectory.dump",
    atom_types=["Si"]
)

if msd_result["success"]:
    print("MSD computed successfully!")
    
    # Get diffusion coefficient
    if msd_result["msd_data"]["diffusion_coefficient"]:
        D = msd_result["msd_data"]["diffusion_coefficient"]
        print(f"Diffusion coefficient: {D:.2e} cm²/s")
```

### Thermodynamic Analysis

```python
# Analyze thermodynamic properties
thermo_result = agent.tools[1].analyze_thermodynamic_properties(
    log_file="log.lammps"
)

if thermo_result["success"]:
    properties = thermo_result["properties"]
    
    print("Thermodynamic Properties:")
    print(f"Average Temperature: {properties['average_temperature']:.2f} K")
    print(f"Average Pressure: {properties['average_pressure']:.2f} atm")
    print(f"Average Volume: {properties['average_volume']:.2f} Å³")
    print(f"Average Energy: {properties['average_total_energy']:.2f} eV")
```

## Machine Learning

### Training a Property Predictor

```python
import pandas as pd
import numpy as np

# Generate training data
def generate_training_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Si_fraction': np.random.uniform(0, 1, n_samples),
        'Al_fraction': np.random.uniform(0, 1, n_samples),
        'O_fraction': np.random.uniform(0, 1, n_samples),
        'temperature': np.random.uniform(200, 800, n_samples),
        'pressure': np.random.uniform(0.1, 10, n_samples),
        'volume': np.random.uniform(50, 200, n_samples),
    }
    
    # Synthetic thermal conductivity
    thermal_conductivity = (
        100 * data['Si_fraction'] +
        50 * data['Al_fraction'] +
        20 * data['O_fraction'] +
        0.1 * data['temperature'] +
        -0.5 * data['pressure'] +
        -0.2 * data['volume'] +
        np.random.normal(0, 10, n_samples)
    )
    
    data['thermal_conductivity'] = thermal_conductivity
    
    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    return "training_data.csv"

# Generate data
training_file = generate_training_data()

# Train model
train_result = agent.tools[3].train_property_predictor(
    training_data=training_file,
    target_property="thermal_conductivity",
    model_type="random_forest",
    test_size=0.2
)

if train_result["success"]:
    print(f"Model trained successfully!")
    print(f"Test R²: {train_result['test_r2']:.3f}")
    print(f"Test MSE: {train_result['test_mse']:.3f}")
```

### Making Predictions

```python
# Make predictions
test_cases = [
    [0.8, 0.1, 0.1, 300, 1.0, 100],  # Silicon-rich
    [0.1, 0.6, 0.3, 400, 2.0, 120],  # Aluminum oxide
    [0.5, 0.3, 0.2, 500, 1.5, 90],   # Mixed composition
]

for i, features in enumerate(test_cases):
    pred_result = agent.tools[3].predict_property(
        model_name=train_result["model_name"],
        features=features
    )
    
    if pred_result["success"]:
        print(f"Test case {i+1}: {pred_result['prediction']:.2f} W/m·K")
```

### Neural Network Training

```python
# Train neural network
nn_result = agent.tools[3].train_neural_network(
    training_data=training_file,
    target_property="thermal_conductivity",
    hidden_layers=[128, 64, 32],
    learning_rate=0.001,
    epochs=100,
    batch_size=32
)

if nn_result["success"]:
    print(f"Neural network trained successfully!")
    print(f"Test R²: {nn_result['test_r2']:.3f}")
    print(f"Test MSE: {nn_result['test_mse']:.3f}")
```

### Uncertainty Quantification

```python
# Predict with uncertainty
uq_result = agent.tools[3].predict_with_uncertainty(
    model_name=train_result["model_name"],
    features=test_cases[0],
    n_samples=1000
)

if uq_result["success"]:
    print(f"Prediction: {uq_result['prediction']:.2f} W/m·K")
    print(f"Uncertainty: {uq_result['uncertainty']:.2f} W/m·K")
    print(f"95% CI: [{uq_result['confidence_interval']['lower']:.2f}, "
          f"{uq_result['confidence_interval']['upper']:.2f}] W/m·K")
```

## Database Integration

### Querying Materials Project

```python
# Query Materials Project
mp_result = agent.tools[2].query_materials_project(
    formula="Si",
    properties=["band_gap", "formation_energy_per_atom", "density"]
)

if mp_result["success"]:
    print(f"Found {mp_result['n_materials']} materials")
    
    for material in mp_result["materials"]:
        print(f"Material ID: {material['material_id']}")
        print(f"Formula: {material['formula_pretty']}")
        print(f"Band Gap: {material.get('band_gap', 'N/A')} eV")
        print(f"Formation Energy: {material.get('formation_energy_per_atom', 'N/A')} eV/atom")
        print(f"Density: {material.get('density', 'N/A')} g/cm³")
        print("-" * 40)
```

### Elastic Properties

```python
# Get elastic properties
elastic_result = agent.tools[2].get_elastic_properties(
    material_id="mp-149"  # Silicon
)

if elastic_result["success"]:
    print("Elastic Properties:")
    print(f"Bulk Modulus: {elastic_result['bulk_modulus']:.2f} GPa")
    print(f"Shear Modulus: {elastic_result['shear_modulus']:.2f} GPa")
    print(f"Young's Modulus: {elastic_result['young_modulus']:.2f} GPa")
    print(f"Poisson's Ratio: {elastic_result['poisson_ratio']:.3f}")
```

### Structure Search

```python
# Search by structure
structure_dict = {
    "cell": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
    "positions": [[0, 0, 0], [1.3575, 1.3575, 1.3575]],
    "numbers": [14, 14]  # Silicon
}

search_result = agent.tools[2].search_by_structure(
    structure=structure_dict,
    tolerance=0.1
)

if search_result["success"]:
    print(f"Found {search_result['n_similar']} similar structures")
    
    for structure in search_result["similar_structures"]:
        print(f"Material: {structure['formula']}")
        print(f"Energy: {structure['energy_per_atom']:.3f} eV/atom")
```

### NOMAD Query

```python
# Query NOMAD database
nomad_result = agent.tools[2].query_nomad(
    query="silicon thermal conductivity",
    max_results=5
)

if nomad_result["success"]:
    print(f"Found {nomad_result['n_results']} results from NOMAD")
    
    for result in nomad_result["results"]:
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Authors: {result.get('authors', 'N/A')}")
        print(f"DOI: {result.get('doi', 'N/A')}")
        print("-" * 40)
```

## Advanced Workflows

### Complete Materials Study

```python
def study_material(material, temperature, pressure):
    """Complete study of a material."""
    print(f"Studying {material} at {temperature} K, {pressure} atm")
    
    # 1. Query database for reference data
    print("1. Querying database...")
    db_result = agent.query_database(f"{material} properties at {temperature} K")
    
    # 2. Run simulation
    print("2. Running simulation...")
    sim_result = agent.run_simulation(
        f"Simulate {material} at {temperature} K and {pressure} atm"
    )
    
    if not sim_result["success"]:
        return {"error": sim_result["error"]}
    
    # 3. Analyze results
    print("3. Analyzing results...")
    analysis_result = agent.analyze_results(
        sim_result["result"]["simulation_directory"]
    )
    
    # 4. Predict properties with ML
    print("4. Predicting properties...")
    pred_result = agent.predict_properties(
        material, ["thermal_conductivity", "elastic_modulus"]
    )
    
    # 5. Compare with database
    print("5. Comparing with database...")
    comparison_result = agent.tools[2].compare_with_database(
        properties=analysis_result.get("properties", {}),
        material_formula=material
    )
    
    return {
        "database": db_result,
        "simulation": sim_result,
        "analysis": analysis_result,
        "prediction": pred_result,
        "comparison": comparison_result
    }

# Study multiple materials
materials = [
    ("Si", 300, 1.0),
    ("Al2O3", 500, 1.0),
    ("Fe", 300, 1.0)
]

results = {}
for material, temp, press in materials:
    results[material] = study_material(material, temp, press)
```

### High-Throughput Screening

```python
def screen_materials(material_list, property_target):
    """Screen multiple materials for a target property."""
    results = []
    
    for material in material_list:
        print(f"Screening {material}...")
        
        # Quick property prediction
        pred_result = agent.predict_properties(
            material, [property_target]
        )
        
        if pred_result["success"]:
            value = pred_result["predictions"].get(property_target)
            if value and value > threshold:  # Define threshold
                results.append({
                    "material": material,
                    "property": property_target,
                    "value": value,
                    "promising": True
                })
            else:
                results.append({
                    "material": material,
                    "property": property_target,
                    "value": value,
                    "promising": False
                })
    
    return results

# Screen for high thermal conductivity
materials = ["Si", "Ge", "C", "Al2O3", "Fe", "Cu", "Ag", "Au"]
screening_results = screen_materials(materials, "thermal_conductivity")

promising_materials = [r for r in screening_results if r["promising"]]
print(f"Found {len(promising_materials)} promising materials")
```

### Transfer Learning

```python
def transfer_learning_workflow():
    """Demonstrate transfer learning between materials."""
    
    # 1. Train on silicon data
    print("Training on silicon data...")
    si_train_result = agent.tools[3].train_property_predictor(
        training_data="silicon_training_data.csv",
        target_property="thermal_conductivity",
        model_type="neural_network"
    )
    
    # 2. Fine-tune on germanium data
    print("Fine-tuning on germanium data...")
    ge_finetune_result = agent.tools[3].train_neural_network(
        training_data="germanium_training_data.csv",
        target_property="thermal_conductivity",
        hidden_layers=[100, 50],  # Same architecture
        learning_rate=0.0001,     # Lower learning rate
        epochs=50
    )
    
    # 3. Test on carbon
    print("Testing on carbon...")
    c_pred_result = agent.tools[3].predict_property(
        model_name=ge_finetune_result["model_name"],
        features=get_carbon_features()
    )
    
    return {
        "silicon_model": si_train_result,
        "germanium_model": ge_finetune_result,
        "carbon_prediction": c_pred_result
    }
```

## HPC Integration

### SLURM Job Submission

```python
def submit_hpc_job(material, temperature, n_cores=32):
    """Submit simulation job to HPC cluster."""
    
    job_script = f"""#!/bin/bash
#SBATCH --job-name={material}_{temperature}K
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={n_cores}
#SBATCH --time=24:00:00
#SBATCH --partition=compute

module load lammps
module load python/3.9

cd $SLURM_SUBMIT_DIR

python -c "
from materials_ai_agent import MaterialsAgent
agent = MaterialsAgent()
result = agent.run_simulation('Simulate {material} at {temperature} K')
print(result)
"
"""
    
    with open(f"{material}_{temperature}K.slurm", "w") as f:
        f.write(job_script)
    
    # Submit job
    import subprocess
    result = subprocess.run(
        ["sbatch", f"{material}_{temperature}K.slurm"],
        capture_output=True, text=True
    )
    
    return result.stdout

# Submit multiple jobs
materials = ["Si", "Ge", "C"]
temperatures = [300, 500, 700]

for material in materials:
    for temp in temperatures:
        job_id = submit_hpc_job(material, temp)
        print(f"Submitted {material} at {temp}K: {job_id}")
```

### Parallel Analysis

```python
import concurrent.futures
from pathlib import Path

def analyze_simulation(sim_dir):
    """Analyze a single simulation directory."""
    return agent.analyze_results(str(sim_dir))

def parallel_analysis(simulation_dirs):
    """Analyze multiple simulations in parallel."""
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_simulation, sim_dir): sim_dir 
            for sim_dir in simulation_dirs
        }
        
        results = {}
        for future in concurrent.futures.as_completed(futures):
            sim_dir = futures[future]
            try:
                result = future.result()
                results[str(sim_dir)] = result
                print(f"✓ Analyzed {sim_dir}")
            except Exception as e:
                print(f"✗ Failed to analyze {sim_dir}: {e}")
                results[str(sim_dir)] = {"error": str(e)}
    
    return results

# Find all simulation directories
sim_dirs = list(Path("simulations").glob("*/"))
analysis_results = parallel_analysis(sim_dirs)
```

### Resource Monitoring

```python
def monitor_resources():
    """Monitor computational resources during simulation."""
    import psutil
    import time
    
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_usage}%")
        
        if cpu_percent > 90 or memory_percent > 90:
            print("⚠ High resource usage detected!")
        
        time.sleep(60)  # Check every minute

# Run monitoring in background
import threading
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.daemon = True
monitor_thread.start()
```

These examples demonstrate the full capabilities of the Materials AI Agent, from basic simulations to advanced workflows and HPC integration. Each example can be adapted and extended for specific research needs.
