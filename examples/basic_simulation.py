"""Basic simulation example using Materials AI Agent."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_ai_agent import MaterialsAgent, Config


def main():
    """Run a basic simulation example."""
    print("Materials AI Agent - Basic Simulation Example")
    print("=" * 50)
    
    # Initialize agent
    print("Initializing Materials AI Agent...")
    agent = MaterialsAgent()
    
    # Example 1: Run a simple simulation
    print("\n1. Running silicon simulation...")
    result = agent.run_simulation(
        "Simulate silicon at 300 K using Tersoff potential for 10000 steps"
    )
    
    if result["success"]:
        print("✓ Simulation completed successfully!")
        print(f"Result: {result['result']}")
    else:
        print(f"✗ Simulation failed: {result['error']}")
    
    # Example 2: Analyze results
    print("\n2. Analyzing simulation results...")
    if result["success"] and "simulation_directory" in result["result"]:
        sim_dir = result["result"]["simulation_directory"]
        analysis_result = agent.analyze_results(sim_dir)
        
        if analysis_result["success"]:
            print("✓ Analysis completed successfully!")
            print(f"Analysis: {analysis_result['analysis']}")
        else:
            print(f"✗ Analysis failed: {analysis_result['error']}")
    
    # Example 3: Query database
    print("\n3. Querying Materials Project database...")
    db_result = agent.query_database("silicon band gap and formation energy")
    
    if db_result["success"]:
        print("✓ Database query completed!")
        print(f"Results: {db_result['results']}")
    else:
        print(f"✗ Database query failed: {db_result['error']}")
    
    # Example 4: Predict properties
    print("\n4. Predicting material properties...")
    pred_result = agent.predict_properties("Al2O3", ["elastic_modulus", "thermal_conductivity"])
    
    if pred_result["success"]:
        print("✓ Property prediction completed!")
        print(f"Predictions: {pred_result['predictions']}")
    else:
        print(f"✗ Property prediction failed: {pred_result['error']}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
