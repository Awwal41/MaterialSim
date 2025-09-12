"""Machine learning training example using Materials AI Agent."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_ai_agent import MaterialsAgent, Config


def generate_training_data():
    """Generate synthetic training data for demonstration."""
    print("Generating synthetic training data...")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Features: composition, temperature, pressure, volume
    data = {
        'Si_fraction': np.random.uniform(0, 1, n_samples),
        'Al_fraction': np.random.uniform(0, 1, n_samples),
        'O_fraction': np.random.uniform(0, 1, n_samples),
        'temperature': np.random.uniform(200, 800, n_samples),
        'pressure': np.random.uniform(0.1, 10, n_samples),
        'volume': np.random.uniform(50, 200, n_samples),
    }
    
    # Target: synthetic thermal conductivity
    # Based on simple physical relationships
    thermal_conductivity = (
        100 * data['Si_fraction'] +  # Silicon contribution
        50 * data['Al_fraction'] +   # Aluminum contribution
        20 * data['O_fraction'] +    # Oxygen contribution
        0.1 * data['temperature'] +  # Temperature dependence
        -0.5 * data['pressure'] +    # Pressure dependence
        -0.2 * data['volume'] +      # Volume dependence
        np.random.normal(0, 10, n_samples)  # Noise
    )
    
    data['thermal_conductivity'] = thermal_conductivity
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = Path("training_data.csv")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Training data saved to: {output_file}")
    return str(output_file)


def main():
    """Run ML training example."""
    print("Materials AI Agent - ML Training Example")
    print("=" * 50)
    
    # Generate training data
    training_file = generate_training_data()
    
    # Initialize agent
    print("\nInitializing Materials AI Agent...")
    agent = MaterialsAgent()
    
    # Train ML model
    print("\n1. Training thermal conductivity predictor...")
    train_result = agent.tools[3].train_property_predictor(
        training_data=training_file,
        target_property="thermal_conductivity",
        model_type="random_forest"
    )
    
    if train_result["success"]:
        print("✓ Model training completed!")
        print(f"Model: {train_result['model_name']}")
        print(f"Test R²: {train_result['test_r2']:.3f}")
        print(f"Test MSE: {train_result['test_mse']:.3f}")
        
        # Make predictions
        print("\n2. Making predictions...")
        
        # Test cases
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
                print(f"Test case {i+1}: {pred_result['prediction']:.2f} W/m·K "
                      f"(uncertainty: {pred_result['uncertainty']:.2f})")
            else:
                print(f"Test case {i+1}: Prediction failed - {pred_result['error']}")
        
        # Train neural network
        print("\n3. Training neural network...")
        nn_result = agent.tools[3].train_neural_network(
            training_data=training_file,
            target_property="thermal_conductivity",
            hidden_layers=[64, 32],
            epochs=50
        )
        
        if nn_result["success"]:
            print("✓ Neural network training completed!")
            print(f"Test R²: {nn_result['test_r2']:.3f}")
            print(f"Test MSE: {nn_result['test_mse']:.3f}")
        
        # Uncertainty quantification
        print("\n4. Uncertainty quantification...")
        uq_result = agent.tools[3].predict_with_uncertainty(
            model_name=train_result["model_name"],
            features=test_cases[0],
            n_samples=100
        )
        
        if uq_result["success"]:
            print(f"Prediction: {uq_result['prediction']:.2f} W/m·K")
            print(f"Uncertainty: {uq_result['uncertainty']:.2f} W/m·K")
            print(f"95% CI: [{uq_result['confidence_interval']['lower']:.2f}, "
                  f"{uq_result['confidence_interval']['upper']:.2f}] W/m·K")
        
        # List available models
        print("\n5. Available models:")
        models_result = agent.tools[3].list_available_models()
        if models_result["success"]:
            for model in models_result["models"]:
                print(f"  - {model['name']} ({model['type']})")
    
    else:
        print(f"✗ Model training failed: {train_result['error']}")
    
    print("\nML training example completed!")


if __name__ == "__main__":
    main()
