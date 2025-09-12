#!/usr/bin/env python3
"""
Demonstration of working Materials AI Agent components.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt

def demo_openai_materials_advice():
    """Demonstrate OpenAI giving materials science advice."""
    print("🤖 OpenAI Materials Science Advisor")
    print("=" * 50)
    
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    question = """
    I want to simulate aluminum oxide (Al2O3) at 500 K using molecular dynamics.
    What would be the best approach? Please be specific about:
    1. Force field choice
    2. Simulation parameters
    3. Expected challenges
    """
    
    response = llm.invoke(question)
    print("Question:", question)
    print("\nAI Response:")
    print(response.content)
    print("\n" + "=" * 50)


def demo_structure_generation():
    """Demonstrate atomic structure generation."""
    print("\n🧬 Atomic Structure Generation")
    print("=" * 50)
    
    # Generate different materials
    materials = {
        "Silicon": bulk("Si", "diamond", a=5.43),
        "Aluminum": bulk("Al", "fcc", a=4.05),
        "Iron": bulk("Fe", "bcc", a=2.87),
    }
    
    for name, structure in materials.items():
        print(f"\n{name}:")
        print(f"  Formula: {structure.get_chemical_formula()}")
        print(f"  Atoms: {len(structure)}")
        print(f"  Volume: {structure.get_volume():.2f} Å³")
        print(f"  Density: {len(structure) / structure.get_volume():.3f} atoms/Å³")
        
        # Calculate nearest neighbor distance
        if len(structure) > 1:
            positions = structure.get_positions()
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            min_dist = min(distances)
            print(f"  Nearest neighbor: {min_dist:.2f} Å")


def demo_ml_property_prediction():
    """Demonstrate ML property prediction."""
    print("\n🤖 Machine Learning Property Prediction")
    print("=" * 50)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Create synthetic materials data
    np.random.seed(42)
    n_samples = 200
    
    # Features: composition, temperature, pressure
    X = np.random.rand(n_samples, 3)
    X[:, 0] *= 2  # Temperature range 0-2000 K
    X[:, 1] *= 10  # Pressure range 0-10 GPa
    X[:, 2] *= 1  # Composition parameter
    
    # Target: synthetic thermal conductivity
    thermal_conductivity = (
        100 * X[:, 2] +  # Composition effect
        0.1 * X[:, 0] +  # Temperature effect
        -5 * X[:, 1] +   # Pressure effect
        np.random.normal(0, 10, n_samples)  # Noise
    )
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, thermal_conductivity, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    
    print(f"Model Performance:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.2f}")
    
    # Show some predictions
    print(f"\nSample Predictions:")
    for i in range(5):
        temp, press, comp = X_test[i]
        actual = y_test[i]
        predicted = y_pred[i]
        print(f"  T={temp:.0f}K, P={press:.1f}GPa, Comp={comp:.2f}: "
              f"Actual={actual:.1f}, Predicted={predicted:.1f} W/m·K")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📊 Visualization Demo")
    print("=" * 50)
    
    # Create sample data
    x = np.linspace(0, 4*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    plt.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trigonometric Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y3, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin(x) × cos(x)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Simulate RDF data
    r = np.linspace(0, 10, 100)
    g_r = np.exp(-(r-2.5)**2/0.5) + 0.5*np.exp(-(r-4.5)**2/0.3)
    plt.plot(r, g_r, 'purple', linewidth=2)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Simulate MSD data
    t = np.linspace(0, 10, 50)
    msd = 0.1 * t + 0.01 * t**2
    plt.plot(t, msd, 'orange', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title('Mean Squared Displacement')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('materials_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated comprehensive materials science plots!")
    print("  Saved as: materials_demo.png")


def main():
    """Run all demonstrations."""
    print("🚀 Materials AI Agent - Working Components Demo")
    print("=" * 60)
    print("This demonstrates the core functionality that's currently working.")
    print("=" * 60)
    
    try:
        demo_openai_materials_advice()
        demo_structure_generation()
        demo_ml_property_prediction()
        demo_visualization()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n✅ What you just saw:")
        print("  🤖 AI-powered materials science advice")
        print("  🧬 Atomic structure generation and analysis")
        print("  🤖 Machine learning property prediction")
        print("  📊 Scientific visualization")
        
        print("\n🚀 Your Materials AI Agent is ready for:")
        print("  • Natural language materials queries")
        print("  • Structure generation and analysis")
        print("  • Property prediction with ML")
        print("  • Scientific plotting and visualization")
        
        print("\nNext steps:")
        print("1. Install Materials Project API: pip install mp-api")
        print("2. Try the full agent: python -c \"from materials_ai_agent import MaterialsAgent\"")
        print("3. Run simulations: materials-agent interactive")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your environment and try again.")


if __name__ == "__main__":
    main()
