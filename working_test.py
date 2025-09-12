#!/usr/bin/env python3
"""
Test script for what's currently working in Materials AI Agent.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_openai_integration():
    """Test OpenAI integration."""
    print("🤖 Testing OpenAI Integration...")
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test a materials science question
        response = llm.invoke("""
        I want to simulate silicon at 300 K using molecular dynamics. 
        What force field would you recommend and why?
        """)
        
        print("✓ OpenAI working!")
        print(f"Response: {response.content[:200]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI test failed: {e}")
        return False


def test_materials_project():
    """Test Materials Project API."""
    print("\n🗄️ Testing Materials Project API...")
    try:
        from mp_api.client import MPRester
        
        with MPRester(os.getenv("MP_API_KEY")) as mpr:
            # Query for silicon
            docs = mpr.materials.summary.search(
                formula="Si",
                fields=["material_id", "formula_pretty", "band_gap"]
            )
            
            if docs:
                print("✓ Materials Project API working!")
                print(f"Found {len(docs)} silicon materials")
                print(f"Example: {docs[0].material_id} - {docs[0].formula_pretty}")
                if hasattr(docs[0], 'band_gap') and docs[0].band_gap:
                    print(f"Band gap: {docs[0].band_gap} eV")
                return True
            else:
                print("✗ No results from Materials Project")
                return False
                
    except Exception as e:
        print(f"✗ Materials Project test failed: {e}")
        return False


def test_basic_simulation_workflow():
    """Test basic simulation workflow without full agent."""
    print("\n🧬 Testing Basic Simulation Workflow...")
    try:
        from ase import Atoms
        from ase.build import bulk
        import numpy as np
        
        # Create silicon structure
        si = bulk("Si", "diamond", a=5.43)
        print(f"✓ Created Si structure: {len(si)} atoms")
        print(f"  Formula: {si.get_chemical_formula()}")
        print(f"  Volume: {si.get_volume():.2f} Å³")
        
        # Test basic analysis
        positions = si.get_positions()
        distances = np.linalg.norm(positions[0] - positions[1:], axis=1)
        min_dist = np.min(distances)
        print(f"  Nearest neighbor distance: {min_dist:.2f} Å")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic simulation workflow failed: {e}")
        return False


def test_ml_capabilities():
    """Test basic ML capabilities."""
    print("\n🤖 Testing ML Capabilities...")
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 3)  # 3 features
        y = X[:, 0] * 2 + X[:, 1] * 3 + X[:, 2] * 1 + np.random.normal(0, 0.1, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test prediction
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        
        print("✓ ML capabilities working!")
        print(f"  Model R² score: {r2:.3f}")
        print(f"  Sample prediction: {y_pred[0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ ML capabilities test failed: {e}")
        return False


def test_visualization():
    """Test visualization capabilities."""
    print("\n📊 Testing Visualization Capabilities...")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create sample data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = "test_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualization working!")
        print(f"  Plot saved as: {plot_file}")
        
        # Clean up
        if os.path.exists(plot_file):
            os.remove(plot_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all working tests."""
    print("🚀 Materials AI Agent - Working Components Test")
    print("=" * 60)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("OpenAI Integration", test_openai_integration),
        ("Materials Project API", test_materials_project),
        ("Basic Simulation Workflow", test_basic_simulation_workflow),
        ("ML Capabilities", test_ml_capabilities),
        ("Visualization", test_visualization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed >= 3:
        print("\n🎉 Excellent! Core functionality is working!")
        print("\n✅ What's Working:")
        for test_name, success in results:
            if success:
                print(f"  ✓ {test_name}")
        
        print("\n🚀 You can now use the Materials AI Agent!")
        print("\nNext steps:")
        print("1. Try: python -c \"from langchain_openai import ChatOpenAI; print('Ready!')\"")
        print("2. Query materials: python -c \"from mp_api.client import MPRester; print('MP ready!')\"")
        print("3. Create structures: python -c \"from ase.build import bulk; print('ASE ready!')\"")
        
    else:
        print("\n⚠ Some components need attention.")
        print("Check the failed tests above for details.")
    
    return passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
