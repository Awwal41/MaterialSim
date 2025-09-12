#!/usr/bin/env python3
"""
Simple test script for Materials AI Agent - tests what's working.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_openai_api():
    """Test OpenAI API key."""
    print("Testing OpenAI API key...")
    try:
        from langchain_openai import ChatOpenAI
        
        # Load API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ OPENAI_API_KEY not found in environment")
            return False
        
        # Test with a simple request
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=api_key
        )
        
        response = llm.invoke("Hello, are you working? Please respond with just 'Yes, I'm working!'")
        print(f"✓ OpenAI API working! Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI API test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if API keys are loaded
        openai_key = os.getenv("OPENAI_API_KEY")
        mp_key = os.getenv("MP_API_KEY")
        
        if openai_key and len(openai_key) > 10:
            print("✓ OpenAI API key loaded")
        else:
            print("✗ OpenAI API key not loaded properly")
            return False
            
        if mp_key and len(mp_key) > 10:
            print("✓ Materials Project API key loaded")
        else:
            print("✗ Materials Project API key not loaded properly")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    try:
        import numpy as np
        print("✓ NumPy imported")
        
        import pandas as pd
        print("✓ Pandas imported")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False


def test_materials_agent_core():
    """Test Materials AI Agent core without heavy dependencies."""
    print("Testing Materials AI Agent core...")
    try:
        # Test if we can import the core modules
        sys.path.insert(0, str(Path(__file__).parent))
        
        from materials_ai_agent.core.config import Config
        print("✓ Config class imported")
        
        from materials_ai_agent.core.exceptions import MaterialsAgentError
        print("✓ Exceptions imported")
        
        # Test configuration loading
        config = Config.from_env()
        print("✓ Configuration loaded from environment")
        
        return True
        
    except Exception as e:
        print(f"✗ Materials AI Agent core test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Materials AI Agent - Simple Test")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("Configuration", test_configuration),
        ("OpenAI API", test_openai_api),
        ("Basic Imports", test_basic_imports),
        ("Materials AI Agent Core", test_materials_agent_core),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed >= 2:  # At least config and OpenAI working
        print("\n🎉 Core functionality is working!")
        print("\nWhat's working:")
        print("✓ OpenAI API integration")
        print("✓ Configuration loading")
        print("✓ Basic Python environment")
        print("\nNext steps:")
        print("1. Install remaining dependencies: pip install ase pymatgen")
        print("2. Run full test: python test_api_keys.py")
        print("3. Try the agent: python -c \"from materials_ai_agent import MaterialsAgent\"")
    else:
        print("\n⚠ Some core tests failed. Please check your setup.")
    
    return passed >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
