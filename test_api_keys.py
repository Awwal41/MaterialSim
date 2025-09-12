#!/usr/bin/env python3
"""
Test script to verify API keys are working correctly.
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
        
        response = llm.invoke("Hello, are you working?")
        print(f"✓ OpenAI API working! Response: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI API test failed: {e}")
        return False


def test_materials_project_api():
    """Test Materials Project API key."""
    print("Testing Materials Project API key...")
    try:
        from mp_api.client import MPRester
        
        # Load API key from environment
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            print("✗ MP_API_KEY not found in environment")
            return False
        
        # Test with a simple query
        with MPRester(api_key) as mpr:
            # Query for silicon
            docs = mpr.materials.summary.search(
                formula="Si",
                fields=["material_id", "formula_pretty"]
            )
            
            if docs:
                print(f"✓ Materials Project API working! Found {len(docs)} silicon materials")
                print(f"  Example: {docs[0].material_id} - {docs[0].formula_pretty}")
                return True
            else:
                print("✗ No results returned from Materials Project API")
                return False
                
    except Exception as e:
        print(f"✗ Materials Project API test failed: {e}")
        return False


def test_agent_initialization():
    """Test Materials AI Agent initialization."""
    print("Testing Materials AI Agent initialization...")
    try:
        from materials_ai_agent import MaterialsAgent
        
        # Initialize agent
        agent = MaterialsAgent()
        print("✓ Materials AI Agent initialized successfully!")
        
        # Test tools
        print(f"✓ {len(agent.tools)} tools loaded:")
        for i, tool in enumerate(agent.tools):
            print(f"  {i+1}. {tool.name}: {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"✗ Materials AI Agent initialization failed: {e}")
        return False


def main():
    """Run all API key tests."""
    print("🔑 Materials AI Agent - API Key Test")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("OpenAI API", test_openai_api),
        ("Materials Project API", test_materials_project_api),
        ("Materials AI Agent", test_agent_initialization),
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
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Materials AI Agent is ready to use!")
        print("\nNext steps:")
        print("1. Run: materials-agent interactive")
        print("2. Or try: python examples/basic_simulation.py")
    else:
        print("\n⚠ Some tests failed. Please check your API keys and configuration.")
        print("Make sure your .env file contains valid API keys.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
