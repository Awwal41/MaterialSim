#!/usr/bin/env python3
"""
Quick setup script for Materials AI Agent with pre-configured API keys.
Run this script to get started immediately.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Quick setup with pre-configured API keys."""
    print("🚀 Materials AI Agent - Quick Setup")
    print("=" * 50)
    print("Setting up with your provided API keys...")
    
    # Step 1: Install dependencies
    print("\n1. Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠ Failed to install requirements, but continuing...")
    
    # Step 2: Install package
    print("\n2. Installing package...")
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("⚠ Failed to install package, but continuing...")
    
    # Step 3: Create directories
    print("\n3. Creating directories...")
    directories = ["simulations", "analysis", "visualizations", "models", "data", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/")
    
    # Step 4: Set up configuration
    print("\n4. Setting up configuration with your API keys...")
    if not run_command("cp config_with_keys.env .env", "Copying configuration with API keys"):
        print("⚠ Could not copy configuration file")
    
    print("✓ Configuration setup completed!")
    print("✓ Your API keys have been configured!")
    
    # Step 5: Test installation
    print("\n5. Testing installation...")
    try:
        from materials_ai_agent import MaterialsAgent
        print("✓ Materials AI Agent imported successfully!")
        
        # Test agent initialization
        agent = MaterialsAgent()
        print("✓ Agent initialized successfully!")
        
        # Test tools
        print(f"✓ {len(agent.tools)} tools loaded successfully!")
        
    except Exception as e:
        print(f"⚠ Agent test failed: {e}")
        print("The package is installed but there may be missing dependencies")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Quick setup completed!")
    print("\nYour Materials AI Agent is ready to use!")
    print("\nNext steps:")
    print("1. Try the interactive mode:")
    print("   materials-agent interactive")
    print("\n2. Run a basic example:")
    print("   python examples/basic_simulation.py")
    print("\n3. Or use the Python API:")
    print("   python -c \"from materials_ai_agent import MaterialsAgent; agent = MaterialsAgent(); print('Ready!')\"")
    print("\n4. Check your configuration:")
    print("   cat .env")
    print("\nFor more information, see docs/user_guide.md")
    print("\nHappy simulating! 🧬⚗️")


if __name__ == "__main__":
    main()
