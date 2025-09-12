#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def check_agent():
    """Check if MaterialSim AI Agent can be imported."""
    try:
        from materials_ai_agent import MaterialsAgent
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        # Install core dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        # Install GUI dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_gui.txt"
        ], check=True)
        
        return True
    except subprocess.CalledProcessError:
        return False

def launch_gui():
    """Launch the Streamlit GUI."""
    print("Launching MaterialSim AI Agent GUI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "gui_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nGUI closed by user.")
    except Exception as e:
        print(f"Error launching GUI: {e}")

def main():
    """Main launcher function."""
    print("🚀 MaterialSim AI Agent")
    print("=" * 60)
    print("")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("gui_app.py").exists():
        print("❌ Error: gui_app.py not found!")
        print("Please run this script from the MaterialSim AI Agent directory.")
        return
    
    # Check Streamlit
    if not check_streamlit():
        print("📦 Streamlit not found. Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies.")
            print("Please install manually: pip install -r requirements_gui.txt")
            return
        print("✅ Dependencies installed!")
    
    # Check MaterialSim AI Agent
    if not check_agent():
        print("📦 MaterialSim AI Agent not found. Installing...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], check=True)
            print("✅ MaterialSim AI Agent installed!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install MaterialSim AI Agent.")
            print("Please install manually: pip install -e .")
            return
    
    # Launch GUI
    print("🌐 Starting interface...")
    print("The GUI will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\n🎯 Features:")
    print("  • Completely dynamic")
    print("  • Frontend interface")
    print("  • Full agent integration")
    print("  • Natural language interaction")
    print("  • All existing capabilities")
    print("\nPress Ctrl+C to stop the GUI.")
    print("-" * 60)
    
    launch_gui()

if __name__ == "__main__":
    main()
