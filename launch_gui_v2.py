#!/usr/bin/env python3
"""
Launcher script for Materials AI Agent Enhanced GUI.
This script launches the conversational simulation interface.
"""

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

def install_gui_dependencies():
    """Install GUI dependencies."""
    print("Installing GUI dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_gui.txt"
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def launch_gui():
    """Launch the Streamlit GUI."""
    print("Launching Materials AI Agent Enhanced GUI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "gui_app_v2.py",
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
    print("🚀 Materials AI Agent - Enhanced GUI Launcher")
    print("=" * 60)
    print("Conversational Interface for Materials Simulation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("gui_app_v2.py").exists():
        print("❌ Error: gui_app_v2.py not found!")
        print("Please run this script from the Materials AI Agent directory.")
        return
    
    # Check Streamlit
    if not check_streamlit():
        print("📦 Streamlit not found. Installing GUI dependencies...")
        if not install_gui_dependencies():
            print("❌ Failed to install GUI dependencies.")
            print("Please install manually: pip install -r requirements_gui.txt")
            return
        print("✅ GUI dependencies installed!")
    
    # Launch GUI
    print("🌐 Starting conversational interface...")
    print("The GUI will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\n🎯 Features:")
    print("  • Step-by-step simulation guidance")
    print("  • Interactive chat interface")
    print("  • Real-time structure generation")
    print("  • Property calculation and visualization")
    print("  • File download and data export")
    print("\nPress Ctrl+C to stop the GUI.")
    print("-" * 60)
    
    launch_gui()

if __name__ == "__main__":
    main()
