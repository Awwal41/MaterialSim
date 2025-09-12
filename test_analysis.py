#!/usr/bin/env python3
"""
Test script for MSD, RDF, and thermodynamic analysis functions
"""

import sys
import os
sys.path.append('.')

# Import the analysis functions
from gui_app import perform_msd_analysis, perform_rdf_analysis, perform_thermodynamic_analysis

def test_analysis_functions():
    """Test all analysis functions with the latest simulation data."""
    
    print("🧪 Testing Analysis Functions")
    print("=" * 50)
    
    # Test 1: MSD Analysis
    print("\n1. Testing MSD Analysis...")
    try:
        msd_result = perform_msd_analysis()
        print("✅ MSD Analysis Result:")
        print(msd_result[:200] + "..." if len(msd_result) > 200 else msd_result)
    except Exception as e:
        print(f"❌ MSD Analysis failed: {e}")
    
    # Test 2: RDF Analysis
    print("\n2. Testing RDF Analysis...")
    try:
        rdf_result = perform_rdf_analysis()
        print("✅ RDF Analysis Result:")
        print(rdf_result[:200] + "..." if len(rdf_result) > 200 else rdf_result)
    except Exception as e:
        print(f"❌ RDF Analysis failed: {e}")
    
    # Test 3: Thermodynamic Analysis
    print("\n3. Testing Thermodynamic Analysis...")
    try:
        thermo_result = perform_thermodynamic_analysis()
        print("✅ Thermodynamic Analysis Result:")
        print(thermo_result[:200] + "..." if len(thermo_result) > 200 else thermo_result)
    except Exception as e:
        print(f"❌ Thermodynamic Analysis failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Analysis Testing Complete!")

if __name__ == "__main__":
    test_analysis_functions()
