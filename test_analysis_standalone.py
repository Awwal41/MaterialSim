#!/usr/bin/env python3
"""
Standalone test script for MSD, RDF, and thermodynamic analysis functions
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def test_msd_analysis():
    """Test MSD analysis on simulation results."""
    try:
        # Find simulation directory
        sim_dirs = glob.glob("simulations/Si_*K_*steps")
        if not sim_dirs:
            return "❌ No simulation directories found"
        
        sim_dir = sim_dirs[0]  # Use the first one found
        print(f"📁 Using simulation directory: {sim_dir}")
        
        # Look for trajectory file first, then fall back to structure file
        trajectory_file = os.path.join(sim_dir, "trajectory.xyz")
        structure_file = os.path.join(sim_dir, "structure.xyz")
        
        if os.path.exists(trajectory_file):
            data_file = trajectory_file
            file_type = "trajectory"
        elif os.path.exists(structure_file):
            data_file = structure_file
            file_type = "structure"
        else:
            return "❌ Neither trajectory.xyz nor structure.xyz found. Cannot compute MSD."
        
        print(f"📄 Reading data from: {data_file} ({file_type})")
        
        # Read trajectory data
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Extract positions for each timestep
        positions = []
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():  # Number of atoms line
                n_atoms = int(lines[i].strip())
                i += 1  # Skip comment line
                
                timestep_positions = []
                for j in range(n_atoms):
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 4:
                        # For custom format: element xu yu zu fx fy fz
                        # We want positions (xu, yu, zu) which are indices 1, 2, 3
                        timestep_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                if timestep_positions:
                    positions.append(np.array(timestep_positions))
                
                i += n_atoms
            else:
                i += 1
        
        print(f"📊 Found {len(positions)} timesteps")
        
        if len(positions) < 2:
            return f"❌ Not enough trajectory data for MSD analysis. Need at least 2 timesteps, got {len(positions)}"
        
        # Calculate MSD
        positions = np.array(positions)
        n_timesteps, n_atoms, _ = positions.shape
        
        # Calculate MSD for each atom
        msd_values = []
        for atom_idx in range(n_atoms):
            atom_positions = positions[:, atom_idx, :]
            initial_pos = atom_positions[0]
            
            msd_atom = []
            for t in range(n_timesteps):
                displacement = atom_positions[t] - initial_pos
                msd = np.sum(displacement**2)
                msd_atom.append(msd)
            
            msd_values.append(msd_atom)
        
        # Average MSD over all atoms
        msd_avg = np.mean(msd_values, axis=0)
        time_steps = np.arange(len(msd_avg))
        
        # Create MSD plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, msd_avg, 'b-', linewidth=2, label='Average MSD')
        plt.xlabel('Time Step')
        plt.ylabel('Mean Squared Displacement (Å²)')
        plt.title('MSD Analysis for Si')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_file = os.path.join(sim_dir, "msd_analysis_test.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate diffusion coefficient
        if len(msd_avg) > 10:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_steps[10:], msd_avg[10:])
            diffusion_coeff = slope / 6.0  # D = slope/6 for 3D
        else:
            diffusion_coeff = 0.0
        
        return f"✅ MSD Analysis successful!\n- Timesteps: {n_timesteps}\n- Atoms: {n_atoms}\n- Diffusion coefficient: {diffusion_coeff:.2e} Å²/timestep\n- Plot saved: {plot_file}"
        
    except Exception as e:
        return f"❌ Error performing MSD analysis: {str(e)}"

def test_rdf_analysis():
    """Test RDF analysis on simulation results."""
    try:
        # Find simulation directory
        sim_dirs = glob.glob("simulations/Si_*K_*steps")
        if not sim_dirs:
            return "❌ No simulation directories found"
        
        sim_dir = sim_dirs[0]
        
        # Look for trajectory file first, then structure file
        trajectory_file = os.path.join(sim_dir, "trajectory.xyz")
        structure_file = os.path.join(sim_dir, "structure.xyz")
        
        if os.path.exists(trajectory_file):
            data_file = trajectory_file
        elif os.path.exists(structure_file):
            data_file = structure_file
        else:
            return "❌ Neither trajectory.xyz nor structure.xyz found. Cannot compute RDF."
        
        print(f"📄 Reading data from: {data_file}")
        
        # Read atomic positions
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Get number of atoms
        n_atoms = int(lines[0].strip())
        
        # Read atomic positions
        positions = []
        for i in range(2, 2 + n_atoms):  # Skip first two lines (atom count and comment)
            coords = lines[i].strip().split()
            if len(coords) >= 4:  # For custom format: element xu yu zu fx fy fz
                # We want positions (xu, yu, zu) which are indices 1, 2, 3
                positions.append([float(coords[1]), float(coords[2]), float(coords[3])])
        
        positions = np.array(positions)
        
        # Calculate RDF
        r_max = 10.0
        dr = 0.1
        r_bins = np.arange(0, r_max + dr, dr)
        rdf = np.zeros(len(r_bins) - 1)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < len(rdf):
                        rdf[bin_idx] += 2  # Count both i-j and j-i
        
        # Normalize RDF
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        volume = 4 * np.pi * r_centers**2 * dr
        rdf = rdf / (len(positions) * volume)
        
        # Create RDF plot
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, rdf, 'b-', linewidth=2)
        plt.xlabel('Distance (Å)')
        plt.ylabel('Radial Distribution Function')
        plt.title('RDF Analysis for Si')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join(sim_dir, "rdf_analysis_test.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"✅ RDF Analysis successful!\n- Atoms: {len(positions)}\n- RDF range: 0 to {r_max} Å\n- Plot saved: {plot_file}"
        
    except Exception as e:
        return f"❌ Error performing RDF analysis: {str(e)}"

def test_thermodynamic_analysis():
    """Test thermodynamic analysis on simulation results."""
    try:
        # Find simulation directory
        sim_dirs = glob.glob("simulations/Si_*K_*steps")
        if not sim_dirs:
            return "❌ No simulation directories found"
        
        sim_dir = sim_dirs[0]
        
        # Read the log file
        log_file = os.path.join(sim_dir, "output.log")
        if not os.path.exists(log_file):
            return "❌ Log file not found. Cannot compute thermodynamic analysis."
        
        print(f"📄 Reading log file: {log_file}")
        
        # Read LAMMPS log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find the data section
        data_start = None
        for i, line in enumerate(lines):
            if "Step Temp PotEng KinEng TotEng Press Volume" in line:
                data_start = i + 1
                break
        
        if data_start is None:
            return "❌ Could not find thermodynamic data in log file."
        
        # Parse data
        data = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 7:  # Step Temp PotEng KinEng TotEng Press Volume
                    try:
                        step = int(parts[0])
                        temp = float(parts[1])
                        poteng = float(parts[2])
                        kineng = float(parts[3])
                        toteng = float(parts[4])
                        press = float(parts[5])
                        volume = float(parts[6])
                        data.append([step, temp, poteng, kineng, toteng, press, volume])
                    except (ValueError, IndexError):
                        continue
        
        if not data:
            return "❌ No valid thermodynamic data found in log file."
        
        # Convert to numpy array
        data = np.array(data)
        steps = data[:, 0]
        temps = data[:, 1]
        potengs = data[:, 2]
        kinengs = data[:, 3]
        totengs = data[:, 4]
        pressures = data[:, 5]
        volumes = data[:, 6]
        
        # Calculate statistics
        avg_temp = np.mean(temps)
        std_temp = np.std(temps)
        avg_press = np.mean(pressures)
        std_press = np.std(pressures)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature plot
        ax1.plot(steps, temps, 'b-', linewidth=2)
        ax1.axhline(y=avg_temp, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_temp:.1f}K')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Evolution - Si')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Energy plot
        ax2.plot(steps, potengs, 'r-', linewidth=2, label='Potential Energy')
        ax2.plot(steps, kinengs, 'g-', linewidth=2, label='Kinetic Energy')
        ax2.plot(steps, totengs, 'b-', linewidth=2, label='Total Energy')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title('Energy Evolution - Si')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Pressure plot
        ax3.plot(steps, pressures, 'g-', linewidth=2)
        ax3.axhline(y=avg_press, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_press:.1f} bar')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Pressure (bar)')
        ax3.set_title('Pressure Evolution - Si')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Volume plot
        ax4.plot(steps, volumes, 'm-', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Volume (Å³)')
        ax4.set_title('Volume Evolution - Si')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(sim_dir, 'thermodynamic_analysis_test.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"✅ Thermodynamic Analysis successful!\n- Data points: {len(data)}\n- Average temperature: {avg_temp:.1f} ± {std_temp:.1f} K\n- Average pressure: {avg_press:.1f} ± {std_press:.1f} bar\n- Plot saved: {plot_file}"
        
    except Exception as e:
        return f"❌ Error performing thermodynamic analysis: {str(e)}"

def main():
    """Run all analysis tests."""
    print("🧪 Testing Analysis Functions (Standalone)")
    print("=" * 60)
    
    # Test 1: MSD Analysis
    print("\n1. Testing MSD Analysis...")
    msd_result = test_msd_analysis()
    print(msd_result)
    
    # Test 2: RDF Analysis
    print("\n2. Testing RDF Analysis...")
    rdf_result = test_rdf_analysis()
    print(rdf_result)
    
    # Test 3: Thermodynamic Analysis
    print("\n3. Testing Thermodynamic Analysis...")
    thermo_result = test_thermodynamic_analysis()
    print(thermo_result)
    
    print("\n" + "=" * 60)
    print("🎯 Analysis Testing Complete!")

if __name__ == "__main__":
    main()
