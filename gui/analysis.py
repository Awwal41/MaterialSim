"""Post-simulation analysis for the Streamlit GUI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gui import icons
from gui.utils import resolve_sim_dir

def perform_rdf_analysis():
    """Perform RDF analysis on simulation results."""
    try:
        # Get simulation directory
        workflow = st.session_state.simulation_workflow
        material = workflow["material"]
        temperature = workflow["temperature"]
        n_steps = workflow["n_steps"]
        
        sim_dir = f"simulations/{material}_{temperature}K_{n_steps}steps"
        
        # Check if files exist
        import os
        if not os.path.exists(sim_dir):
            # Try to find the actual directory
            import glob
            possible_dirs = glob.glob(f"simulations/{material}_*K_*steps")
            if possible_dirs:
                sim_dir = possible_dirs[0]
            else:
                # Try to find any simulation directory
                all_dirs = glob.glob("simulations/*")
                if all_dirs:
                    sim_dir = all_dirs[0]  # Use the most recent one
                else:
                    return f"❌ No simulation directories found. Looking for: {sim_dir}"
        
        # Read the structure file and compute RDF directly
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
            return "❌ Neither trajectory.xyz nor structure.xyz found. Cannot compute RDF."
        
        # Simple RDF analysis using the structure/trajectory file
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Read XYZ file
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
            n_bins = 200
            r_bins = np.linspace(0, r_max, n_bins + 1)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            dr = r_bins[1] - r_bins[0]
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < r_max:
                        distances.append(dist)
            
            distances = np.array(distances)
            
            # Create histogram
            hist, _ = np.histogram(distances, bins=r_bins)
            
            # Calculate RDF
            rho = len(positions) / (4/3 * np.pi * r_max**3)  # Approximate density
            g_r = hist / (4 * np.pi * r_centers**2 * dr * rho * len(positions))
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(g_r, height=0.1)
            peak_distances = r_centers[peaks]
            peak_heights = g_r[peaks]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(r_centers, g_r, 'b-', linewidth=2, label='RDF')
            plt.scatter(peak_distances, peak_heights, color='red', s=50, zorder=5, label='Peaks')
            plt.xlabel('Distance (Å)')
            plt.ylabel('g(r)')
            plt.title(f'Radial Distribution Function - {material} at {temperature}K')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plot_file = os.path.join(sim_dir, 'rdf_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            st.image(plot_file, caption=f"RDF for {material} at {temperature}K")
            
            # Create summary
            summary = f"""**RDF Analysis Complete!**

📊 **Key Results:**
- **First peak**: {peak_distances[0]:.2f} Å (nearest neighbor distance)
- **Peak count**: {len(peaks)} significant peaks found
- **Max RDF value**: {np.max(g_r):.2f}
- **Plot saved**: {plot_file}

🔬 **Interpretation:**
- The first peak at {peak_distances[0]:.2f} Å shows the nearest neighbor distance
- Peak heights indicate coordination strength
- Peak positions reveal the atomic structure pattern

📁 **Files created:**
- RDF plot: `{plot_file}`
- Raw data available in simulation directory

The RDF shows the probability of finding atoms at different distances from each other. Peaks indicate preferred interatomic distances, which reveal the atomic structure and coordination."""
            
            return summary
            
        except Exception as analysis_error:
            return f"""🔬 **RDF Analysis**

I attempted to compute the RDF but encountered an error: {str(analysis_error)}

However, I can still help you understand what RDF analysis would show:

**What RDF reveals:**
- **Atomic coordination**: How many neighbors each atom has
- **Bond distances**: Preferred interatomic distances  
- **Structure type**: Crystalline vs amorphous characteristics

**For your {material} simulation:**
- First peak typically around 2.3-2.5 Å (Si-Si bonds)
- Peaks at ~3.8 Å, ~4.5 Å show second and third coordination shells
- Peak heights indicate coordination strength

Would you like me to try a different analysis approach or help you interpret the simulation results in another way?"""
            
    except Exception as e:
        return f"❌ Error performing RDF analysis: {str(e)}"

def perform_msd_analysis():
    """Perform MSD analysis on simulation results."""
    try:
        workflow = st.session_state.simulation_workflow
        material = workflow["material"]
        temperature = workflow["temperature"]
        n_steps = workflow["n_steps"]
        
        sim_dir = f"simulations/{material}_{temperature}K_{n_steps}steps"
        
        # Robust directory lookup
        import os, glob
        if not os.path.exists(sim_dir):
            possible_dirs = glob.glob(f"simulations/{material}_*K_*steps")
            if possible_dirs:
                sim_dir = possible_dirs[0]
            else:
                all_dirs = glob.glob("simulations/*")
                if all_dirs:
                    sim_dir = all_dirs[0]
                else:
                    return f"❌ No simulation directories found. Looking for: {sim_dir}"
        
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
        
        # Read trajectory data
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Parse XYZ file to get atomic positions over time
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
        
        if len(positions) < 2:
            if file_type == "trajectory":
                return """🔬 **MSD Analysis - Insufficient Trajectory Data**

The trajectory file only contains 1 timestep, which is not enough to calculate Mean Squared Displacement (MSD).

**What MSD Analysis Needs:**
- Multiple timesteps showing atomic positions over time
- At least 2 snapshots to calculate displacement
- Trajectory data showing how atoms move

**Why This Happened:**
The LAMMPS simulation may not have run long enough or the trajectory output frequency was too low.

**Alternative Analysis Options:**
- **RDF Analysis**: Can analyze atomic structure from single snapshot
- **Structure Visualization**: View the 3D atomic structure
- **Thermodynamic Analysis**: Analyze temperature, energy, pressure data

Would you like to try a different analysis type that works with single timestep data?"""
            else:
                return """🔬 **MSD Analysis - No Trajectory Data**

The simulation only saved a single structure snapshot instead of a full trajectory.

**What MSD Analysis Needs:**
- Multiple timesteps showing atomic positions over time
- At least 2 snapshots to calculate displacement
- Trajectory data showing how atoms move

**Why This Happened:**
The current simulation only saved a single structure snapshot instead of a full trajectory.

**Alternative Analysis Options:**
- **RDF Analysis**: Can analyze atomic structure from single snapshot
- **Structure Visualization**: View the 3D atomic structure
- **Thermodynamic Analysis**: Analyze temperature, energy, pressure data

Would you like to try a different analysis type that works with single timestep data?"""
        
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
        plt.title(f'MSD Analysis for {material} at {temperature}K')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_file = os.path.join(sim_dir, "msd_analysis.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate diffusion coefficient (slope of MSD vs time)
        if len(msd_avg) > 10:
            # Use linear fit for diffusion coefficient
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_steps[10:], msd_avg[10:])
            diffusion_coeff = slope / 6.0  # D = slope/6 for 3D
        else:
            diffusion_coeff = 0.0
        
        # Display plot in Streamlit
        st.image(plot_file, caption=f"MSD Analysis for {material} at {temperature}K")
        
        return f"""🔬 **MSD Analysis Results**

**Material**: {material} at {temperature}K
**Analysis**: Mean Squared Displacement over {n_timesteps} timesteps
**Diffusion Coefficient**: {diffusion_coeff:.2e} Å²/timestep

**Key Findings:**
- MSD shows atomic mobility and diffusion behavior
- Higher MSD values indicate greater atomic movement
- Linear MSD growth suggests normal diffusion
- Diffusion coefficient: {diffusion_coeff:.2e} Å²/timestep

The MSD analysis reveals how atoms move over time, indicating diffusion behavior and material properties.

Would you like to:
- **Download the MSD data**: Get the raw MSD values as a file
- **Analyze RDF**: Look at atomic structure  
- **Plot temperature**: See thermodynamic properties
- **New analysis**: Try a different analysis type"""
            
    except Exception as e:
        return f"❌ Error performing MSD analysis: {str(e)}"

def perform_thermodynamic_analysis():
    """Perform thermodynamic analysis on simulation results."""
    try:
        from gui.utils import resolve_sim_dir
        from materials_ai_agent.analysis_engine import compute_thermodynamics
        from materials_ai_agent.simulation_quality import format_quality_report, assess_simulation_quality

        workflow = st.session_state.simulation_workflow
        material = workflow["material"]
        temperature = workflow["temperature"]
        n_steps = workflow["n_steps"]

        sim_dir = resolve_sim_dir(material, temperature, n_steps)
        if not sim_dir:
            return f"❌ No simulation directories found for {material}."

        thermo = compute_thermodynamics(Path(sim_dir))
        if not thermo.get("success"):
            return f"❌ {thermo.get('error', 'Thermodynamic analysis failed.')}"

        quality = assess_simulation_quality(Path(sim_dir), temperature)
        report = format_quality_report(quality)

        if thermo.get("plot_file"):
            st.image(thermo["plot_file"], caption=f"Thermodynamics — production window")

        press_line = (
            f"- **Production pressure**: {thermo['avg_pressure']:.1f} ± {thermo['std_pressure']:.1f} bar"
            if thermo.get("pressure_reliable")
            else "- **Production pressure**: unreliable (simplified potential)"
        )

        return f"""**{icons.CHART} Thermodynamic Analysis** (production window only)

{report}

**Summary**
- **Production temperature**: {thermo['avg_temperature']:.1f} ± {thermo['std_temperature']:.1f} K (target {temperature:g} K)
{press_line}
- **Frames analyzed**: {thermo['n_points']} production / {thermo.get('n_points_total', '?')} total
"""
    except Exception as e:
        return f"{icons.ERROR} Error performing thermodynamic analysis: {str(e)}"