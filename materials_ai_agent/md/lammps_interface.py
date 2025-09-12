"""LAMMPS interface for molecular dynamics simulations."""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from ase import Atoms

from ..core.config import Config
from ..core.exceptions import SimulationError
from .trajectory_parser import TrajectoryParser


class LAMMPSInterface:
    """Interface for LAMMPS molecular dynamics simulations."""
    
    def __init__(self, config: Config):
        """Initialize LAMMPS interface.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.lammps_executable = config.lammps_executable
        self.trajectory_parser = TrajectoryParser()
    
    def generate_input_file(
        self,
        structure: Atoms,
        params: Any,
        output_dir: Path
    ) -> Path:
        """Generate LAMMPS input file.
        
        Args:
            structure: ASE Atoms object
            params: Simulation parameters
            output_dir: Output directory
            
        Returns:
            Path to generated input file
        """
        input_file = output_dir / "in.lammps"
        
        # Write structure file
        structure_file = output_dir / "structure.data"
        self._write_structure_file(structure, structure_file)
        
        # Generate input script
        input_script = self._generate_input_script(structure, params, structure_file)
        
        with open(input_file, 'w') as f:
            f.write(input_script)
        
        return input_file
    
    def run_simulation(self, input_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Run LAMMPS simulation.
        
        Args:
            input_file: Path to LAMMPS input file
            output_dir: Output directory
            
        Returns:
            Dictionary containing simulation results
        """
        start_time = time.time()
        
        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(output_dir)
        
        try:
            # Run LAMMPS
            cmd = [self.lammps_executable, "-in", input_file.name]
            
            with open("lammps.log", "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=output_dir
                )
            
            if result.returncode != 0:
                raise SimulationError(f"LAMMPS simulation failed: {result.stderr}")
            
            # List output files
            output_files = list(output_dir.glob("*.dump")) + list(output_dir.glob("*.log"))
            
            simulation_time = time.time() - start_time
            
            return {
                "output_files": [str(f) for f in output_files],
                "simulation_time": simulation_time,
                "return_code": result.returncode
            }
            
        finally:
            os.chdir(original_dir)
    
    def check_simulation_status(self, output_dir: Path) -> Dict[str, Any]:
        """Check if simulation is running.
        
        Args:
            output_dir: Simulation output directory
            
        Returns:
            Dictionary containing simulation status
        """
        # Check for running processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", "lmp"],
                capture_output=True,
                text=True
            )
            
            running = result.returncode == 0
            
            return {
                "running": running,
                "status": "running" if running else "completed"
            }
            
        except Exception:
            return {
                "running": False,
                "status": "unknown"
            }
    
    def get_simulation_progress(self, output_dir: Path) -> Dict[str, Any]:
        """Get simulation progress.
        
        Args:
            output_dir: Simulation output directory
            
        Returns:
            Dictionary containing progress information
        """
        # Look for log files to extract progress
        log_files = list(output_dir.glob("*.log"))
        
        if not log_files:
            return {"progress": 0, "current_step": 0, "total_steps": 0}
        
        # Parse the most recent log file
        latest_log = max(log_files, key=os.path.getctime)
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Look for step information
            current_step = 0
            total_steps = 0
            
            for line in lines:
                if "Step" in line and "CPU" in line:
                    try:
                        current_step = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif "Total wall time" in line:
                    # Simulation completed
                    break
            
            # Try to get total steps from input file
            input_file = output_dir / "in.lammps"
            if input_file.exists():
                with open(input_file, 'r') as f:
                    content = f.read()
                    if "run" in content:
                        try:
                            total_steps = int(content.split("run")[1].split()[0])
                        except (IndexError, ValueError):
                            pass
            
            progress = (current_step / total_steps * 100) if total_steps > 0 else 0
            
            return {
                "progress": progress,
                "current_step": current_step,
                "total_steps": total_steps
            }
            
        except Exception:
            return {"progress": 0, "current_step": 0, "total_steps": 0}
    
    def _write_structure_file(self, structure: Atoms, output_file: Path) -> None:
        """Write structure file in LAMMPS format.
        
        Args:
            structure: ASE Atoms object
            output_file: Output file path
        """
        # Get unique elements
        elements = list(set(structure.get_chemical_symbols()))
        element_map = {elem: i+1 for i, elem in enumerate(elements)}
        
        with open(output_file, 'w') as f:
            f.write("LAMMPS data file generated by Materials AI Agent\n\n")
            f.write(f"{len(structure)} atoms\n")
            f.write(f"{len(elements)} atom types\n\n")
            
            # Box bounds
            cell = structure.get_cell()
            f.write("0.0 {:.6f} xlo xhi\n".format(cell[0, 0]))
            f.write("0.0 {:.6f} ylo yhi\n".format(cell[1, 1]))
            f.write("0.0 {:.6f} zlo zhi\n".format(cell[2, 2]))
            f.write("\n")
            
            # Masses
            f.write("Masses\n\n")
            for elem in elements:
                # Get atomic mass (simplified)
                mass = self._get_atomic_mass(elem)
                f.write(f"{element_map[elem]} {mass:.6f}\n")
            f.write("\n")
            
            # Atoms
            f.write("Atoms\n\n")
            positions = structure.get_positions()
            symbols = structure.get_chemical_symbols()
            
            for i, (pos, symbol) in enumerate(zip(positions, symbols)):
                f.write(f"{i+1} {element_map[symbol]} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    def _generate_input_script(
        self,
        structure: Atoms,
        params: Any,
        structure_file: Path
    ) -> str:
        """Generate LAMMPS input script.
        
        Args:
            structure: ASE Atoms object
            params: Simulation parameters
            structure_file: Path to structure file
            
        Returns:
            LAMMPS input script as string
        """
        script = f"""# LAMMPS input file generated by Materials AI Agent
# Material: {params.material}
# Temperature: {params.temperature} K
# Pressure: {params.pressure} atm
# Ensemble: {params.ensemble}

# Initialize
units metal
dimension 3
boundary p p p
atom_style atomic

# Read structure
read_data {structure_file.name}

# Force field setup
"""
        
        # Add force field specific commands
        if params.force_field.lower() == "tersoff":
            script += """# Tersoff potential for covalent materials
pair_style tersoff
pair_coeff * * Si.tersoff Si
"""
        elif params.force_field.lower() == "lj":
            script += """# Lennard-Jones potential
pair_style lj/cut 10.0
pair_coeff * * 1.0 1.0
"""
        elif params.force_field.lower() == "eam":
            script += """# EAM potential for metals
pair_style eam/alloy
pair_coeff * * Al99.eam.alloy Al
"""
        else:
            script += f"""# Using {params.force_field} potential
# Note: You may need to provide appropriate potential files
pair_style lj/cut 10.0
pair_coeff * * 1.0 1.0
"""
        
        script += f"""
# Neighbor settings
neighbor 2.0 bin
neigh_modify delay 0 every 20 check no

# Output settings
thermo_style custom step temp press pe ke etotal vol
thermo {params.output_frequency}
dump 1 all atom {params.output_frequency} trajectory.dump
dump_modify 1 sort id

# Initial velocities
velocity all create {params.temperature} 12345

# Minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Equilibration
"""
        
        # Add ensemble-specific commands
        if params.ensemble.upper() == "NVT":
            script += f"""fix 1 all nvt temp {params.temperature} {params.temperature} 0.1
"""
        elif params.ensemble.upper() == "NPT":
            script += f"""fix 1 all npt temp {params.temperature} {params.temperature} 0.1 iso {params.pressure} {params.pressure} 1.0
"""
        elif params.ensemble.upper() == "NVE":
            script += """fix 1 all nve
"""
        
        script += f"""
# Production run
timestep {params.timestep}
run {params.n_steps}

# Final output
write_data final_structure.data
"""
        
        return script
    
    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass for element.
        
        Args:
            element: Element symbol
            
        Returns:
            Atomic mass in amu
        """
        masses = {
            "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012,
            "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
            "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
            "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.065,
            "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
            "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Ag": 107.868,
            "Au": 196.967
        }
        return masses.get(element, 1.0)
