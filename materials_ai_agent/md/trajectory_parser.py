"""Trajectory parser for molecular dynamics output files."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


class TrajectoryParser:
    """Parser for LAMMPS trajectory and log files."""
    
    def __init__(self):
        """Initialize trajectory parser."""
        pass
    
    def parse_trajectory(self, dump_file: Path) -> Dict[str, Any]:
        """Parse LAMMPS trajectory dump file.
        
        Args:
            dump_file: Path to LAMMPS dump file
            
        Returns:
            Dictionary containing trajectory data
        """
        try:
            with open(dump_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header information
            n_atoms = int(lines[3])
            
            # Find timestep and box information
            timestep = None
            box_bounds = None
            
            i = 0
            while i < len(lines):
                if "ITEM: TIMESTEP" in lines[i]:
                    timestep = int(lines[i+1])
                elif "ITEM: BOX BOUNDS" in lines[i]:
                    box_bounds = []
                    for j in range(3):
                        bounds = [float(x) for x in lines[i+1+j].split()]
                        box_bounds.append(bounds)
                elif "ITEM: ATOMS" in lines[i]:
                    # Parse atomic data
                    columns = lines[i].split()[2:]  # Skip "ITEM: ATOMS"
                    data = []
                    
                    for j in range(i+1, i+1+n_atoms):
                        if j < len(lines):
                            row = [float(x) for x in lines[j].split()]
                            data.append(row)
                    
                    break
                i += 1
            
            # Convert to numpy arrays
            if data:
                data = np.array(data)
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=columns)
                
                return {
                    "timestep": timestep,
                    "n_atoms": n_atoms,
                    "box_bounds": box_bounds,
                    "data": df,
                    "columns": columns
                }
            else:
                return {"error": "No atomic data found"}
                
        except Exception as e:
            return {"error": f"Failed to parse trajectory: {str(e)}"}
    
    def parse_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Parse LAMMPS log file for thermodynamic data.
        
        Args:
            log_file: Path to LAMMPS log file
            
        Returns:
            Dictionary containing thermodynamic data
        """
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find thermo data section
            thermo_data = []
            in_thermo = False
            columns = None
            
            for line in lines:
                if "Step" in line and "Temp" in line and "Press" in line:
                    # Found thermo header
                    columns = line.split()
                    in_thermo = True
                    continue
                
                if in_thermo and line.strip() and not line.startswith("Loop"):
                    try:
                        data = [float(x) for x in line.split()]
                        if len(data) == len(columns):
                            thermo_data.append(data)
                    except ValueError:
                        # End of thermo data
                        break
            
            if thermo_data and columns:
                df = pd.DataFrame(thermo_data, columns=columns)
                
                return {
                    "thermo_data": df,
                    "columns": columns,
                    "n_steps": len(thermo_data)
                }
            else:
                return {"error": "No thermodynamic data found"}
                
        except Exception as e:
            return {"error": f"Failed to parse log file: {str(e)}"}
    
    def extract_properties(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic properties from trajectory data.
        
        Args:
            trajectory_data: Parsed trajectory data
            
        Returns:
            Dictionary containing extracted properties
        """
        if "error" in trajectory_data:
            return trajectory_data
        
        try:
            df = trajectory_data["data"]
            
            # Calculate basic properties
            properties = {}
            
            if "x" in df.columns and "y" in df.columns and "z" in df.columns:
                # Calculate center of mass
                positions = df[["x", "y", "z"]].values
                masses = df.get("mass", np.ones(len(df)))
                
                com = np.average(positions, axis=0, weights=masses)
                properties["center_of_mass"] = com.tolist()
                
                # Calculate box dimensions
                if trajectory_data["box_bounds"]:
                    box = np.array(trajectory_data["box_bounds"])
                    box_size = box[:, 1] - box[:, 0]
                    properties["box_dimensions"] = box_size.tolist()
                    properties["volume"] = np.prod(box_size)
            
            if "vx" in df.columns and "vy" in df.columns and "vz" in df.columns:
                # Calculate kinetic energy and temperature
                velocities = df[["vx", "vy", "vz"]].values
                v_squared = np.sum(velocities**2, axis=1)
                kinetic_energy = 0.5 * masses * v_squared
                properties["kinetic_energy"] = np.sum(kinetic_energy)
                
                # Temperature from kinetic energy (simplified)
                n_atoms = len(df)
                temperature = (2.0 / 3.0) * properties["kinetic_energy"] / n_atoms
                properties["temperature"] = temperature
            
            return properties
            
        except Exception as e:
            return {"error": f"Failed to extract properties: {str(e)}"}
    
    def get_trajectory_summary(self, dump_file: Path) -> Dict[str, Any]:
        """Get summary of trajectory file.
        
        Args:
            dump_file: Path to LAMMPS dump file
            
        Returns:
            Dictionary containing trajectory summary
        """
        try:
            with open(dump_file, 'r') as f:
                lines = f.readlines()
            
            # Count timesteps
            timestep_count = 0
            n_atoms = 0
            
            for line in lines:
                if "ITEM: TIMESTEP" in line:
                    timestep_count += 1
                elif "ITEM: NUMBER OF ATOMS" in line:
                    n_atoms = int(line.split()[-1])
            
            file_size = dump_file.stat().st_size
            
            return {
                "file_size_mb": file_size / (1024 * 1024),
                "n_timesteps": timestep_count,
                "n_atoms": n_atoms,
                "file_path": str(dump_file)
            }
            
        except Exception as e:
            return {"error": f"Failed to get trajectory summary: {str(e)}"}
