"""Simulation tools for molecular dynamics simulations."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from langchain.tools import tool
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write
import numpy as np

from .base import BaseMaterialsTool
from ..core.exceptions import SimulationError
from ..md.lammps_interface import LAMMPSInterface


class SimulationParameters(BaseModel):
    """Parameters for MD simulation."""
    material: str = Field(..., description="Material formula or structure")
    temperature: float = Field(300.0, description="Temperature in K")
    pressure: float = Field(1.0, description="Pressure in atm")
    timestep: float = Field(0.001, description="Timestep in ps")
    n_steps: int = Field(10000, description="Number of simulation steps")
    ensemble: str = Field("NVT", description="Thermodynamic ensemble")
    force_field: str = Field("tersoff", description="Force field to use")
    output_frequency: int = Field(100, description="Output frequency")


class SimulationTool(BaseMaterialsTool):
    """Tool for setting up and running molecular dynamics simulations."""
    
    name: str = "simulation"
    description: str = "Set up and run molecular dynamics simulations using LAMMPS"
    
    def __init__(self, config):
        super().__init__(config)
        self.lammps_interface = LAMMPSInterface(config)
    
    def setup_simulation(
        self,
        material: str,
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        n_steps: int = 10000,
        ensemble: str = "NVT",
        force_field: str = "tersoff",
        output_frequency: int = 100
    ) -> Dict[str, Any]:
        """Set up a molecular dynamics simulation.
        
        Args:
            material: Material formula (e.g., 'Si', 'Al2O3', 'H2O')
            temperature: Temperature in K
            pressure: Pressure in atm
            timestep: Timestep in ps
            n_steps: Number of simulation steps
            ensemble: Thermodynamic ensemble (NVT, NPT, NVE)
            force_field: Force field to use (tersoff, lj, eam, etc.)
            output_frequency: How often to output data
            
        Returns:
            Dictionary containing simulation setup information
        """
        try:
            self._validate_input(locals())
            
            # Create simulation parameters
            params = SimulationParameters(
                material=material,
                temperature=temperature,
                pressure=pressure,
                timestep=timestep,
                n_steps=n_steps,
                ensemble=ensemble,
                force_field=force_field,
                output_frequency=output_frequency
            )
            
            # Generate structure
            structure = self._generate_structure(material)
            
            # Create simulation directory
            sim_dir = self.config.simulation_output_dir / f"{material}_{ensemble}_{temperature}K"
            sim_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate LAMMPS input files
            input_file = self.lammps_interface.generate_input_file(
                structure, params, sim_dir
            )
            
            return {
                "success": True,
                "simulation_directory": str(sim_dir),
                "input_file": str(input_file),
                "parameters": params.dict(),
                "structure_info": {
                    "formula": structure.get_chemical_formula(),
                    "n_atoms": len(structure),
                    "cell_volume": structure.get_volume(),
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "setup_simulation")
            }
    
    def run_simulation(
        self,
        simulation_directory: str,
        input_file: str = "in.lammps"
    ) -> Dict[str, Any]:
        """Run a molecular dynamics simulation.
        
        Args:
            simulation_directory: Path to simulation directory
            input_file: Name of LAMMPS input file
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            sim_dir = Path(simulation_directory)
            input_path = sim_dir / input_file
            
            if not input_path.exists():
                raise SimulationError(f"Input file not found: {input_path}")
            
            # Run LAMMPS simulation
            result = self.lammps_interface.run_simulation(input_path, sim_dir)
            
            return {
                "success": True,
                "simulation_directory": str(sim_dir),
                "output_files": result["output_files"],
                "simulation_time": result["simulation_time"],
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "run_simulation")
            }
    
    def monitor_simulation(
        self,
        simulation_directory: str
    ) -> Dict[str, Any]:
        """Monitor a running simulation.
        
        Args:
            simulation_directory: Path to simulation directory
            
        Returns:
            Dictionary containing simulation status and progress
        """
        try:
            sim_dir = Path(simulation_directory)
            
            # Check if simulation is running
            status = self.lammps_interface.check_simulation_status(sim_dir)
            
            if status["running"]:
                # Get current progress
                progress = self.lammps_interface.get_simulation_progress(sim_dir)
                
                return {
                    "success": True,
                    "running": True,
                    "progress": progress,
                    "simulation_directory": str(sim_dir)
                }
            else:
                return {
                    "success": True,
                    "running": False,
                    "status": status["status"],
                    "simulation_directory": str(sim_dir)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "monitor_simulation")
            }
    
    def _generate_structure(self, material: str) -> Atoms:
        """Generate atomic structure for the given material.
        
        Args:
            material: Material formula
            
        Returns:
            ASE Atoms object
        """
        # Simple structure generation - can be extended
        if material.upper() == "SI":
            return bulk("Si", "diamond", a=5.43)
        elif material.upper() == "AL2O3":
            # Alpha-Al2O3 structure
            return bulk("Al2O3", "corundum", a=4.76, c=12.99)
        elif material.upper() == "H2O":
            # Water molecule
            return molecule("H2O")
        elif material.upper() == "FE":
            return bulk("Fe", "bcc", a=2.87)
        elif material.upper() == "CU":
            return bulk("Cu", "fcc", a=3.61)
        else:
            # Default: try to create a simple cubic structure
            # This is a simplified approach - in practice, you'd want more sophisticated structure generation
            raise SimulationError(f"Unknown material: {material}. Please provide a known material or structure file.")
    
    def list_available_materials(self) -> Dict[str, Any]:
        """List available materials for simulation.
        
        Returns:
            Dictionary containing available materials
        """
        materials = {
            "elements": ["Si", "Al", "Fe", "Cu", "C", "O", "H", "N"],
            "compounds": ["Al2O3", "SiO2", "H2O", "NH3", "CH4"],
            "crystal_structures": {
                "Si": "diamond",
                "Al": "fcc", 
                "Fe": "bcc",
                "Cu": "fcc",
                "Al2O3": "corundum"
            }
        }
        
        return {
            "success": True,
            "materials": materials
        }
    
    def get_force_fields(self) -> Dict[str, Any]:
        """Get available force fields.
        
        Returns:
            Dictionary containing available force fields
        """
        force_fields = {
            "tersoff": {
                "description": "Tersoff potential for covalent materials",
                "materials": ["Si", "C", "Ge", "SiC"],
                "properties": ["elastic", "thermal", "defects"]
            },
            "lj": {
                "description": "Lennard-Jones potential for noble gases and simple systems",
                "materials": ["Ar", "Ne", "Kr", "Xe"],
                "properties": ["liquid", "gas", "crystal"]
            },
            "eam": {
                "description": "Embedded Atom Method for metals",
                "materials": ["Al", "Cu", "Fe", "Ni", "Ag", "Au"],
                "properties": ["elastic", "thermal", "defects", "surfaces"]
            },
            "reaxff": {
                "description": "ReaxFF reactive force field",
                "materials": ["C", "H", "O", "N", "S", "P"],
                "properties": ["reactive", "combustion", "catalysis"]
            }
        }
        
        return {
            "success": True,
            "force_fields": force_fields
        }
