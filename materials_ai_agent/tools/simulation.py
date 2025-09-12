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
    temperature: float = Field(None, description="Temperature in K (uses config default if None)")
    pressure: float = Field(None, description="Pressure in atm (uses config default if None)")
    timestep: float = Field(None, description="Timestep in ps (uses config default if None)")
    n_steps: int = Field(None, description="Number of simulation steps (uses config default if None)")
    ensemble: str = Field(None, description="Thermodynamic ensemble (uses config default if None)")
    force_field: str = Field(None, description="Force field to use (uses config default if None)")
    output_frequency: int = Field(None, description="Output frequency (uses config default if None)")


class SimulationTool(BaseMaterialsTool):
    """Tool for setting up and running molecular dynamics simulations."""
    
    name: str = Field(default="simulation", description="Tool name")
    description: str = Field(default="Set up and run molecular dynamics simulations using LAMMPS", description="Tool description")
    
    def __init__(self, config):
        super().__init__(config)
        self.lammps_interface = LAMMPSInterface(config)
    
    def setup_simulation(
        self,
        material: str,
        temperature: float = None,
        pressure: float = None,
        timestep: float = None,
        n_steps: int = None,
        ensemble: str = None,
        force_field: str = None,
        output_frequency: int = None
    ) -> Dict[str, Any]:
        """Set up a molecular dynamics simulation.
        
        Args:
            material: Material formula (e.g., 'Si', 'Al2O3', 'H2O')
            temperature: Temperature in K (uses config default if None)
            pressure: Pressure in atm (uses config default if None)
            timestep: Timestep in ps (uses config default if None)
            n_steps: Number of simulation steps (uses config default if None)
            ensemble: Thermodynamic ensemble (uses config default if None)
            force_field: Force field to use (uses config default if None)
            output_frequency: How often to output data (uses config default if None)
            
        Returns:
            Dictionary containing simulation setup information
        """
        try:
            # Use configuration defaults if not provided
            if temperature is None:
                temperature = self.config.default_temperature
            if pressure is None:
                pressure = self.config.default_pressure
            if timestep is None:
                timestep = self.config.default_timestep
            if n_steps is None:
                n_steps = self.config.default_n_steps
            if ensemble is None:
                ensemble = self.config.default_ensemble
            if force_field is None:
                force_field = self.config.default_force_field
            if output_frequency is None:
                output_frequency = 100  # Default output frequency
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
    
    def run_md_simulation(
        self,
        material: str,
        temperature: float = None,
        n_steps: int = None,
        force_field: str = None
    ) -> Dict[str, Any]:
        """Run a complete molecular dynamics simulation from start to finish.
        
        Args:
            material: Material formula (e.g., 'Si', 'Al2O3', 'H2O')
            temperature: Temperature in K (uses config default if None)
            n_steps: Number of simulation steps (uses config default if None)
            force_field: Force field to use (uses config default if None)
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            # Use configuration defaults if not provided
            if temperature is None:
                temperature = self.config.default_temperature
            if n_steps is None:
                n_steps = self.config.default_n_steps
            if force_field is None:
                force_field = self.config.default_force_field
            # First set up the simulation
            setup_result = self.setup_simulation(
                material=material,
                temperature=temperature,
                n_steps=n_steps,
                force_field=force_field
            )
            
            if not setup_result["success"]:
                return setup_result
            
            # Then run the simulation
            run_result = self.run_simulation(
                simulation_directory=setup_result["simulation_directory"]
            )
            
            if not run_result["success"]:
                return run_result
            
            return {
                "success": True,
                "material": material,
                "temperature": temperature,
                "n_steps": n_steps,
                "force_field": force_field,
                "simulation_directory": setup_result["simulation_directory"],
                "output_files": run_result["output_files"],
                "simulation_time": run_result["simulation_time"],
                "status": "completed",
                "message": f"Successfully completed {n_steps} step MD simulation of {material} at {temperature}K using {force_field} potential"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "run_md_simulation")
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


# Create tool instances for LangChain
def create_simulation_tools(config):
    """Create simulation tools for LangChain agent."""
    sim_tool = SimulationTool(config)
    
    @tool
    def run_md_simulation(
        material: str,
        temperature: float = None,
        n_steps: int = None,
        force_field: str = None
    ) -> str:
        """Run a complete molecular dynamics simulation from start to finish.
        
        Args:
            material: Material formula (e.g., 'Si', 'Al2O3', 'H2O')
            temperature: Temperature in K
            n_steps: Number of simulation steps
            force_field: Force field to use (tersoff, lj, eam, etc.)
            
        Returns:
            String description of simulation results
        """
        result = sim_tool.run_md_simulation(
            material=material,
            temperature=temperature,
            n_steps=n_steps,
            force_field=force_field
        )
        
        if result["success"]:
            return f"✅ {result['message']}\nSimulation directory: {result['simulation_directory']}\nOutput files: {result['output_files']}"
        else:
            return f"❌ Simulation failed: {result['error']}"
    
    @tool
    def setup_simulation(
        material: str,
        temperature: float = None,
        pressure: float = None,
        timestep: float = None,
        n_steps: int = None,
        ensemble: str = None,
        force_field: str = None,
        output_frequency: int = None
    ) -> str:
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
            String description of simulation setup
        """
        result = sim_tool.setup_simulation(
            material=material,
            temperature=temperature,
            pressure=pressure,
            timestep=timestep,
            n_steps=n_steps,
            ensemble=ensemble,
            force_field=force_field,
            output_frequency=output_frequency
        )
        
        if result["success"]:
            return f"✅ Simulation setup complete!\nMaterial: {result['material']}\nTemperature: {result['temperature']}K\nSteps: {result['n_steps']}\nDirectory: {result['simulation_directory']}"
        else:
            return f"❌ Setup failed: {result['error']}"
    
    return [run_md_simulation, setup_simulation]
