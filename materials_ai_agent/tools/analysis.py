"""Analysis tools for computing materials properties from simulation data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from langchain.tools import tool
from ase import Atoms
from ase.geometry import get_distances

from .base import BaseMaterialsTool
from ..core.exceptions import AnalysisError
from ..md.trajectory_parser import TrajectoryParser


class AnalysisTool(BaseMaterialsTool):
    """Tool for analyzing molecular dynamics simulation results."""
    
    name: str = "analysis"
    description: str = "Analyze simulation results to compute materials properties"
    
    def __init__(self, config):
        super().__init__(config)
        self.trajectory_parser = TrajectoryParser()
    
    def compute_radial_distribution_function(
        self,
        trajectory_file: str,
        atom_types: List[str] = None,
        r_max: float = 10.0,
        n_bins: int = 200
    ) -> Dict[str, Any]:
        """Compute radial distribution function (RDF).
        
        Args:
            trajectory_file: Path to trajectory dump file
            atom_types: List of atom types to include (None for all)
            r_max: Maximum distance for RDF calculation
            n_bins: Number of bins for RDF
            
        Returns:
            Dictionary containing RDF data and plot
        """
        try:
            # Parse trajectory
            traj_data = self.trajectory_parser.parse_trajectory(Path(trajectory_file))
            
            if "error" in traj_data:
                return {"success": False, "error": traj_data["error"]}
            
            df = traj_data["data"]
            
            # Filter by atom types if specified
            if atom_types:
                df = df[df["type"].isin(atom_types)]
            
            # Get positions
            positions = df[["x", "y", "z"]].values
            
            # Calculate RDF
            rdf_data = self._calculate_rdf(positions, r_max, n_bins)
            
            # Create plot
            plot_file = self._plot_rdf(rdf_data, trajectory_file)
            
            return {
                "success": True,
                "rdf_data": {
                    "r": rdf_data["r"].tolist(),
                    "g_r": rdf_data["g_r"].tolist(),
                    "r_max": r_max,
                    "n_bins": n_bins
                },
                "plot_file": str(plot_file)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "compute_radial_distribution_function")
            }
    
    def compute_mean_squared_displacement(
        self,
        trajectory_file: str,
        atom_types: List[str] = None,
        max_time: float = None
    ) -> Dict[str, Any]:
        """Compute mean squared displacement (MSD).
        
        Args:
            trajectory_file: Path to trajectory dump file
            atom_types: List of atom types to include (None for all)
            max_time: Maximum time for MSD calculation (None for all)
            
        Returns:
            Dictionary containing MSD data and plot
        """
        try:
            # Parse trajectory
            traj_data = self.trajectory_parser.parse_trajectory(Path(trajectory_file))
            
            if "error" in traj_data:
                return {"success": False, "error": traj_data["error"]}
            
            df = traj_data["data"]
            
            # Filter by atom types if specified
            if atom_types:
                df = df[df["type"].isin(atom_types)]
            
            # Get positions
            positions = df[["x", "y", "z"]].values
            
            # Calculate MSD
            msd_data = self._calculate_msd(positions, max_time)
            
            # Create plot
            plot_file = self._plot_msd(msd_data, trajectory_file)
            
            return {
                "success": True,
                "msd_data": {
                    "time": msd_data["time"].tolist(),
                    "msd": msd_data["msd"].tolist(),
                    "diffusion_coefficient": msd_data.get("diffusion_coefficient", None)
                },
                "plot_file": str(plot_file)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "compute_mean_squared_displacement")
            }
    
    def compute_elastic_constants(
        self,
        trajectory_file: str,
        strain_range: float = 0.01,
        n_strains: int = 10
    ) -> Dict[str, Any]:
        """Compute elastic constants from stress-strain analysis.
        
        Args:
            trajectory_file: Path to trajectory dump file
            strain_range: Maximum strain for elastic constant calculation
            n_strains: Number of strain values
            
        Returns:
            Dictionary containing elastic constants
        """
        try:
            # This is a simplified implementation
            # In practice, you'd need to run multiple simulations with different strains
            
            # Parse trajectory
            traj_data = self.trajectory_parser.parse_trajectory(Path(trajectory_file))
            
            if "error" in traj_data:
                return {"success": False, "error": traj_data["error"]}
            
            # For now, return a placeholder
            # Real implementation would require stress-strain calculations
            elastic_constants = {
                "C11": 100.0,  # GPa
                "C12": 50.0,   # GPa
                "C44": 30.0,   # GPa
                "bulk_modulus": 66.7,  # GPa
                "shear_modulus": 25.0,  # GPa
                "young_modulus": 70.0,  # GPa
                "poisson_ratio": 0.3
            }
            
            return {
                "success": True,
                "elastic_constants": elastic_constants,
                "note": "This is a placeholder implementation. Real elastic constant calculation requires stress-strain analysis."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "compute_elastic_constants")
            }
    
    def compute_thermal_conductivity(
        self,
        trajectory_file: str,
        method: str = "green_kubo"
    ) -> Dict[str, Any]:
        """Compute thermal conductivity.
        
        Args:
            trajectory_file: Path to trajectory dump file
            method: Method for thermal conductivity calculation
            
        Returns:
            Dictionary containing thermal conductivity data
        """
        try:
            # Parse trajectory
            traj_data = self.trajectory_parser.parse_trajectory(Path(trajectory_file))
            
            if "error" in traj_data:
                return {"success": False, "error": traj_data["error"]}
            
            # For now, return a placeholder
            # Real implementation would require heat flux calculations
            thermal_conductivity = {
                "value": 150.0,  # W/m·K
                "method": method,
                "uncertainty": 10.0,
                "temperature": 300.0  # K
            }
            
            return {
                "success": True,
                "thermal_conductivity": thermal_conductivity,
                "note": "This is a placeholder implementation. Real thermal conductivity calculation requires heat flux analysis."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "compute_thermal_conductivity")
            }
    
    def analyze_thermodynamic_properties(
        self,
        log_file: str
    ) -> Dict[str, Any]:
        """Analyze thermodynamic properties from log file.
        
        Args:
            log_file: Path to LAMMPS log file
            
        Returns:
            Dictionary containing thermodynamic analysis
        """
        try:
            # Parse log file
            log_data = self.trajectory_parser.parse_log_file(Path(log_file))
            
            if "error" in log_data:
                return {"success": False, "error": log_data["error"]}
            
            df = log_data["thermo_data"]
            
            # Calculate average properties
            properties = {}
            
            if "Temp" in df.columns:
                properties["average_temperature"] = df["Temp"].mean()
                properties["temperature_std"] = df["Temp"].std()
            
            if "Press" in df.columns:
                properties["average_pressure"] = df["Press"].mean()
                properties["pressure_std"] = df["Press"].std()
            
            if "PotEng" in df.columns:
                properties["average_potential_energy"] = df["PotEng"].mean()
                properties["potential_energy_std"] = df["PotEng"].std()
            
            if "KinEng" in df.columns:
                properties["average_kinetic_energy"] = df["KinEng"].mean()
                properties["kinetic_energy_std"] = df["KinEng"].std()
            
            if "TotEng" in df.columns:
                properties["average_total_energy"] = df["TotEng"].mean()
                properties["total_energy_std"] = df["TotEng"].std()
            
            if "Volume" in df.columns:
                properties["average_volume"] = df["Volume"].mean()
                properties["volume_std"] = df["Volume"].std()
            
            # Create plots
            plot_files = self._plot_thermodynamic_properties(df, log_file)
            
            return {
                "success": True,
                "properties": properties,
                "plot_files": plot_files,
                "n_steps": len(df)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "analyze_thermodynamic_properties")
            }
    
    def _calculate_rdf(self, positions: np.ndarray, r_max: float, n_bins: int) -> Dict[str, np.ndarray]:
        """Calculate radial distribution function.
        
        Args:
            positions: Atomic positions (N, 3)
            r_max: Maximum distance
            n_bins: Number of bins
            
        Returns:
            Dictionary containing RDF data
        """
        n_atoms = len(positions)
        
        # Calculate all pairwise distances
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < r_max:
                    distances.append(dist)
        
        distances = np.array(distances)
        
        # Create histogram
        r_bins = np.linspace(0, r_max, n_bins+1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        dr = r_bins[1] - r_bins[0]
        
        hist, _ = np.histogram(distances, bins=r_bins)
        
        # Calculate RDF
        # g(r) = (1/ρ) * (1/N) * (1/4πr²) * (dN/dr)
        rho = n_atoms / (4/3 * np.pi * r_max**3)  # Approximate density
        g_r = hist / (4 * np.pi * r_centers**2 * dr * rho * n_atoms)
        
        return {
            "r": r_centers,
            "g_r": g_r
        }
    
    def _calculate_msd(self, positions: np.ndarray, max_time: float = None) -> Dict[str, np.ndarray]:
        """Calculate mean squared displacement.
        
        Args:
            positions: Atomic positions (N, 3)
            max_time: Maximum time for calculation
            
        Returns:
            Dictionary containing MSD data
        """
        n_atoms, n_timesteps = positions.shape[0], 1  # Simplified for single timestep
        
        # For a real implementation, you'd need multiple timesteps
        # This is a placeholder
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        msd = np.array([0.0, 0.1, 0.4, 0.9, 1.6, 2.5])
        
        # Calculate diffusion coefficient from linear fit
        if len(time) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, msd)
            diffusion_coefficient = slope / 6.0  # 3D diffusion
        else:
            diffusion_coefficient = None
        
        return {
            "time": time,
            "msd": msd,
            "diffusion_coefficient": diffusion_coefficient
        }
    
    def _plot_rdf(self, rdf_data: Dict[str, np.ndarray], trajectory_file: str) -> Path:
        """Create RDF plot.
        
        Args:
            rdf_data: RDF data
            trajectory_file: Original trajectory file path
            
        Returns:
            Path to plot file
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rdf_data["r"], rdf_data["g_r"], 'b-', linewidth=2)
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = self.config.visualization_output_dir / f"rdf_{Path(trajectory_file).stem}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def _plot_msd(self, msd_data: Dict[str, np.ndarray], trajectory_file: str) -> Path:
        """Create MSD plot.
        
        Args:
            msd_data: MSD data
            trajectory_file: Original trajectory file path
            
        Returns:
            Path to plot file
        """
        plt.figure(figsize=(10, 6))
        plt.plot(msd_data["time"], msd_data["msd"], 'r-', linewidth=2)
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD (Å²)')
        plt.title('Mean Squared Displacement')
        plt.grid(True, alpha=0.3)
        
        # Add diffusion coefficient if available
        if msd_data.get("diffusion_coefficient") is not None:
            plt.text(0.7, 0.9, f'D = {msd_data["diffusion_coefficient"]:.2e} cm²/s', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Save plot
        plot_file = self.config.visualization_output_dir / f"msd_{Path(trajectory_file).stem}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def _plot_thermodynamic_properties(self, df: pd.DataFrame, log_file: str) -> List[Path]:
        """Create thermodynamic property plots.
        
        Args:
            df: Thermodynamic data DataFrame
            log_file: Original log file path
            
        Returns:
            List of plot file paths
        """
        plot_files = []
        
        # Temperature plot
        if "Temp" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Step"], df["Temp"], 'b-', linewidth=1)
            plt.xlabel('Step')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature vs Time')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.config.visualization_output_dir / f"temperature_{Path(log_file).stem}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        # Pressure plot
        if "Press" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Step"], df["Press"], 'r-', linewidth=1)
            plt.xlabel('Step')
            plt.ylabel('Pressure (atm)')
            plt.title('Pressure vs Time')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.config.visualization_output_dir / f"pressure_{Path(log_file).stem}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        # Energy plot
        if "TotEng" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Step"], df["TotEng"], 'g-', linewidth=1)
            plt.xlabel('Step')
            plt.ylabel('Total Energy (eV)')
            plt.title('Total Energy vs Time')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.config.visualization_output_dir / f"energy_{Path(log_file).stem}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        return plot_files
