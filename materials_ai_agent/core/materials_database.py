"""Material database with configurable lattice parameters and properties."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MaterialProperties:
    """Material properties and lattice parameters."""
    formula: str
    lattice_type: str
    lattice_parameter: float
    unit_cell_atoms: int
    density: float
    melting_point: float
    recommended_timestep: float
    recommended_force_field: str
    description: str


class MaterialsDatabase:
    """Configurable database of materials with their properties."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize materials database.
        
        Args:
            config_file: Path to custom materials configuration file
        """
        self.config_file = config_file or Path(__file__).parent / "materials_config.json"
        self.materials = self._load_materials()
    
    def _load_materials(self) -> Dict[str, MaterialProperties]:
        """Load materials from configuration file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return {
                    formula: MaterialProperties(**props) 
                    for formula, props in data.items()
                }
        else:
            # Return default materials if no config file exists
            return self._get_default_materials()
    
    def _get_default_materials(self) -> Dict[str, MaterialProperties]:
        """Get default materials database."""
        return {
            "Si": MaterialProperties(
                formula="Si",
                lattice_type="diamond",
                lattice_parameter=5.43,
                unit_cell_atoms=8,
                density=2.33,
                melting_point=1687.0,
                recommended_timestep=0.001,
                recommended_force_field="tersoff",
                description="Silicon - semiconductor material"
            ),
            "Al": MaterialProperties(
                formula="Al",
                lattice_type="fcc",
                lattice_parameter=4.05,
                unit_cell_atoms=4,
                density=2.70,
                melting_point=933.5,
                recommended_timestep=0.001,
                recommended_force_field="eam",
                description="Aluminum - metallic material"
            ),
            "Cu": MaterialProperties(
                formula="Cu",
                lattice_type="fcc",
                lattice_parameter=3.61,
                unit_cell_atoms=4,
                density=8.96,
                melting_point=1357.8,
                recommended_timestep=0.001,
                recommended_force_field="eam",
                description="Copper - metallic material"
            ),
            "Fe": MaterialProperties(
                formula="Fe",
                lattice_type="bcc",
                lattice_parameter=2.87,
                unit_cell_atoms=2,
                density=7.87,
                melting_point=1811.0,
                recommended_timestep=0.001,
                recommended_force_field="eam",
                description="Iron - metallic material"
            ),
            "H2O": MaterialProperties(
                formula="H2O",
                lattice_type="molecular",
                lattice_parameter=0.0,  # Not applicable for molecular
                unit_cell_atoms=3,
                density=1.0,
                melting_point=273.15,
                recommended_timestep=0.0005,
                recommended_force_field="lj",
                description="Water - molecular material"
            ),
            "C": MaterialProperties(
                formula="C",
                lattice_type="diamond",
                lattice_parameter=3.57,
                unit_cell_atoms=8,
                density=3.52,
                melting_point=3823.0,
                recommended_timestep=0.0005,
                recommended_force_field="tersoff",
                description="Carbon - covalent material"
            ),
            "Ge": MaterialProperties(
                formula="Ge",
                lattice_type="diamond",
                lattice_parameter=5.66,
                unit_cell_atoms=8,
                density=5.32,
                melting_point=1211.4,
                recommended_timestep=0.001,
                recommended_force_field="tersoff",
                description="Germanium - semiconductor material"
            ),
            "GaAs": MaterialProperties(
                formula="GaAs",
                lattice_type="zincblende",
                lattice_parameter=5.65,
                unit_cell_atoms=8,
                density=5.32,
                melting_point=1511.0,
                recommended_timestep=0.001,
                recommended_force_field="tersoff",
                description="Gallium Arsenide - compound semiconductor"
            )
        }
    
    def get_material(self, formula: str) -> Optional[MaterialProperties]:
        """Get material properties by formula.
        
        Args:
            formula: Material formula (e.g., 'Si', 'Al', 'H2O')
            
        Returns:
            MaterialProperties object or None if not found
        """
        return self.materials.get(formula.upper())
    
    def get_all_materials(self) -> Dict[str, MaterialProperties]:
        """Get all available materials.
        
        Returns:
            Dictionary of all materials
        """
        return self.materials
    
    def add_material(self, material: MaterialProperties) -> None:
        """Add a new material to the database.
        
        Args:
            material: MaterialProperties object to add
        """
        self.materials[material.formula.upper()] = material
        self._save_materials()
    
    def remove_material(self, formula: str) -> bool:
        """Remove a material from the database.
        
        Args:
            formula: Material formula to remove
            
        Returns:
            True if removed, False if not found
        """
        if formula.upper() in self.materials:
            del self.materials[formula.upper()]
            self._save_materials()
            return True
        return False
    
    def _save_materials(self) -> None:
        """Save materials to configuration file."""
        data = {
            formula: {
                "formula": props.formula,
                "lattice_type": props.lattice_type,
                "lattice_parameter": props.lattice_parameter,
                "unit_cell_atoms": props.unit_cell_atoms,
                "density": props.density,
                "melting_point": props.melting_point,
                "recommended_timestep": props.recommended_timestep,
                "recommended_force_field": props.recommended_force_field,
                "description": props.description
            }
            for formula, props in self.materials.items()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_recommended_parameters(self, formula: str) -> Dict[str, Any]:
        """Get recommended simulation parameters for a material.
        
        Args:
            formula: Material formula
            
        Returns:
            Dictionary with recommended parameters
        """
        material = self.get_material(formula)
        if not material:
            return {}
        
        return {
            "timestep": material.recommended_timestep,
            "force_field": material.recommended_force_field,
            "lattice_parameter": material.lattice_parameter,
            "lattice_type": material.lattice_type,
            "description": material.description
        }
    
    def search_materials(self, query: str) -> Dict[str, MaterialProperties]:
        """Search materials by formula or description.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of matching materials
        """
        query_lower = query.lower()
        results = {}
        
        for formula, props in self.materials.items():
            if (query_lower in formula.lower() or 
                query_lower in props.description.lower() or
                query_lower in props.lattice_type.lower()):
                results[formula] = props
        
        return results
