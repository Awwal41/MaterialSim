"""Database tools for querying materials databases."""

import requests
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain.tools import tool
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from .base import BaseMaterialsTool
from ..core.exceptions import DatabaseError


class DatabaseTool(BaseMaterialsTool):
    """Tool for querying materials databases."""
    
    name: str = "database"
    description: str = "Query materials databases for properties and structures"
    
    def __init__(self, config):
        super().__init__(config)
        self.mp_api_key = config.mp_api_key
        self.nomad_api_key = config.nomad_api_key
        
        # Initialize Materials Project client
        if self.mp_api_key:
            self.mp_client = MPRester(self.mp_api_key)
        else:
            self.mp_client = None
            self.logger.warning("Materials Project API key not provided")
    
    def query_materials_project(
        self,
        formula: str,
        properties: List[str] = None
    ) -> Dict[str, Any]:
        """Query Materials Project database.
        
        Args:
            formula: Chemical formula (e.g., 'Si', 'Al2O3')
            properties: List of properties to retrieve
            
        Returns:
            Dictionary containing Materials Project data
        """
        try:
            if not self.mp_client:
                return {
                    "success": False,
                    "error": "Materials Project API key not configured"
                }
            
            # Default properties to retrieve
            if properties is None:
                properties = [
                    "material_id", "formula_pretty", "structure", 
                    "energy_per_atom", "formation_energy_per_atom",
                    "band_gap", "density", "volume", "nsites"
                ]
            
            # Query Materials Project
            docs = self.mp_client.materials.summary.search(
                formula=formula,
                fields=properties
            )
            
            if not docs:
                return {
                    "success": False,
                    "error": f"No materials found for formula: {formula}"
                }
            
            # Convert to serializable format
            results = []
            for doc in docs:
                result = {}
                for prop in properties:
                    if hasattr(doc, prop):
                        value = getattr(doc, prop)
                        # Convert Structure objects to dict
                        if isinstance(value, Structure):
                            result[prop] = value.as_dict()
                        else:
                            result[prop] = value
                results.append(result)
            
            return {
                "success": True,
                "formula": formula,
                "n_materials": len(results),
                "materials": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "query_materials_project")
            }
    
    def get_elastic_properties(
        self,
        material_id: str
    ) -> Dict[str, Any]:
        """Get elastic properties from Materials Project.
        
        Args:
            material_id: Materials Project material ID
            
        Returns:
            Dictionary containing elastic properties
        """
        try:
            if not self.mp_client:
                return {
                    "success": False,
                    "error": "Materials Project API key not configured"
                }
            
            # Query elastic properties
            elastic_doc = self.mp_client.materials.elasticity.get_data_by_id(material_id)
            
            if not elastic_doc:
                return {
                    "success": False,
                    "error": f"No elastic data found for material: {material_id}"
                }
            
            # Extract elastic constants
            elastic_tensor = elastic_doc.elastic_tensor
            compliance_tensor = elastic_doc.compliance_tensor
            
            # Calculate derived properties
            bulk_modulus = elastic_doc.k_vrh
            shear_modulus = elastic_doc.g_vrh
            young_modulus = elastic_doc.e_vrh
            poisson_ratio = elastic_doc.nu_vrh
            
            return {
                "success": True,
                "material_id": material_id,
                "elastic_tensor": elastic_tensor.tolist(),
                "compliance_tensor": compliance_tensor.tolist(),
                "bulk_modulus": bulk_modulus,
                "shear_modulus": shear_modulus,
                "young_modulus": young_modulus,
                "poisson_ratio": poisson_ratio,
                "elastic_anisotropy": elastic_doc.elastic_anisotropy,
                "universal_anisotropy": elastic_doc.universal_anisotropy
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "get_elastic_properties")
            }
    
    def search_by_structure(
        self,
        structure: Dict[str, Any],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """Search Materials Project by structure similarity.
        
        Args:
            structure: Structure dictionary (from ASE or pymatgen)
            tolerance: Structure matching tolerance
            
        Returns:
            Dictionary containing similar structures
        """
        try:
            if not self.mp_client:
                return {
                    "success": False,
                    "error": "Materials Project API key not configured"
                }
            
            # Convert structure dict to pymatgen Structure
            pymatgen_structure = Structure.from_dict(structure)
            
            # Search for similar structures
            docs = self.mp_client.materials.summary.search(
                structure=pymatgen_structure,
                fields=["material_id", "formula_pretty", "structure", "energy_per_atom"]
            )
            
            if not docs:
                return {
                    "success": False,
                    "error": "No similar structures found"
                }
            
            # Calculate similarity scores
            matcher = StructureMatcher(ltol=tolerance, stol=tolerance, angle_tol=5)
            results = []
            
            for doc in docs:
                if doc.structure:
                    # Calculate similarity
                    is_similar = matcher.fit(pymatgen_structure, doc.structure)
                    if is_similar:
                        results.append({
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "energy_per_atom": doc.energy_per_atom,
                            "similar": True
                        })
            
            return {
                "success": True,
                "n_similar": len(results),
                "similar_structures": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "search_by_structure")
            }
    
    def query_nomad(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Query NOMAD database.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary containing NOMAD search results
        """
        try:
            # NOMAD API endpoint
            url = "https://nomad-lab.eu/prod/rae/api/query"
            
            # Search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "format": "json"
            }
            
            # Make request
            response = requests.get(url, params=search_params)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "success": True,
                "query": query,
                "n_results": len(data.get("results", [])),
                "results": data.get("results", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "query_nomad")
            }
    
    def get_phase_diagram(
        self,
        elements: List[str]
    ) -> Dict[str, Any]:
        """Get phase diagram for given elements.
        
        Args:
            elements: List of element symbols
            
        Returns:
            Dictionary containing phase diagram data
        """
        try:
            if not self.mp_client:
                return {
                    "success": False,
                    "error": "Materials Project API key not configured"
                }
            
            # Get phase diagram
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            from pymatgen.entries.computed import ComputedEntry
            
            # Get entries for the elements
            entries = self.mp_client.get_entries_in_chemsys(elements)
            
            if not entries:
                return {
                    "success": False,
                    "error": f"No entries found for elements: {elements}"
                }
            
            # Create phase diagram
            pd = PhaseDiagram(entries)
            
            # Get stable phases
            stable_entries = pd.stable_entries
            
            # Get convex hull
            convex_hull = []
            for entry in stable_entries:
                convex_hull.append({
                    "formula": entry.composition.reduced_formula,
                    "energy_per_atom": entry.energy_per_atom,
                    "formation_energy": pd.get_form_energy_per_atom(entry)
                })
            
            return {
                "success": True,
                "elements": elements,
                "n_stable_phases": len(stable_entries),
                "stable_phases": convex_hull
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "get_phase_diagram")
            }
    
    def compare_with_database(
        self,
        properties: Dict[str, float],
        material_formula: str,
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """Compare computed properties with database values.
        
        Args:
            properties: Dictionary of computed properties
            material_formula: Material formula
            tolerance: Tolerance for comparison
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Query Materials Project for reference data
            mp_data = self.query_materials_project(material_formula)
            
            if not mp_data["success"]:
                return {
                    "success": False,
                    "error": f"Could not retrieve reference data: {mp_data['error']}"
                }
            
            # Compare properties
            comparison = {}
            reference_material = mp_data["materials"][0]  # Use first result
            
            for prop, computed_value in properties.items():
                if prop in reference_material:
                    ref_value = reference_material[prop]
                    if ref_value is not None:
                        error = abs(computed_value - ref_value) / ref_value
                        comparison[prop] = {
                            "computed": computed_value,
                            "reference": ref_value,
                            "relative_error": error,
                            "within_tolerance": error < tolerance
                        }
            
            return {
                "success": True,
                "material": material_formula,
                "comparison": comparison,
                "tolerance": tolerance
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "compare_with_database")
            }
    
    def get_available_databases(self) -> Dict[str, Any]:
        """Get list of available databases and their capabilities.
        
        Returns:
            Dictionary containing database information
        """
        databases = {
            "materials_project": {
                "name": "Materials Project",
                "description": "Comprehensive database of computed materials properties",
                "capabilities": [
                    "Crystal structures",
                    "Formation energies",
                    "Band gaps",
                    "Elastic properties",
                    "Phase diagrams",
                    "Structure search"
                ],
                "api_key_required": True,
                "status": "available" if self.mp_client else "api_key_required"
            },
            "nomad": {
                "name": "NOMAD",
                "description": "European materials database with experimental and computed data",
                "capabilities": [
                    "Experimental data",
                    "Computed data",
                    "Structure search",
                    "Property search"
                ],
                "api_key_required": False,
                "status": "available"
            },
            "open_catalyst": {
                "name": "Open Catalyst Project",
                "description": "Database of catalyst materials and properties",
                "capabilities": [
                    "Catalyst structures",
                    "Adsorption energies",
                    "Reaction pathways",
                    "Surface properties"
                ],
                "api_key_required": False,
                "status": "not_implemented"
            }
        }
        
        return {
            "success": True,
            "databases": databases
        }
