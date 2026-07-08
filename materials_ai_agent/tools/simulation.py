"""Simulation tool wrapper (engine-agnostic).

This used to embed a hardcoded, silicon-specific LAMMPS input generator. It now
routes every request through the pluggable orchestrator so the engine, potential,
and protocol are resolved from the registries with no material special-casing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.tools import tool

from .base import BaseMaterialsTool
from ..simple_simulation import run_simple_simulation


class SimulationTool(BaseMaterialsTool):
    """Set up and run molecular dynamics simulations across engines."""

    name: str = "simulation"
    description: str = (
        "Run molecular dynamics simulations via ASE, LAMMPS, or OpenMM. "
        "Engine, potential, and protocol are selected automatically or on request."
    )

    def run_md_simulation(
        self,
        material: str,
        temperature: Optional[float] = None,
        n_steps: Optional[int] = None,
        force_field: Optional[str] = None,
        ensemble: Optional[str] = None,
        engine: Optional[str] = None,
        protocol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a complete MD simulation and return the normalized result dict."""
        try:
            return run_simple_simulation(
                material=material,
                temperature=temperature,
                n_steps=n_steps,
                force_field=force_field,
                ensemble=ensemble,
                engine=engine,
                protocol=protocol,
                **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": self._handle_error(e, "run_md_simulation")}

    def capabilities(self) -> Dict[str, Any]:
        """Report which engines and potentials are actually runnable here."""
        return {
            "engines": self.config.runnable_engines(),
            "force_fields": self.config.runnable_force_fields(),
        }


def create_simulation_tools(config) -> List:
    """Create LangChain tools for MD simulation (routes through the orchestrator)."""

    @tool
    def run_md_simulation(
        material: str,
        temperature: Optional[float] = None,
        n_steps: Optional[int] = None,
        force_field: Optional[str] = None,
        ensemble: Optional[str] = None,
        engine: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> str:
        """Run a molecular dynamics simulation (ASE/LAMMPS/OpenMM, auto-selected).

        Args:
            material: Formula (e.g. 'Cu', 'Al2O3'), SMILES, or 'mp-1234'.
            temperature: Temperature in K.
            n_steps: Number of MD steps.
            force_field: Potential kind ('auto', 'emt', 'lj', 'eam', 'tersoff',
                'mace', 'chgnet', 'opls', ...).
            ensemble: 'NVE', 'NVT', or 'NPT'.
            engine: 'auto', 'ase', 'lammps', or 'openmm'.
            protocol: 'equilibrium', 'nemd', 'msst', or 'deformation'.
        """
        result = run_simple_simulation(
            material=material,
            temperature=temperature,
            n_steps=n_steps,
            force_field=force_field,
            ensemble=ensemble,
            engine=engine,
            protocol=protocol,
        )
        if result.get("success"):
            return f"{result['message']} Output written to {result['simulation_directory']}."
        if result.get("needs_clarification"):
            return f"Need more information: {result['error']}"
        return f"Simulation failed: {result.get('error')}"

    return [run_md_simulation]
