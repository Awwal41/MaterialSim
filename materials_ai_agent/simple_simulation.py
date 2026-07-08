"""Backward-compatible entry point for molecular dynamics.

Historically this module contained a bespoke ASE MD loop. It is now a thin
compatibility shim: it converts the legacy keyword arguments into a
:class:`~materials_ai_agent.spec.SimulationSpec` and dispatches through the
engine-agnostic :mod:`~materials_ai_agent.orchestrator`. This keeps the CLI,
GUI, and agent tools working while the real work happens in the pluggable
engine/potential/protocol/structure registries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .orchestrator import run_spec
from .spec import SimulationSpec

logger = logging.getLogger(__name__)


def run_simple_simulation(
    material: str,
    temperature: Optional[float] = None,
    pressure: Optional[float] = None,
    n_steps: Optional[int] = None,
    force_field: Optional[str] = None,
    ensemble: Optional[str] = None,
    thermostat: Optional[str] = None,
    timestep: Optional[float] = None,
    output_frequency: Optional[int] = None,
    structure_source: str = "generate",
    structure_file: Optional[str] = None,
    mp_material_id: Optional[str] = None,
    supercell_reps: Optional[Tuple[int, int, int]] = None,
    target_atoms: int = 64,
    alloy_elements: Optional[List[str]] = None,
    alloy_fractions: Optional[List[float]] = None,
    engine: Optional[str] = None,
    protocol: Optional[str] = None,
    protocol_params: Optional[Dict[str, Any]] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Run an MD simulation from legacy keyword arguments."""
    try:
        from .core.config import Config

        config = Config.from_env()
        temperature = float(temperature if temperature is not None else config.default_temperature)
        pressure = float(pressure if pressure is not None else config.default_pressure)
        n_steps = int(n_steps if n_steps is not None else config.default_n_steps)
        ensemble = (ensemble or config.default_ensemble)
        thermostat = thermostat or config.default_thermostat
        force_field = force_field or config.default_force_field
        output_frequency = int(output_frequency or 100)

        spec = SimulationSpec.from_legacy_kwargs(
            material=material,
            temperature=temperature,
            pressure=pressure,
            n_steps=n_steps,
            force_field=force_field,
            ensemble=ensemble,
            thermostat=thermostat,
            timestep=timestep,
            output_frequency=output_frequency,
            structure_source=structure_source,
            structure_file=structure_file,
            mp_material_id=mp_material_id,
            supercell_reps=supercell_reps,
            target_atoms=target_atoms,
            alloy_elements=alloy_elements,
            alloy_fractions=alloy_fractions,
            engine=engine,
            protocol=protocol,
            protocol_params=protocol_params,
        )

        result = run_spec(spec, mp_api_key=config.mp_api_key)
        if result.get("success"):
            result.setdefault("status", "completed")
            result.setdefault("thermostat", thermostat)
            result.setdefault("temperature", temperature)
            result.setdefault("pressure", pressure)
            result.setdefault("n_steps", n_steps)
        return result

    except Exception as exc:  # noqa: BLE001
        logger.exception("Simulation failed")
        return {"success": False, "error": f"Simulation failed: {exc}"}


def run_from_spec(spec) -> Dict[str, Any]:
    """Run MD from a legacy :class:`SimulationSpec` (from ``simulation_parser``)."""
    return run_simple_simulation(**spec.to_run_kwargs())
