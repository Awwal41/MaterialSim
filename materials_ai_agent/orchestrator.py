"""Top-level entry point: turn a SimulationSpec into a completed run.

Flow: validate -> build system -> resolve engine/potential/protocol ->
execute via engine adapter -> assess quality. Every step fails loudly with an
actionable message rather than silently guessing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from . import bootstrap, capabilities
from .engines.base import ResolvedJob
from .engines.registry import get_engine
from .potentials.registry import resolve_potential
from .protocols.registry import get_protocol
from .runners.base import Runner
from .runners.local import LocalRunner
from .spec import SimulationSpec
from .structures.registry import build_system

logger = logging.getLogger(__name__)


def _dir_name(spec: SimulationSpec, label: str) -> str:
    safe = re.sub(r"[^\w.-]", "_", label)[:40] or "system"
    e = spec.ensemble
    parts = [safe, f"{e.temperature:g}K", e.name, spec.protocol.name, f"{spec.run.n_steps}steps"]
    if e.name.upper() == "NPT" and abs(e.pressure - 1.0) > 0.01:
        parts.insert(3, f"{e.pressure:g}atm")
    return "_".join(parts)


def run_spec(
    spec: SimulationSpec,
    *,
    mp_api_key: Optional[str] = None,
    runner: Optional[Runner] = None,
    base_dir: str = "simulations",
) -> Dict[str, Any]:
    """Validate and execute a simulation spec, returning a normalized dict."""
    bootstrap.ensure()

    try:
        # 1) Validate against the capability matrix.
        vr = capabilities.validate(spec)
        if vr.clarifications:
            return {
                "success": False,
                "needs_clarification": True,
                "error": " ".join(vr.clarifications),
                "clarifications": vr.clarifications,
            }
        if not vr.ok:
            return {
                "success": False,
                "error": " ".join(vr.errors) or "Simulation spec is not runnable.",
                "errors": vr.errors,
                "warnings": vr.warnings,
                "engine": vr.engine,
            }

        engine_name = vr.engine
        warnings = list(vr.warnings)

        # 2) Build the system (structure + optional topology).
        built = build_system(spec.system, mp_api_key=mp_api_key)
        atoms = built.atoms
        warnings.extend(built.warnings)
        label = built.label

        # 3) Resolve potential + protocol.
        elements = set(atoms.get_chemical_symbols())
        bonded = bool(built.topology and built.topology.is_bonded)
        potential = resolve_potential(
            spec.potential.kind,
            elements,
            engine=engine_name,
            bonded=bonded,
            spec=spec.potential,
        )
        warnings.extend(potential.warnings_for(elements))
        protocol = get_protocol(spec.protocol.name, spec.protocol.params)

        # 4) Prepare working directory.
        workdir = Path(base_dir) / _dir_name(spec, label)
        workdir.mkdir(parents=True, exist_ok=True)

        runner = runner or LocalRunner()
        job = ResolvedJob(
            spec=spec,
            atoms=atoms,
            topology=built.topology,
            potential=potential,
            protocol=protocol,
            workdir=workdir,
            runner=runner,
            material_label=label,
            warnings=warnings,
        )

        # 5) Execute.
        engine = get_engine(engine_name)
        result = engine.run(job)
        result_dict = result.to_dict()

        # 6) Persist a manifest and assess quality (shared analysis contract).
        _write_meta(workdir, spec, label, engine_name, potential, result_dict)
        result_dict.update(_quality(workdir, spec.ensemble.temperature))
        result_dict.setdefault("material", label)
        result_dict.setdefault("engine", engine_name)
        result_dict["potential"] = potential.kind
        result_dict["protocol"] = spec.protocol.name
        # protocol-specific extras
        try:
            result_dict.update(protocol.postprocess(workdir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Protocol post-processing failed: %s", exc)
        return result_dict

    except Exception as exc:  # noqa: BLE001
        logger.exception("Simulation failed")
        return {"success": False, "error": f"Simulation failed: {exc}"}


def _write_meta(workdir, spec, label, engine_name, potential, result_dict) -> None:
    meta = {
        "material": label,
        "engine": engine_name,
        "potential": potential.kind,
        "requested_potential": spec.potential.kind,
        "protocol": spec.protocol.name,
        "protocol_params": spec.protocol.params,
        "ensemble": spec.ensemble.name,
        "thermostat": spec.ensemble.thermostat,
        "target_temperature": spec.ensemble.temperature,
        "target_pressure_atm": spec.ensemble.pressure,
        "timestep_ps": spec.run.timestep,
        "n_steps": spec.run.n_steps,
        "production_start_step": result_dict.get("production_start_step"),
        "n_atoms": result_dict.get("n_atoms"),
        "warnings": result_dict.get("warnings", []),
    }
    (Path(workdir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _quality(workdir, temperature) -> Dict[str, Any]:
    try:
        from .simulation_quality import assess_simulation_quality

        return {"quality": assess_simulation_quality(Path(workdir), temperature)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quality assessment failed: %s", exc)
        return {}
