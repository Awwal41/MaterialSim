"""molecular dynamics engine built on ASE.

This module runs molecular dynamics simulations using ASE's built-in
integrators and interatomic potentials (EMT for supported metals/light elements,
element-tuned Lennard-Jones as a fallback). It performs energy minimization,
an equilibration phase, and a production phase, then writes trajectory and
thermodynamic output files plus a ``meta.json`` quality manifest.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms, units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import Stationary, ZeroRotation, thermalize_momenta
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from .md.potentials import (
    recommended_equilibration_fraction,
    recommended_timestep_ps,
    select_calculator,
)
from .simulation_quality import assess_simulation_quality
from .structure_builder import build_atoms, infer_material_label, normalize_structure_source

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
    **_ignored: Any,
) -> Dict[str, Any]:
    """Run a real molecular dynamics simulation with equilibration + production."""
    try:
        from .core.config import Config
        from .core.materials_database import MaterialsDatabase

        config = Config.from_env()
        materials_db = MaterialsDatabase()

        temperature = float(temperature if temperature is not None else config.default_temperature)
        pressure = float(pressure if pressure is not None else config.default_pressure)
        n_steps = int(n_steps if n_steps is not None else config.default_n_steps)
        force_field = force_field or config.default_force_field
        ensemble = (ensemble or config.default_ensemble).upper()
        thermostat = thermostat or config.default_thermostat
        output_frequency = int(output_frequency or 100)
        output_frequency = max(1, min(output_frequency, max(1, n_steps)))
        structure_source = normalize_structure_source(structure_source)

        atoms = build_atoms(
            material,
            structure_source=structure_source,
            structure_file=structure_file,
            mp_material_id=mp_material_id,
            supercell_reps=supercell_reps,
            target_atoms=target_atoms,
            alloy_elements=alloy_elements,
            alloy_fractions=alloy_fractions,
            mp_api_key=config.mp_api_key,
        )

        if material in {"", "custom", "user", "uploaded"} or structure_file or mp_material_id:
            material = infer_material_label(atoms)

        sim_dir = Path("simulations") / _simulation_dir_name(
            material, temperature, ensemble, pressure, n_steps, len(alloy_elements or [])
        )
        sim_dir.mkdir(parents=True, exist_ok=True)

        material_props = materials_db.get_material(material)
        calculator, used_force_field, potential_warnings = select_calculator(
            atoms, force_field, material
        )

        if timestep is None and material_props is not None:
            timestep = material_props.recommended_timestep
        if timestep is None:
            timestep = recommended_timestep_ps(material, used_force_field)

        atoms.calc = calculator
        write(str(sim_dir / "structure.xyz"), atoms)

        n_eq = max(
            200,
            int(n_steps * recommended_equilibration_fraction(used_force_field, material)),
        )
        n_eq = min(n_eq, max(1, n_steps - 100))
        n_prod = n_steps - n_eq

        n_frames, production_start_step = _run_md(
            atoms=atoms,
            sim_dir=sim_dir,
            temperature=temperature,
            pressure_atm=pressure,
            n_equilibration_steps=n_eq,
            n_production_steps=n_prod,
            ensemble=ensemble,
            thermostat=thermostat,
            timestep_ps=timestep,
            output_frequency=output_frequency,
        )

        meta = {
            "material": material,
            "target_temperature": temperature,
            "target_pressure_atm": pressure,
            "ensemble": ensemble,
            "thermostat": thermostat,
            "force_field": used_force_field,
            "requested_force_field": force_field,
            "timestep_ps": timestep,
            "n_atoms": len(atoms),
            "structure_source": structure_source,
            "structure_file": structure_file,
            "mp_material_id": mp_material_id,
            "supercell_reps": list(supercell_reps) if supercell_reps else None,
            "target_atoms": target_atoms,
            "alloy_elements": alloy_elements,
            "alloy_fractions": alloy_fractions,
            "n_equilibration_steps": n_eq,
            "n_production_steps": n_prod,
            "production_start_step": production_start_step,
            "warnings": potential_warnings,
        }
        (sim_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        quality = assess_simulation_quality(sim_dir, temperature)
        output_files = [str(f) for f in sorted(sim_dir.glob("*"))]

        message = (
            f"Completed {n_steps}-step {ensemble} MD simulation of {material} "
            f"({len(atoms)} atoms) at {temperature:g} K"
        )
        if ensemble == "NPT":
            message += f" and {pressure:g} atm"
        message += f" using the {used_force_field} potential."
        if not quality.get("converged"):
            message += " Warning: the run did not fully equilibrate — see quality report."

        return {
            "success": True,
            "material": material,
            "temperature": temperature,
            "pressure": pressure,
            "n_steps": n_steps,
            "force_field": used_force_field,
            "ensemble": ensemble,
            "thermostat": thermostat,
            "timestep": timestep,
            "n_atoms": len(atoms),
            "n_frames": n_frames,
            "simulation_directory": str(sim_dir),
            "output_files": output_files,
            "status": "completed",
            "message": message,
            "warnings": potential_warnings,
            "quality": quality,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Simulation failed")
        return {"success": False, "error": f"Simulation failed: {exc}"}


def _simulation_dir_name(
    material: str,
    temperature: float,
    ensemble: str,
    pressure_atm: float,
    n_steps: int,
    n_alloy_elements: int,
) -> str:
    """Build a filesystem-safe directory name for a run."""
    safe_material = re.sub(r"[^\w.-]", "_", material)[:40]
    parts = [safe_material, f"{temperature:g}K", ensemble, f"{n_steps}steps"]
    if ensemble == "NPT" and abs(pressure_atm - 1.0) > 0.01:
        parts.insert(3, f"{pressure_atm:g}atm")
    if n_alloy_elements >= 2:
        parts.insert(1, "alloy")
    return "_".join(parts)


def _run_md(
    atoms: Atoms,
    sim_dir: Path,
    temperature: float,
    pressure_atm: float,
    n_equilibration_steps: int,
    n_production_steps: int,
    ensemble: str,
    thermostat: str,
    timestep_ps: float,
    output_frequency: int,
) -> tuple[int, Optional[int]]:
    """Minimize, equilibrate, then run production MD."""
    _minimize_structure(atoms)

    dt = timestep_ps * 1000.0 * units.fs
    trajectory_file = sim_dir / "trajectory.xyz"
    log_file = sim_dir / "output.log"
    symbols = atoms.get_chemical_symbols()
    frame_counter = {"n": 0}
    production_start_step: Optional[int] = None

    with open(trajectory_file, "w") as traj, open(log_file, "w") as log:
        log.write(f"LAMMPS-style MD log generated by MaterialSim (ASE {ensemble})\n")
        log.write(f"# target_temperature_K={temperature}\n")
        log.write(f"# target_pressure_atm={pressure_atm}\n")
        log.write("Step Temp PotEng KinEng TotEng Press Volume\n")

        def _record(mark_production: bool = True) -> None:
            nonlocal production_start_step
            step = dyn.nsteps
            if mark_production and production_start_step is None and phase["name"] == "production":
                production_start_step = step
                log.write(f"# production_start step={step}\n")

            temp = atoms.get_temperature()
            pe = atoms.get_potential_energy()
            ke = atoms.get_kinetic_energy()
            press = _pressure_bar(atoms)
            vol = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
            log.write(
                f"{step:8d} {temp:12.4f} {pe:14.6f} {ke:14.6f} "
                f"{pe + ke:14.6f} {press:14.4f} {vol:14.4f}\n"
            )
            log.flush()

            if mark_production:
                forces = atoms.get_forces()
                positions = atoms.get_positions()
                traj.write(f"{len(atoms)}\n")
                traj.write(f"Step={step} Temp={temp:.4f} phase={phase['name']}\n")
                for sym, pos, frc in zip(symbols, positions, forces):
                    traj.write(
                        f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                        f"{frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
                    )
                traj.flush()
                frame_counter["n"] += 1

        phase = {"name": "equilibration"}

        # --- Equilibration: strong thermostat coupling ---
        thermalize_momenta(atoms, temperature_K=temperature)
        _zero_drift(atoms)
        dyn = _make_integrator(
            atoms, ensemble, thermostat, dt, temperature, pressure_atm, friction=0.08
        )
        dyn.attach(lambda: _record(mark_production=False), interval=output_frequency)
        _record(mark_production=False)
        if n_equilibration_steps > 0:
            dyn.run(n_equilibration_steps)

        # --- Production ---
        phase["name"] = "production"
        thermalize_momenta(atoms, temperature_K=temperature)
        _zero_drift(atoms)
        dyn = _make_integrator(
            atoms, ensemble, thermostat, dt, temperature, pressure_atm, friction=0.03
        )
        dyn.attach(lambda: _record(mark_production=True), interval=output_frequency)
        _record(mark_production=True)
        if n_production_steps > 0:
            dyn.run(n_production_steps)

        if production_start_step is not None:
            log.write(f"# production_start_step={production_start_step}\n")
        log.write("Total wall time: 0:00:00\n")

    write(str(sim_dir / "final_structure.xyz"), atoms)
    return frame_counter["n"], production_start_step


def _minimize_structure(atoms: Atoms, steps: int = 150) -> None:
    """Relax atomic positions before dynamics to reduce initial energy spikes."""
    try:
        optimizer = FIRE(atoms, logfile=None)
        optimizer.run(fmax=0.08, steps=steps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Energy minimization skipped: %s", exc)


def _zero_drift(atoms: Atoms) -> None:
    Stationary(atoms)
    if not atoms.pbc.all():
        ZeroRotation(atoms)


def _make_integrator(
    atoms: Atoms,
    ensemble: str,
    thermostat: str,
    dt: float,
    temperature: float,
    pressure_atm: float = 1.0,
    friction: float = 0.03,
):
    """Choose an ASE integrator based on ensemble and thermostat."""
    thermostat_l = (thermostat or "").lower()
    pressure_bar = pressure_atm * 1.01325 * units.bar

    if ensemble == "NVE":
        return VelocityVerlet(atoms, timestep=dt)

    if ensemble == "NPT":
        return NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            pressure_au=pressure_bar,
            taut=max(50 * dt, 25.0),
            taup=max(200 * dt, 100.0),
        )

    if "berendsen" in thermostat_l:
        return NVTBerendsen(
            atoms, timestep=dt, temperature_K=temperature, taut=max(50 * dt, 25.0)
        )

    return Langevin(
        atoms,
        timestep=dt,
        temperature_K=temperature,
        friction=friction,
        fixcm=False,
    )


def _pressure_bar(atoms: Atoms) -> float:
    """Return scalar pressure in bar, or 0.0 if stress is unavailable."""
    try:
        if not atoms.pbc.all() or atoms.cell.rank != 3:
            return 0.0
        stress = atoms.get_stress(voigt=True)
        pressure_eva3 = -(stress[0] + stress[1] + stress[2]) / 3.0
        return float(pressure_eva3 / units.bar)
    except Exception:  # noqa: BLE001
        return 0.0


def run_from_spec(spec) -> Dict[str, Any]:
    """Run MD from a :class:`~materials_ai_agent.simulation_spec.SimulationSpec`."""
    return run_simple_simulation(**spec.to_run_kwargs())
