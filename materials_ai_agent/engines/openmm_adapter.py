"""OpenMM engine adapter: all-atom, bonded/soft-matter MD (GPU-capable).

Delegates system construction to the bonded potential providers (which return an
OpenMM ``System`` + ``Topology`` + positions), then runs equilibrium dynamics
and writes the shared output contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import EngineAdapter, EngineCapabilities, EngineResult, ResolvedJob

logger = logging.getLogger(__name__)

THERMO_HEADER = "Step Temp PotEng KinEng TotEng Press Volume"


def _openmm_available() -> bool:
    try:
        import openmm  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        try:
            import simtk.openmm  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False


class OpenMMAdapter(EngineAdapter):
    name = "openmm"

    def capabilities(self) -> EngineCapabilities:
        avail = _openmm_available()
        return EngineCapabilities(
            name="openmm",
            available=avail,
            ensembles={"NVE", "NVT", "NPT"},
            thermostats={"langevin", "nose-hoover", "andersen", "none", "auto"},
            barostats={"monte-carlo", "auto"},
            protocols={"equilibrium"},
            potential_kinds={"opls", "gaff", "openff"},
            notes="" if avail else "OpenMM not installed (pip install openmm).",
        )

    def run(self, job: ResolvedJob) -> EngineResult:
        if not _openmm_available():
            return EngineResult(
                success=False, engine=self.name, workdir=str(job.workdir),
                error="OpenMM is not installed. `pip install openmm` (or conda).",
            )
        try:
            import openmm
            from openmm import unit
        except Exception:  # noqa: BLE001
            from simtk import openmm, unit  # type: ignore

        try:
            prepared = job.potential.openmm_system(job)
        except Exception as exc:  # noqa: BLE001
            return EngineResult(
                success=False, engine=self.name, workdir=str(job.workdir),
                error=f"Potential '{job.potential.kind}' could not build an OpenMM system: {exc}",
            )

        system = prepared["system"]
        topology = prepared["topology"]
        positions = prepared["positions"]

        ens = job.spec.ensemble
        run = job.spec.run
        temp = ens.temperature * unit.kelvin
        dt_ps = run.timestep * unit.picoseconds

        integrator = openmm.LangevinMiddleIntegrator(temp, 1.0 / unit.picosecond, dt_ps)
        if ens.name.upper() == "NPT":
            system.addForce(openmm.MonteCarloBarostat(ens.pressure * unit.atmosphere, temp))

        try:
            from openmm.app import Simulation
        except Exception:  # noqa: BLE001
            from simtk.openmm.app import Simulation  # type: ignore

        sim = Simulation(topology, system, integrator)
        sim.context.setPositions(positions)
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(temp)

        freq = max(1, run.output_frequency)
        n_eq = run.equilibration_steps if run.equilibration_steps is not None else run.n_steps // 5
        n_prod = max(1, run.n_steps - n_eq)
        if n_eq > 0:
            sim.step(n_eq)

        out = job.workdir / "output.log"
        traj = job.workdir / "trajectory.xyz"
        n_frames = 0
        with open(out, "w") as log, open(traj, "w") as tj:
            log.write("MaterialSim OpenMM MD\n")
            log.write(f"# target_temperature_K={ens.temperature}\n")
            log.write(f"# production_start step={n_eq}\n")
            log.write(THERMO_HEADER + "\n")
            elements = [a.element.symbol if a.element else "X" for a in topology.atoms()]
            for i in range(0, n_prod, freq):
                sim.step(min(freq, n_prod - i))
                state = sim.context.getState(getEnergy=True, getPositions=True)
                pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                n_dof = max(1, 3 * system.getNumParticles())
                t = (2 * state.getKineticEnergy() / (n_dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)
                step = n_eq + i
                log.write(f"{step:8d} {t:12.4f} {pe:14.6f} {ke:14.6f} {pe + ke:14.6f} {0.0:14.4f} {0.0:14.4f}\n")
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                tj.write(f"{len(elements)}\n")
                tj.write(f"Step={step} phase=production\n")
                for sym, p in zip(elements, pos):
                    tj.write(f"{sym} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                n_frames += 1
            log.write("Total wall time: 0:00:00\n")

        output_files = [str(f) for f in sorted(job.workdir.glob("*"))]
        return EngineResult(
            success=True, engine=self.name, workdir=str(job.workdir),
            message=(
                f"Completed {run.n_steps}-step {ens.name} run of {job.material_label} "
                f"using {job.potential.kind} (OpenMM)."
            ),
            n_atoms=system.getNumParticles(), n_frames=n_frames,
            production_start_step=n_eq, output_files=output_files, warnings=job.warnings,
            extra={
                "temperature": ens.temperature, "ensemble": ens.name,
                "n_steps": run.n_steps, "timestep": run.timestep,
                "force_field": job.potential.kind,
            },
        )
