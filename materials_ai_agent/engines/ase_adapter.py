"""ASE engine adapter: in-process MD for small/quick runs and MLIP potentials.

Writes the shared output contract (``output.log`` with a fixed thermo header,
``trajectory.xyz``, ``structure.xyz``, ``final_structure.xyz``) consumed by the
analysis and quality modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import EngineAdapter, EngineCapabilities, EngineResult, ResolvedJob

logger = logging.getLogger(__name__)

THERMO_HEADER = "Step Temp PotEng KinEng TotEng Press Volume"


class ASEContext:
    """Helpers exposed to protocols so they can drive an ASE run uniformly."""

    def __init__(self, job: ResolvedJob):
        from ase import units

        self.job = job
        self.atoms = job.atoms
        self.units = units
        run = job.spec.run
        ens = job.spec.ensemble
        self.dt = run.timestep * 1000.0 * units.fs
        self.temperature = ens.temperature
        self.pressure_atm = ens.pressure
        self.ensemble = ens.name.upper()
        self.thermostat = (ens.thermostat or "auto").lower()
        self.barostat = (ens.barostat or "auto").lower()
        self.output_frequency = max(1, run.output_frequency)
        self._traj = None
        self._log = None
        self._symbols = self.atoms.get_chemical_symbols()
        self.n_frames = 0
        self.production_start_step: Optional[int] = None
        self._phase = "equilibration"
        self._dyn = None

    # -- lifecycle ----------------------------------------------------
    def open(self) -> None:
        wd = self.job.workdir
        from ase.io import write

        write(str(wd / "structure.xyz"), self.atoms)
        self._traj = open(wd / "trajectory.xyz", "w")
        self._log = open(wd / "output.log", "w")
        self._log.write(f"MaterialSim ASE MD ({self.ensemble})\n")
        self._log.write(f"# target_temperature_K={self.temperature}\n")
        self._log.write(f"# target_pressure_atm={self.pressure_atm}\n")
        self._log.write(THERMO_HEADER + "\n")

    def close(self) -> None:
        from ase.io import write

        if self.production_start_step is not None:
            self._log.write(f"# production_start_step={self.production_start_step}\n")
        self._log.write("Total wall time: 0:00:00\n")
        self._traj.close()
        self._log.close()
        write(str(self.job.workdir / "final_structure.xyz"), self.atoms)

    # -- primitives ---------------------------------------------------
    def minimize(self, fmax: float = 0.08, steps: int = 150) -> None:
        from ase.optimize import FIRE

        try:
            FIRE(self.atoms, logfile=None).run(fmax=fmax, steps=steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Energy minimization skipped: %s", exc)

    def thermalize(self) -> None:
        from ase.md.velocitydistribution import (
            Stationary,
            ZeroRotation,
            thermalize_momenta,
        )

        thermalize_momenta(self.atoms, temperature_K=self.temperature)
        Stationary(self.atoms)
        if not self.atoms.pbc.all():
            ZeroRotation(self.atoms)

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def make_integrator(self, friction: float = 0.03):
        self._dyn = self._build_integrator(friction)
        return self._dyn

    def run(self, dyn, n_steps: int) -> None:
        dyn.attach(self._record, interval=self.output_frequency)
        self._record()
        if n_steps > 0:
            dyn.run(n_steps)

    # -- integrator selection (honest: raises on unsupported combos) --
    def _build_integrator(self, friction: float):
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.verlet import VelocityVerlet

        dt = self.dt
        thermostat = self.thermostat
        pressure_au = self.pressure_atm * 1.01325 * self.units.bar

        if self.ensemble == "NVE":
            return VelocityVerlet(self.atoms, timestep=dt)

        if self.ensemble == "NPT":
            if not self.atoms.pbc.all() or self.atoms.cell.rank != 3:
                raise ValueError(
                    "NPT requires a fully periodic 3D cell; this system is not "
                    "periodic. Use NVT/NVE or supply a bulk/periodic structure."
                )
            if self.barostat in {"parrinello-rahman", "nose-hoover"}:
                return self._true_npt(pressure_au)
            return NPTBerendsen(
                self.atoms,
                timestep=dt,
                temperature_K=self.temperature,
                pressure_au=pressure_au,
                taut=max(50 * dt, 25.0),
                taup=max(200 * dt, 100.0),
            )

        # NVT family
        if thermostat in {"berendsen"}:
            return NVTBerendsen(self.atoms, timestep=dt, temperature_K=self.temperature,
                                taut=max(50 * dt, 25.0))
        if thermostat in {"nose-hoover", "nose_hoover", "nosehoover"}:
            return self._nose_hoover_nvt()
        if thermostat in {"langevin", "auto"}:
            return Langevin(self.atoms, timestep=dt, temperature_K=self.temperature,
                            friction=friction, fixcm=False)
        raise ValueError(
            f"ASE engine does not implement the '{thermostat}' thermostat. "
            "Use langevin, berendsen, or nose-hoover, or run this on LAMMPS."
        )

    def _nose_hoover_nvt(self):
        try:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "True Nose-Hoover requires ASE >= 3.23 (NoseHooverChainNVT). "
                f"Upgrade ASE or choose langevin/berendsen. ({exc})"
            )
        tdamp = (self.job.spec.ensemble.tdamp or 100 * (self.dt / self.units.fs)) * self.units.fs
        return NoseHooverChainNVT(
            self.atoms,
            timestep=self.dt,
            temperature_K=self.temperature,
            tdamp=tdamp,
        )

    def _true_npt(self, pressure_au: float):
        from ase.md.npt import NPT

        # ASE's NPT needs an upper-triangular cell.
        try:
            self.atoms.set_cell(self.atoms.cell.standard_form()[0], scale_atoms=True)
        except Exception:  # noqa: BLE001
            pass
        ttime = (self.job.spec.ensemble.tdamp or 25.0) * self.units.fs
        ptime = (self.job.spec.ensemble.pdamp or 75.0) * self.units.fs
        bulk_modulus = 100.0 * self.units.GPa
        pfactor = ptime ** 2 * bulk_modulus
        return NPT(
            self.atoms,
            timestep=self.dt,
            temperature_K=self.temperature,
            externalstress=pressure_au,
            ttime=ttime,
            pfactor=pfactor,
        )

    # -- recording ----------------------------------------------------
    def _pressure_bar(self) -> float:
        try:
            if not self.atoms.pbc.all() or self.atoms.cell.rank != 3:
                return 0.0
            s = self.atoms.get_stress(voigt=True)
            return float(-(s[0] + s[1] + s[2]) / 3.0 / self.units.bar)
        except Exception:  # noqa: BLE001
            return 0.0

    def _record(self) -> None:
        step = self._dyn.nsteps if self._dyn is not None else 0
        if self._phase == "production" and self.production_start_step is None:
            self.production_start_step = step
            self._log.write(f"# production_start step={step}\n")

        temp = self.atoms.get_temperature()
        pe = self.atoms.get_potential_energy()
        ke = self.atoms.get_kinetic_energy()
        press = self._pressure_bar()
        vol = self.atoms.get_volume() if self.atoms.cell.rank == 3 else 0.0
        self._log.write(
            f"{step:8d} {temp:12.4f} {pe:14.6f} {ke:14.6f} {pe + ke:14.6f} "
            f"{press:14.4f} {vol:14.4f}\n"
        )
        self._log.flush()

        if self._phase != "production":
            return
        forces = self.atoms.get_forces()
        positions = self.atoms.get_positions()
        self._traj.write(f"{len(self.atoms)}\n")
        self._traj.write(f"Step={step} Temp={temp:.4f} phase={self._phase}\n")
        for sym, pos, frc in zip(self._symbols, positions, forces):
            self._traj.write(
                f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
            )
        self._traj.flush()
        self.n_frames += 1


class ASEAdapter(EngineAdapter):
    name = "ase"

    def capabilities(self) -> EngineCapabilities:
        available = True
        note = ""
        try:
            import ase  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            available = False
            note = f"ASE not importable: {exc}"
        return EngineCapabilities(
            name="ase",
            available=available,
            ensembles={"NVE", "NVT", "NPT"},
            thermostats={"langevin", "berendsen", "nose-hoover", "none", "auto"},
            barostats={"berendsen", "parrinello-rahman", "nose-hoover", "auto"},
            protocols={"equilibrium"},
            potential_kinds={"emt", "lj", "mace", "chgnet", "m3gnet", "auto"},
            notes=note,
        )

    def run(self, job: ResolvedJob) -> EngineResult:
        try:
            calc = job.potential.ase_calculator(job.atoms)
            job.atoms.calc = calc
        except Exception as exc:  # noqa: BLE001
            return EngineResult(
                success=False, engine=self.name, workdir=str(job.workdir),
                error=f"Could not attach potential '{job.potential.kind}': {exc}",
            )

        ctx = ASEContext(job)
        ctx.open()
        try:
            job.protocol.ase_run(job, ctx)
        except Exception as exc:  # noqa: BLE001
            ctx.close()
            logger.exception("ASE protocol run failed")
            return EngineResult(
                success=False, engine=self.name, workdir=str(job.workdir),
                error=f"ASE run failed: {exc}", warnings=job.warnings,
            )
        ctx.close()

        output_files = [str(f) for f in sorted(job.workdir.glob("*"))]
        msg = (
            f"Completed {job.spec.run.n_steps}-step {ctx.ensemble} {job.spec.protocol.name} "
            f"run of {job.material_label} ({len(job.atoms)} atoms) at "
            f"{ctx.temperature:g} K using the {job.potential.kind} potential (ASE)."
        )
        return EngineResult(
            success=True,
            engine=self.name,
            workdir=str(job.workdir),
            message=msg,
            n_atoms=len(job.atoms),
            n_frames=ctx.n_frames,
            production_start_step=ctx.production_start_step,
            output_files=output_files,
            warnings=job.warnings,
            extra={
                "temperature": ctx.temperature,
                "pressure": ctx.pressure_atm,
                "ensemble": ctx.ensemble,
                "n_steps": job.spec.run.n_steps,
                "timestep": job.spec.run.timestep,
                "force_field": job.potential.kind,
            },
        )
