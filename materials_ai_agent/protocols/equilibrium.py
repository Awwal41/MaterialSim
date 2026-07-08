"""Equilibrium MD: minimize -> equilibrate -> production (NVE/NVT/NPT)."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Protocol


def _eq_steps(job) -> int:
    run = job.spec.run
    if run.equilibration_steps is not None:
        return max(0, min(run.equilibration_steps, run.n_steps - 1))
    kind = job.potential.kind.lower()
    frac = 0.30 if kind in {"lj"} else 0.15
    n_eq = max(200, int(run.n_steps * frac))
    return min(n_eq, max(1, run.n_steps - 100))


class EquilibriumProtocol(Protocol):
    name = "equilibrium"
    engines = {"ase", "lammps", "openmm"}
    description = "Standard equilibrium MD in NVE/NVT/NPT."

    # -- ASE ----------------------------------------------------------
    def ase_run(self, job, ctx) -> Dict[str, Any]:
        n_eq = _eq_steps(job)
        n_prod = job.spec.run.n_steps - n_eq

        ctx.minimize()

        ctx.set_phase("equilibration")
        ctx.thermalize()
        dyn = ctx.make_integrator(friction=0.08)
        ctx.run(dyn, n_eq)

        ctx.set_phase("production")
        ctx.thermalize()
        dyn = ctx.make_integrator(friction=0.03)
        ctx.run(dyn, n_prod)
        return {"n_equilibration_steps": n_eq, "n_production_steps": n_prod}

    # -- LAMMPS -------------------------------------------------------
    def lammps_blocks(self, job) -> List[str]:
        ens = job.spec.ensemble
        run = job.spec.run
        T = ens.temperature
        P_bar = ens.pressure * 1.01325  # atm -> bar (LAMMPS 'metal' uses bars)
        dt = run.timestep
        seed = run.seed
        thermostat = (ens.thermostat or "auto").lower()
        tdamp = ens.tdamp or 100 * dt
        pdamp = ens.pdamp or 1000 * dt

        blocks: List[str] = [
            f"velocity all create {T} {seed} dist gaussian",
            "minimize 1.0e-4 1.0e-6 1000 10000",
            "reset_timestep 0",
        ]

        name = ens.name.upper()
        if name == "NVE":
            blocks.append("fix integrate all nve")
        elif name == "NPT":
            blocks.append(
                f"fix integrate all npt temp {T} {T} {tdamp} iso {P_bar} {P_bar} {pdamp}"
            )
        else:  # NVT
            if thermostat in {"nose-hoover", "auto", "nose_hoover", "nosehoover"}:
                blocks.append(f"fix integrate all nvt temp {T} {T} {tdamp}")
            elif thermostat == "berendsen":
                blocks.append("fix integrate all nve")
                blocks.append(f"fix thermostat all temp/berendsen {T} {T} {tdamp}")
            elif thermostat == "langevin":
                blocks.append("fix integrate all nve")
                blocks.append(f"fix thermostat all langevin {T} {T} {tdamp} {seed}")
            else:
                blocks.append(f"fix integrate all nvt temp {T} {T} {tdamp}")

        n_eq = _eq_steps(job)
        n_prod = run.n_steps - n_eq
        if n_eq > 0:
            blocks.append(f"run {n_eq}")
        blocks.append(f"run {n_prod}")
        return blocks
