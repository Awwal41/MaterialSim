"""Multi-Scale Shock Technique (LAMMPS ``fix msst``).

Simulates a steady shock wave in a small cell by evolving the simulation box
along a shock direction at a given shock velocity ``vs`` (m/s -> LAMMPS units).
"""

from __future__ import annotations

from typing import List

from .base import Protocol


class MSSTProtocol(Protocol):
    name = "msst"
    engines = {"lammps"}
    description = "Multi-Scale Shock Technique (fix msst)."

    def lammps_blocks(self, job) -> List[str]:
        ens = job.spec.ensemble
        run = job.spec.run
        T = ens.temperature
        seed = run.seed
        tdamp = ens.tdamp or 100 * run.timestep

        direction = str(self.params.get("direction", "z")).lower()
        # shock velocity in km/s (metal units use A/ps -> 1 km/s = 10 A/ps)
        vs_kms = float(self.params.get("shock_velocity_kms", 8.0))
        vs = vs_kms * 10.0  # A/ps for metal units
        q = float(self.params.get("q", 40.0))  # cell mass-like parameter
        mu = float(self.params.get("mu", 0.0))  # artificial viscosity
        n_eq = int(self.params.get("equilibration_steps", max(1000, run.n_steps // 5)))
        n_prod = max(1, run.n_steps - n_eq)

        return [
            f"velocity all create {T} {seed} dist gaussian",
            "minimize 1.0e-4 1.0e-6 1000 10000",
            "reset_timestep 0",
            f"fix equil all npt temp {T} {T} {tdamp} iso 1.0 1.0 {1000 * run.timestep}",
            f"run {n_eq}",
            "unfix equil",
            "reset_timestep 0",
            f"fix shock all msst {direction} {vs} q {q} mu {mu}",
            "fix_modify shock energy yes",
            f"run {n_prod}",
        ]
