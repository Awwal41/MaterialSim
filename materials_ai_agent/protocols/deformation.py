"""Uniaxial tensile / compressive deformation (LAMMPS ``fix deform``).

Applies a constant engineering strain rate along an axis under NVT and records
the stress-strain response.
"""

from __future__ import annotations

from typing import List

from .base import Protocol


class DeformationProtocol(Protocol):
    name = "deformation"
    engines = {"lammps"}
    description = "Uniaxial deformation with stress-strain output."

    def lammps_blocks(self, job) -> List[str]:
        ens = job.spec.ensemble
        run = job.spec.run
        T = ens.temperature
        seed = run.seed
        tdamp = ens.tdamp or 100 * run.timestep

        axis = str(self.params.get("axis", "x")).lower()
        rate = float(self.params.get("strain_rate", 1.0e-3))  # 1/ps
        mode = str(self.params.get("mode", "tensile")).lower()
        erate = rate if mode != "compressive" else -rate
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
            f"fix integrate all nvt temp {T} {T} {tdamp}",
            f"fix strain all deform 1 {axis} erate {erate} remap x units box",
            f"variable strain equal (l{axis}-v_L0)/v_L0",
            f"variable stress equal -pxx",
            "fix ss all ave/time 10 100 1000 v_strain v_stress file stress_strain.txt",
            f"run {n_prod}",
        ]
