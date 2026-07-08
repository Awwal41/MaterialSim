"""Non-equilibrium MD for transport coefficients (LAMMPS).

Two flavours selected via ``params['mode']``:
  - ``thermal`` (default): Muller-Plathe reverse NEMD (``fix thermal/conductivity``)
    to drive a heat flux and measure a temperature gradient.
  - ``shear``: SLLOD + fix deform for viscosity from an imposed shear rate.
"""

from __future__ import annotations

from typing import List

from .base import Protocol


class NEMDProtocol(Protocol):
    name = "nemd"
    engines = {"lammps"}
    description = "Non-equilibrium MD (thermal conductivity / viscosity)."

    def lammps_blocks(self, job) -> List[str]:
        ens = job.spec.ensemble
        run = job.spec.run
        T = ens.temperature
        seed = run.seed
        tdamp = ens.tdamp or 100 * run.timestep
        mode = str(self.params.get("mode", "thermal")).lower()
        n_eq = int(self.params.get("equilibration_steps", max(1000, run.n_steps // 5)))
        n_prod = max(1, run.n_steps - n_eq)

        blocks: List[str] = [
            f"velocity all create {T} {seed} dist gaussian",
            "minimize 1.0e-4 1.0e-6 1000 10000",
            "reset_timestep 0",
            f"fix equil all nvt temp {T} {T} {tdamp}",
            f"run {n_eq}",
            "unfix equil",
            "reset_timestep 0",
        ]

        if mode == "shear":
            rate = float(self.params.get("shear_rate", 1.0e-4))
            blocks += [
                "fix integrate all nvt/sllod temp {T} {T} {td}".format(T=T, td=tdamp),
                f"fix deform all deform 1 xy erate {rate} remap v",
                "compute pxy all pressure NULL virial",
                "variable pxy equal c_pxy[4]",
                "fix avg all ave/time 10 100 1000 v_pxy file viscosity.profile",
                f"run {n_prod}",
            ]
        else:  # thermal (reverse NEMD, Muller-Plathe)
            nbins = int(self.params.get("n_bins", 20))
            swap = int(self.params.get("swap_every", 100))
            blocks += [
                f"fix integrate all nve",
                f"fix hot all langevin {T} {T} {tdamp} {seed} tally yes",
                f"compute ke all ke/atom",
                "variable temp atom c_ke/1.5",
                f"fix flux all thermal/conductivity {swap} z {nbins}",
                f"fix profile all ave/chunk 10 100 1000 all bin/1d z lower "
                f"{1.0 / nbins} v_temp file temp.profile",
                f"run {n_prod}",
            ]
        return blocks
