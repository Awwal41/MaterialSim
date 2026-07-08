"""Generic Lennard-Jones (ASE and LAMMPS).

Element parameters come from a small tuned table with a physically-motivated
fallback derived from covalent radii, so any element gets *some* reasonable
(qualitative) LJ description rather than a hardcoded single default.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .base import LammpsPotential, PotentialProvider

_LJ_BY_ELEMENT: Dict[str, Tuple[float, float]] = {
    "Si": (0.05, 2.35), "C": (0.05, 2.0), "Ge": (0.05, 2.45),
    "Cu": (0.015, 2.55), "Al": (0.012, 2.65), "Fe": (0.015, 2.45),
    "Ni": (0.015, 2.50), "Au": (0.015, 2.90), "Ag": (0.015, 2.90),
    "Pt": (0.015, 2.80), "Pd": (0.015, 2.75),
    "Ar": (0.0103, 3.40), "Ne": (0.0031, 2.74), "Kr": (0.0140, 3.65),
}
_COVALENT = frozenset({"Si", "C", "Ge", "Sn"})
_RC = 8.5


def lj_params(element: str) -> Tuple[float, float]:
    if element in _LJ_BY_ELEMENT:
        return _LJ_BY_ELEMENT[element]
    try:
        from ase.data import atomic_numbers, covalent_radii

        r = covalent_radii[atomic_numbers[element]]
        sigma = 2.0 * r / (2.0 ** (1.0 / 6.0))  # place LJ minimum at ~2r
        return (0.010, float(sigma))
    except Exception:  # noqa: BLE001
        return (0.0103, 3.4)


class LennardJonesPotential(PotentialProvider):
    kind = "lj"
    engines = {"ase", "lammps"}
    description = "Lennard-Jones (12-6). Qualitative for non-noble-gas systems."

    def available(self) -> bool:
        return True

    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        return not bonded  # LJ works for anything as a last-resort pair potential

    def warnings_for(self, elements: Set[str]) -> List[str]:
        w = []
        cov = set(elements) & _COVALENT
        if cov:
            w.append(
                f"Lennard-Jones cannot capture directional covalent bonding in "
                f"{', '.join(sorted(cov))}; results are qualitative. Prefer a "
                "Tersoff/MEAM potential or an MLIP (MACE/CHGNet)."
            )
        return w

    def ase_calculator(self, atoms):
        from ase.calculators.lj import LennardJones

        symbols = sorted(set(atoms.get_chemical_symbols()))
        eps, sig = lj_params(symbols[0]) if symbols else (0.0103, 3.4)
        return LennardJones(epsilon=eps, sigma=sig, rc=_RC)

    def lammps_potential(self, job) -> LammpsPotential:
        order = sorted(set(job.atoms.get_chemical_symbols()))
        coeffs: List[str] = []
        for i, el in enumerate(order, start=1):
            eps, sig = lj_params(el)
            coeffs.append(f"pair_coeff {i} {i} {eps} {sig}")
        return LammpsPotential(
            units="metal",
            atom_style="atomic",
            pair_style=f"lj/cut {_RC}",
            pair_coeff=coeffs,
            extra_commands=["pair_modify mix geometric tail yes"],
        )
