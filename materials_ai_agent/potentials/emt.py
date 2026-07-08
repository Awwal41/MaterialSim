"""EMT effective-medium potential (ASE built-in) for supported metals."""

from __future__ import annotations

from typing import Set

from .base import PotentialProvider


def _emt_elements() -> Set[str]:
    try:
        from ase.calculators.emt import parameters

        return set(parameters.keys())
    except Exception:  # noqa: BLE001
        return {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"}


class EMTPotential(PotentialProvider):
    kind = "emt"
    engines = {"ase"}
    description = "Effective Medium Theory (fast, qualitative) for select metals."

    def available(self) -> bool:
        try:
            import ase.calculators.emt  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        if bonded:
            return False
        return set(elements).issubset(_emt_elements())

    def ase_calculator(self, atoms):
        from ase.calculators.emt import EMT

        return EMT()
