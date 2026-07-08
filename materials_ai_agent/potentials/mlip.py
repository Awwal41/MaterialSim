"""Machine-learned interatomic potentials (foundation models) via ASE.

These are the answer to "any system": pre-trained universal potentials that
cover most of the periodic table with near-DFT accuracy, no per-system fitting.
Each is optional; ``available`` reflects whether the package is installed.
"""

from __future__ import annotations

from typing import Set

from .base import PotentialProvider

# Elements covered by common foundation models (MP-trained: ~89 elements up to Bi).
_MLIP_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}


class _MLIPProvider(PotentialProvider):
    engines = {"ase"}

    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        return set(elements).issubset(_MLIP_ELEMENTS)


class MACEPotential(_MLIPProvider):
    kind = "mace"
    description = "MACE-MP foundation model (universal, near-DFT)."

    def available(self) -> bool:
        try:
            import mace  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def ase_calculator(self, atoms):
        from mace.calculators import mace_mp

        model = (self.spec.name if self.spec and self.spec.name else "small")
        params = dict(self.spec.params) if self.spec else {}
        return mace_mp(model=model, default_dtype="float64",
                       device=params.get("device", "cpu"))


class CHGNetPotential(_MLIPProvider):
    kind = "chgnet"
    description = "CHGNet universal potential (charge-informed, 89 elements)."

    def available(self) -> bool:
        try:
            import chgnet  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def ase_calculator(self, atoms):
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator()


class M3GNetPotential(_MLIPProvider):
    kind = "m3gnet"
    description = "M3GNet universal potential (via matgl)."

    def available(self) -> bool:
        try:
            import matgl  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def ase_calculator(self, atoms):
        import matgl
        from matgl.ext.ase import PESCalculator

        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return PESCalculator(pot)
