"""Classical many-body potentials that run through LAMMPS.

None of these embed a specific element or file: they resolve a matching
potential file from a search path (``MATERIALSIM_POTENTIALS_DIR``, ``./potentials``,
or ``$LAMMPS_POTENTIALS``) or a user-supplied ``PotentialSpec.file``. If no file
can be found for the requested chemistry, ``supports`` returns False and the
registry raises an actionable error instead of guessing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

from .base import LammpsPotential, PotentialProvider


def _search_dirs() -> List[Path]:
    dirs: List[Path] = []
    for env in ("MATERIALSIM_POTENTIALS_DIR", "LAMMPS_POTENTIALS"):
        val = os.getenv(env)
        if val:
            dirs.append(Path(val))
    dirs.append(Path("potentials"))
    return [d for d in dirs if d.exists()]


def _find_file(patterns: List[str], elements: Set[str], explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    els = {e.lower() for e in elements}
    best: Optional[Path] = None
    for d in _search_dirs():
        for pattern in patterns:
            for cand in d.glob(pattern):
                name = cand.name.lower()
                if all(e in name for e in els):
                    return cand
                if best is None:
                    best = cand
    return best  # may be None; a generic file may still list the elements internally


class _FileBackedLAMMPSPotential(PotentialProvider):
    engines = {"lammps"}
    _patterns: List[str] = []

    def available(self) -> bool:
        # Available if we might resolve a file (search dir exists) or user gave one.
        if self.spec and getattr(self.spec, "file", None):
            return True
        return bool(_search_dirs())

    def _resolve_file(self, elements: Set[str]) -> Optional[Path]:
        explicit = getattr(self.spec, "file", None) if self.spec else None
        return _find_file(self._patterns, elements, explicit)

    def supports(self, elements: Set[str], *, bonded: bool = False) -> bool:
        if bonded:
            return False
        return self._resolve_file(elements) is not None


class EAMPotential(_FileBackedLAMMPSPotential):
    kind = "eam"
    description = "Embedded Atom Method for metals/alloys (LAMMPS eam/alloy)."
    _patterns = ["*.eam.alloy", "*.eam.fs", "*.eam"]

    def lammps_potential(self, job) -> LammpsPotential:
        order = sorted(set(job.atoms.get_chemical_symbols()))
        f = self._resolve_file(set(order))
        if f is None:
            raise FileNotFoundError(
                "No EAM potential file found. Place one (e.g. Cu.eam.alloy) in "
                "./potentials or set MATERIALSIM_POTENTIALS_DIR."
            )
        style = "eam/alloy" if f.suffix != ".eam" else "eam"
        if style == "eam":
            coeff = [f"pair_coeff * * {f.name}"]
        else:
            coeff = [f"pair_coeff * * {f.name} " + " ".join(order)]
        return LammpsPotential(units="metal", atom_style="atomic",
                               pair_style=style, pair_coeff=coeff,
                               extra_commands=[f"# using {f}"])


class TersoffPotential(_FileBackedLAMMPSPotential):
    kind = "tersoff"
    description = "Tersoff bond-order potential for covalent solids (Si, C, ...)."
    _patterns = ["*.tersoff", "*.tersoff.*"]

    def lammps_potential(self, job) -> LammpsPotential:
        order = sorted(set(job.atoms.get_chemical_symbols()))
        f = self._resolve_file(set(order))
        if f is None:
            raise FileNotFoundError("No Tersoff potential file found (see ./potentials).")
        return LammpsPotential(units="metal", atom_style="atomic",
                               pair_style="tersoff",
                               pair_coeff=[f"pair_coeff * * {f.name} " + " ".join(order)])


class MEAMPotential(_FileBackedLAMMPSPotential):
    kind = "meam"
    description = "Modified EAM (LAMMPS meam/c): library + parameter files."
    _patterns = ["library.meam", "*.meam"]

    def lammps_potential(self, job) -> LammpsPotential:
        order = sorted(set(job.atoms.get_chemical_symbols()))
        lib = _find_file(["library.meam", "*library*.meam"], set(order),
                         getattr(self.spec, "file", None) if self.spec else None)
        par = _find_file(["*.meam"], set(order), None)
        if lib is None or par is None:
            raise FileNotFoundError(
                "MEAM needs a library file and a parameter file in ./potentials."
            )
        elems = " ".join(order)
        return LammpsPotential(
            units="metal", atom_style="atomic", pair_style="meam/c",
            pair_coeff=[f"pair_coeff * * {lib.name} {elems} {par.name} {elems}"],
        )


class ReaxFFPotential(_FileBackedLAMMPSPotential):
    kind = "reaxff"
    description = "ReaxFF reactive force field (needs charge equilibration)."
    _patterns = ["ffield.reax*", "*.reax", "reax*"]

    def lammps_potential(self, job) -> LammpsPotential:
        order = sorted(set(job.atoms.get_chemical_symbols()))
        f = self._resolve_file(set(order))
        if f is None:
            raise FileNotFoundError("No ReaxFF ffield file found (see ./potentials).")
        return LammpsPotential(
            units="real", atom_style="charge", requires_charges=True,
            pair_style="reaxff NULL",
            pair_coeff=[f"pair_coeff * * {f.name} " + " ".join(order)],
            extra_commands=["fix qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff"],
        )
