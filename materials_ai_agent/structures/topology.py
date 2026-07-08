"""Bonded-system topology (bonds/angles/dihedrals/charges).

Only needed for molecular/polymer/biomolecular force fields (OPLS, GAFF, ...).
Crystalline/metallic systems leave this ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Topology:
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    angles: List[Tuple[int, int, int]] = field(default_factory=list)
    dihedrals: List[Tuple[int, int, int, int]] = field(default_factory=list)
    impropers: List[Tuple[int, int, int, int]] = field(default_factory=list)
    charges: Optional[List[float]] = None
    atom_types: Optional[List[str]] = None
    residues: Optional[List[str]] = None
    molecule_ids: Optional[List[int]] = None
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def is_bonded(self) -> bool:
        return bool(self.bonds)
