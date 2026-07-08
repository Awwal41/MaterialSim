"""Pluggable interatomic potential providers (any force field)."""

from .base import LammpsPotential, PotentialProvider
from .registry import (
    list_potentials,
    register_potential,
    resolve_potential,
)

__all__ = [
    "PotentialProvider",
    "LammpsPotential",
    "register_potential",
    "list_potentials",
    "resolve_potential",
]
