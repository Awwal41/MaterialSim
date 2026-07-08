"""Engine- and force-field-agnostic simulation specification (v2).

The :class:`SimulationSpec` describes *what* to simulate (system, potential,
ensemble, protocol, run controls) without committing to *how* it runs. Engines,
potentials, structure builders, and protocols are resolved at run time from the
plugin registries, so no material, force field, or method is special-cased in
the core.
"""

from .simulation_spec import (
    EnsembleSpec,
    PotentialSpec,
    ProtocolSpec,
    RunSpec,
    SimulationSpec,
    SystemSpec,
)

__all__ = [
    "SimulationSpec",
    "SystemSpec",
    "PotentialSpec",
    "EnsembleSpec",
    "ProtocolSpec",
    "RunSpec",
]
