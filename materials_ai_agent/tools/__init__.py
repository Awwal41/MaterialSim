"""Tools for the Materials AI Agent.

Simulation and analysis tools are always importable. Database and ML tools
depend on optional packages (mp-api/pymatgen, torch/scikit-learn); they are
imported defensively so that a missing optional dependency does not break the
whole package.
"""

from .agent_tools import create_agent_tools

__all__ = ["create_agent_tools"]

try:  # optional: requires ase (installed) and LAMMPS for real runs
    from .simulation import SimulationTool

    __all__.append("SimulationTool")
except Exception:  # noqa: BLE001
    SimulationTool = None  # type: ignore

try:
    from .analysis import AnalysisTool

    __all__.append("AnalysisTool")
except Exception:  # noqa: BLE001
    AnalysisTool = None  # type: ignore

try:  # optional: requires mp-api + pymatgen
    from .database import DatabaseTool

    __all__.append("DatabaseTool")
except Exception:  # noqa: BLE001
    DatabaseTool = None  # type: ignore

try:  # optional: requires torch + scikit-learn
    from .ml import MLTool

    __all__.append("MLTool")
except Exception:  # noqa: BLE001
    MLTool = None  # type: ignore

try:
    from .visualization import VisualizationTool

    __all__.append("VisualizationTool")
except Exception:  # noqa: BLE001
    VisualizationTool = None  # type: ignore
