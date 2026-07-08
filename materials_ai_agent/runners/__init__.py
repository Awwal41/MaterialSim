"""Execution backends for external-engine jobs (local, MPI/GPU, HPC)."""

from .base import CommandResult, Runner
from .local import LocalRunner
from .slurm import SlurmRunner

__all__ = ["Runner", "CommandResult", "LocalRunner", "SlurmRunner"]
