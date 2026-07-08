"""Local execution, optionally under MPI (mpirun/mpiexec) for scale."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .base import CommandResult, Runner


class LocalRunner(Runner):
    """Run a command as a local subprocess, optionally parallelized with MPI."""

    name = "local"

    def __init__(self, mpi_ranks: int = 1, mpi_launcher: Optional[str] = None):
        self.mpi_ranks = max(1, int(mpi_ranks))
        self.mpi_launcher = mpi_launcher

    def _wrap_mpi(self, command: List[str]) -> List[str]:
        if self.mpi_ranks <= 1:
            return command
        launcher = self.mpi_launcher or shutil.which("mpirun") or shutil.which("mpiexec")
        if not launcher:
            # No MPI available; run serially rather than failing.
            return command
        return [launcher, "-np", str(self.mpi_ranks), *command]

    def run(
        self,
        command: List[str],
        workdir: Path,
        *,
        log_file: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        full = self._wrap_mpi(list(command))
        start = time.time()
        try:
            proc = subprocess.run(
                full,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            return CommandResult(returncode=127, stderr=str(exc), elapsed_s=time.time() - start)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(returncode=124, stderr=f"Timed out: {exc}", elapsed_s=time.time() - start)

        if log_file:
            (workdir / log_file).write_text(
                (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
            )
        return CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            elapsed_s=time.time() - start,
        )
