"""Optional HPC execution: submit a batch script via Slurm (sbatch).

Used for cluster-scale LAMMPS/OpenMM runs. Falls back cleanly when ``sbatch``
is not on PATH so the platform still works on a laptop.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .base import CommandResult, Runner


class SlurmRunner(Runner):
    """Write and submit an sbatch script, optionally waiting for completion."""

    name = "slurm"

    def __init__(
        self,
        *,
        nodes: int = 1,
        ntasks: int = 1,
        partition: Optional[str] = None,
        time_limit: str = "01:00:00",
        modules: Optional[List[str]] = None,
        wait: bool = True,
    ):
        self.nodes = nodes
        self.ntasks = ntasks
        self.partition = partition
        self.time_limit = time_limit
        self.modules = modules or []
        self.wait = wait

    @staticmethod
    def available() -> bool:
        return shutil.which("sbatch") is not None

    def _script(self, command: List[str]) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.ntasks}",
            f"#SBATCH --time={self.time_limit}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        lines.append("#SBATCH --output=slurm-%j.out")
        lines.append("")
        for mod in self.modules:
            lines.append(f"module load {mod}")
        launcher = "srun" if shutil.which("srun") else ""
        cmd = " ".join(command)
        lines.append(f"{launcher} {cmd}".strip())
        lines.append("")
        return "\n".join(lines)

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
        script = workdir / "job.slurm"
        script.write_text(self._script(command), encoding="utf-8")

        if not self.available():
            return CommandResult(
                returncode=127,
                stderr="sbatch not found; SlurmRunner requires a Slurm scheduler.",
            )

        args = ["sbatch"]
        if self.wait:
            args.append("--wait")
        args.append(script.name)
        start = time.time()
        proc = subprocess.run(
            args, cwd=str(workdir), capture_output=True, text=True, timeout=timeout
        )
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
