"""Runner interface: how an external engine command is executed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0


class Runner(ABC):
    """Executes a command in a working directory."""

    name: str = "runner"

    @abstractmethod
    def run(
        self,
        command: List[str],
        workdir: Path,
        *,
        log_file: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        ...
