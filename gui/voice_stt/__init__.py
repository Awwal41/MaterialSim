"""Browser-based continuous speech recognition for Streamlit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "frontend"

voice_input = components.declare_component(
    "materialsim_voice_input",
    path=str(_FRONTEND),
)


def render_voice_input(
    *,
    auto_listen: bool = False,
    delay_ms: int = 500,
    hint: str = "",
    key: str = "voice_input",
) -> Optional[dict[str, Any]]:
    """Render the voice mic widget. Returns payload when user finishes speaking."""
    return voice_input(
        auto_listen=auto_listen,
        delay_ms=delay_ms,
        hint=hint,
        key=key,
        default=None,
    )
