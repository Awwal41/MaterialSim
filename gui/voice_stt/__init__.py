"""Browser-based continuous speech recognition for Streamlit."""

from __future__ import annotations

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
    always_on: bool = True,
    wake_only: bool = True,
    auto_listen: bool = False,
    pause_listening: bool = False,
    show_caption: bool = False,
    agent_state: str = "idle",
    status_text: str = "",
    status_live: bool = False,
    audio_b64: Optional[str] = None,
    audio_mime: str = "audio/mpeg",
    delay_ms: int = 400,
    hint: str = "",
    key: str = "voice_input",
) -> Optional[dict[str, Any]]:
    """Render the hands-free voice widget.

    Returns a payload dict when the user finishes speaking, when playback
    finishes, or on readiness / errors.
    """
    return voice_input(
        always_on=always_on,
        wake_only=wake_only,
        auto_listen=auto_listen,
        pause_listening=pause_listening,
        show_caption=show_caption,
        agent_state=agent_state,
        status_text=status_text,
        status_live=status_live,
        audio_b64=audio_b64 or "",
        audio_mime=audio_mime,
        delay_ms=delay_ms,
        hint=hint,
        key=key,
    )
