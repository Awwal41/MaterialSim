"""Text-to-speech for the Jarvis voice interface."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Natural conversational voice (ChatGPT-style clarity).
DEFAULT_VOICE = "en-US-AvaNeural"


def synthesize_speech(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Convert *text* to an MP3 file using edge-tts.

    Returns the path to the audio file, or ``None`` if TTS is unavailable.
    """
    if not text or not text.strip():
        return None

    try:
        import edge_tts
    except ImportError:
        logger.warning("edge-tts not installed; voice output disabled.")
        return None

    # Strip markdown for cleaner speech.
    clean = _strip_markdown(text)
    if len(clean) > 800:
        clean = clean[:800] + "..."

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        output_path = Path(tmp.name)
        tmp.close()

    async def _run() -> None:
        communicate = edge_tts.Communicate(clean, voice)
        await communicate.save(str(output_path))

    try:
        asyncio.run(_run())
        return output_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS synthesis failed: %s", exc)
        return None


def synthesize_to_b64(text: str, *, voice: str = DEFAULT_VOICE) -> Optional[str]:
    """Synthesize speech and return base64-encoded MP3 for browser playback."""
    import base64

    path = synthesize_speech(text, voice=voice)
    if path is None or not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting for TTS."""
    import re

    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[📊🔬📁✅❌🚀🧬⚡🤖]", "", text)
    return text.strip()
