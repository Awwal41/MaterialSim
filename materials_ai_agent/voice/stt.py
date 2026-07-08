"""Speech-to-text for the Jarvis voice interface."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_data: Union[bytes, Path, io.BytesIO],
    *,
    language: str = "en-US",
) -> Optional[str]:
    """Transcribe audio bytes or a file path to text.

    Uses ``speech_recognition`` with Google's free web API as the default
    backend. Returns ``None`` if transcription fails or dependencies are
    missing.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        logger.warning("speech_recognition not installed; voice input disabled.")
        return None

    recognizer = sr.Recognizer()

    try:
        if isinstance(audio_data, Path):
            with sr.AudioFile(str(audio_data)) as source:
                audio = recognizer.record(source)
        elif isinstance(audio_data, bytes):
            # Streamlit audio_input returns WAV bytes.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
            Path(tmp_path).unlink(missing_ok=True)
        else:
            data = audio_data.read() if hasattr(audio_data, "read") else audio_data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
            Path(tmp_path).unlink(missing_ok=True)

        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        logger.info("Could not understand audio.")
        return None
    except sr.RequestError as exc:
        logger.warning("Speech recognition service error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcription failed: %s", exc)
        return None
