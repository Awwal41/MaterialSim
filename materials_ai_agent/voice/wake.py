"""Wake-word handling for hands-free voice control."""

from __future__ import annotations

import re
from typing import List, Tuple

# Phrases users can say to get the agent's attention.
WAKE_PHRASES: Tuple[str, ...] = (
    "hey material sim",
    "hey materials sim",
    "hey material sim agent",
    "material sim",
    "material sim agent",
    "hey material",
    "okay material sim",
    "ok material sim",
)

_WAKE_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|yo)\s+)?(?:material\s+sim(?:\s+agent)?|materials\s+sim)\b[,\s]*",
    re.I,
)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading wake phrase so the command body can be routed."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _WAKE_RE.sub("", cleaned, count=1).strip()
        lower = cleaned.lower()
        for phrase in WAKE_PHRASES:
            if lower.startswith(phrase):
                cleaned = cleaned[len(phrase) :].lstrip(" ,.")
                break
    return cleaned.strip(" ,.")


def contains_wake_phrase(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in WAKE_PHRASES)


def greeting_reply() -> str:
    return (
        "I'm here. Tell me what you'd like to simulate, analyze, or look up — "
        "for example, run copper at three hundred kelvin in N V T."
    )
