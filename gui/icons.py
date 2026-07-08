"""Central icon definitions — Streamlit Material Symbols.

Use these shortcodes everywhere instead of ad-hoc emoji so the UI stays
consistent across navigation, chat, buttons, and status messages.
"""

from __future__ import annotations

# Branding
APP = ":material/hub:"

# Primary navigation
NAV_CHAT = ":material/chat:"
NAV_VOICE = ":material/graphic_eq:"
NAV_RESULTS = ":material/insert_chart:"
NAV_SETTINGS = ":material/settings:"

# Alias used across chat/sidebar surfaces
CHAT = NAV_CHAT

# Chat avatars
AVATAR_USER = ":material/account_circle:"
AVATAR_AGENT = ":material/auto_awesome:"

# Actions & sections
CHART = ":material/insert_chart:"
SCIENCE = ":material/science:"
FOLDER = ":material/folder:"
DOWNLOAD = ":material/download:"
MIC = ":material/mic:"
TUNE = ":material/tune:"
PLAY = ":material/play_arrow:"
REFRESH = ":material/refresh:"
CLEAR = ":material/cleaning_services:"

# Status
SUCCESS = ":material/check_circle:"
ERROR = ":material/error:"
WARNING = ":material/warning:"
INFO = ":material/info:"

# Analysis types
RDF = ":material/bubble_chart:"
MSD = ":material/timeline:"
THERMO = ":material/thermostat:"


def label(icon: str, text: str) -> str:
    """Build a markdown label like ``:material/...: Title``."""
    return f"{icon} {text}"


def status(success: bool, text: str) -> str:
    """Prefix a line with a success or error icon."""
    return f"{SUCCESS if success else ERROR} {text}"


def warn(text: str) -> str:
    return f"{WARNING} {text}"
