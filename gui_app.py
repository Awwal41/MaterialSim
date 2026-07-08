#!/usr/bin/env python3
"""MaterialSim AI Agent — Streamlit GUI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from gui import icons  # noqa: E402

st.set_page_config(
    page_title="Material Sim Agent",
    page_icon=icons.APP,
    layout="wide",
    initial_sidebar_state="expanded",
)

from gui.components.header import render_header  # noqa: E402
from gui.components.sidebar import render_sidebar  # noqa: E402
from gui.pages.settings import render_settings_page  # noqa: E402
from gui.pages.workspace import render_workspace_page  # noqa: E402
from gui.state import initialize_agent, initialize_session_state  # noqa: E402
from gui.styles import inject_styles  # noqa: E402

inject_styles()
initialize_session_state()


def page_workspace() -> None:
    render_header()
    render_workspace_page()


def page_jarvis() -> None:
    if not initialize_agent():
        st.error("Agent not initialized. Configure API keys in Settings.")
        return
    from gui.jarvis_panel import display_jarvis_mode
    from gui.state import get_agent

    display_jarvis_mode(get_agent())


def page_settings() -> None:
    render_header()
    render_settings_page()


def main() -> None:
    render_sidebar()

    nav = st.navigation(
        {
            "Workspace": [
                st.Page(page_workspace, title="Workspace", icon=icons.NAV_RESULTS, default=True),
                st.Page(page_jarvis, title="Voice Agent", icon=icons.NAV_VOICE),
            ],
            "System": [
                st.Page(page_settings, title="Settings", icon=icons.NAV_SETTINGS),
            ],
        }
    )
    nav.run()


if __name__ == "__main__":
    main()

# Backward compatibility for scripts importing analysis helpers from gui_app
from gui.analysis import (  # noqa: E402, F401
    perform_msd_analysis,
    perform_rdf_analysis,
    perform_thermodynamic_analysis,
)
