"""Settings page."""

from __future__ import annotations

import os
import sys

import streamlit as st

from gui import icons
from gui.state import get_agent, initialize_agent


def render_settings_page() -> None:
    st.markdown(f"### {icons.NAV_SETTINGS} Settings")

    st.markdown(f"#### {icons.TUNE} API configuration")
    col1, col2 = st.columns(2)

    with col1:
        openai_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Required for LLM chat and routing",
        )

    with col2:
        mp_key = st.text_input(
            "Materials Project API key",
            value=os.getenv("MP_API_KEY", ""),
            type="password",
            help="Optional — structure lookup from Materials Project",
        )

    if st.button("Save API keys", type="primary", use_container_width=True, icon=icons.SUCCESS):
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"MP_API_KEY={mp_key}\n")
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["MP_API_KEY"] = mp_key
        st.cache_resource.clear()
        st.session_state.agent_initialized = False
        st.session_state.pop("agent", None)
        st.success("Keys saved. Reinitializing agent…")
        st.rerun()

    st.markdown("---")
    st.markdown(f"#### {icons.SCIENCE} Agent diagnostics")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test agent", use_container_width=True, icon=icons.PLAY):
            if initialize_agent() and get_agent():
                try:
                    reply = get_agent().chat("Reply with exactly: Agent is working.")
                    st.success(f"OK — {reply[:120]}")
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("Agent not initialized")

    with c2:
        if st.button("Reinitialize agent", use_container_width=True, icon=icons.REFRESH):
            st.cache_resource.clear()
            st.session_state.agent_initialized = False
            st.session_state.pop("agent", None)
            st.rerun()

    st.markdown("---")
    st.markdown(f"#### {icons.INFO} System")

    info1, info2 = st.columns(2)
    with info1:
        st.caption("Python")
        st.code(sys.version.split()[0])
    with info2:
        st.caption("Packages")
        try:
            import streamlit as _st
            st.success(f"Streamlit {_st.__version__}")
        except ImportError:
            st.error("Streamlit missing")
        try:
            from materials_ai_agent import MaterialsAgent  # noqa: F401
            st.success("materials_ai_agent")
        except ImportError:
            st.error("materials_ai_agent missing")
