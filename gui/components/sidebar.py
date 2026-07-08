"""Sidebar with agent status and quick actions."""

from __future__ import annotations

import streamlit as st

from gui import icons
from gui.state import get_agent, initialize_agent


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {icons.APP} Material Sim Agent")
        st.caption("Molecular dynamics · Analysis · Research")

        st.markdown("---")
        st.markdown(f"**{icons.SCIENCE} Agent**")

        if initialize_agent() and get_agent():
            agent = get_agent()
            st.markdown(
                '<div class="ms-status-ready">'
                '<span class="ms-status-dot ms-status-dot--ok"></span>'
                "Ready for simulations and chat.</div>",
                unsafe_allow_html=True,
            )
            tool_count = len(agent.tools) if hasattr(agent, "tools") else 0
            st.caption(f"{tool_count} tools loaded")
        else:
            st.markdown(
                '<div class="ms-status-error">'
                '<span class="ms-status-dot ms-status-dot--err"></span>'
                "Not initialized. Check API keys in Settings.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f"**{icons.TUNE} Quick actions**")

        if st.button(
            "Clear chat",
            icon=icons.CLEAR,
            use_container_width=True,
            key="sidebar_clear_chat",
        ):
            st.session_state.messages = []
            st.session_state.simulation_workflow["step"] = 0
            st.rerun()

        if st.button(
            "Reinitialize agent",
            icon=icons.REFRESH,
            use_container_width=True,
            key="sidebar_reinit",
        ):
            st.cache_resource.clear()
            st.session_state.agent_initialized = False
            st.session_state.pop("agent", None)
            st.rerun()

        st.markdown("---")
        st.markdown(f"**{icons.CHAT} Try asking**")
        examples = [
            ("Simulate copper at 300K", icons.PLAY),
            ("Analyze RDF for my last run", icons.RDF),
            ("What is NVT ensemble?", icons.INFO),
        ]
        for i, (example, icon) in enumerate(examples):
            if st.button(
                example,
                icon=icon,
                use_container_width=True,
                key=f"example_{i}",
            ):
                st.session_state.messages.append({"role": "user", "content": example})
                st.session_state._pending_prompt = example
                st.rerun()
