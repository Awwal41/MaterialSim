"""Compact app header."""

from __future__ import annotations

import streamlit as st

from gui import icons
from gui.state import get_agent, initialize_agent


def render_header(*, compact: bool = True, show_tagline: bool = True) -> None:
    agent_ok = initialize_agent() and get_agent() is not None
    badge = "Ready" if agent_ok else "Offline"
    badge_style = "ms-pill-ok" if agent_ok else "ms-pill-err"

    title_col, badge_col = st.columns([5, 1])
    with title_col:
        st.markdown(f"### {icons.APP} Material Sim Agent")
        if show_tagline:
            st.caption("Computational materials science with AI")
    with badge_col:
        st.markdown(
            f'<p class="ms-pill {badge_style}">{badge}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ms-header-rule"></div>', unsafe_allow_html=True)
