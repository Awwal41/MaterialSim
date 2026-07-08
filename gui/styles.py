"""Shared Streamlit styling for MaterialSim."""

from __future__ import annotations

import streamlit as st

# Design tokens — scientific dark theme with teal accent
ACCENT = "#2dd4bf"
ACCENT_DIM = "#1a9e8f"
SURFACE = "#1a2332"
SURFACE_ALT = "#243044"
BORDER = "rgba(45, 212, 191, 0.18)"
TEXT_MUTED = "#8b9cb3"

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* Ambient background */
.stApp {{
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(45, 212, 191, 0.09) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 90% 10%, rgba(13, 148, 136, 0.07) 0%, transparent 50%),
        #0f1419 !important;
}}

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 5rem;
    max-width: 1500px;
}}

/* Tabs polish for the workspace stage */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
}}

/* Compact app header */
.ms-header-rule {{
    height: 1px;
    background: {BORDER};
    margin: 0.5rem 0 1.1rem 0;
}}

.ms-pill {{
    display: inline-block;
    margin-top: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    text-align: center;
    float: right;
}}

.ms-pill-ok {{
    color: {ACCENT};
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid {BORDER};
}}

.ms-pill-err {{
    color: #f87171;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.25);
}}

.ms-header-brand {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}

.ms-header-icon-wrap {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 10px;
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid {BORDER};
    font-size: 1.35rem;
    color: {ACCENT};
}}

.ms-status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.45rem;
    vertical-align: middle;
}}

.ms-status-dot--ok {{
    background: {ACCENT};
    box-shadow: 0 0 8px rgba(45, 212, 191, 0.65);
}}

.ms-status-dot--err {{
    background: #ef4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.55);
}}

.ms-header-title {{
    font-family: 'Exo 2', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    background: linear-gradient(135deg, #5eead4 0%, {ACCENT} 50%, #0d9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.01em;
}}

.ms-header-tagline {{
    font-size: 0.88rem;
    color: {TEXT_MUTED};
    margin: 0.2rem 0 0 0;
}}

.ms-badge {{
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid {BORDER};
    color: {ACCENT};
    background: rgba(45, 212, 191, 0.1);
    box-shadow: 0 0 20px rgba(45, 212, 191, 0.12);
}}

/* Welcome / empty state */
.ms-welcome {{
    background: linear-gradient(145deg, {SURFACE} 0%, rgba(20, 28, 40, 0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}}

.ms-welcome h3 {{
    font-family: 'Exo 2', sans-serif;
    color: {ACCENT};
    margin-top: 0;
    font-size: 1.1rem;
}}

.ms-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}}

/* Simulation stepper */
.ms-stepper {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
}}

.ms-stepper-label {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.35rem;
}}

/* Result cards */
.ms-result-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}}

.ms-result-card h4 {{
    color: {ACCENT};
    margin: 0 0 0.35rem 0;
    font-size: 1rem;
}}

.ms-result-meta {{
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    margin: 0;
}}

/* Sidebar status */
.ms-status-ready {{
    background: rgba(45, 212, 191, 0.08);
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
}}

.ms-status-error {{
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
}}

/* Voice agent panel */
.voice-hero {{
    background: linear-gradient(160deg, #0f1419 0%, #132030 45%, #0f1a24 100%);
    border: 1px solid rgba(45, 212, 191, 0.22);
    border-radius: 16px;
    padding: 0.5rem 1rem 0.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 0 40px rgba(45, 212, 191, 0.06);
}}

.voice-controls-row {{
    margin-bottom: 1rem;
}}

.voice-controls-card {{
    background: rgba(26, 35, 50, 0.85);
    border: 1px solid rgba(45, 212, 191, 0.15);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}}

/* Chat messages polish */
[data-testid="stChatMessage"] {{
    background: rgba(26, 35, 50, 0.45);
    border: 1px solid rgba(45, 212, 191, 0.08);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #121a24 0%, #0f1419 100%);
    border-right: 1px solid {BORDER};
}}

/* Buttons */
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(45, 212, 191, 0.15) !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 100%) !important;
    border: none !important;
    color: #042f2e !important;
}}

/* Chat input */
[data-testid="stChatInput"] {{
    border-top: 1px solid {BORDER};
    padding-top: 0.85rem;
    background: transparent;
}}

[data-testid="stChatInput"] textarea {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    background: {SURFACE} !important;
    min-height: 3.5rem !important;
}}

[data-testid="stTextArea"] textarea {{
    min-height: 10rem !important;
    line-height: 1.5 !important;
}}

/* Expander cards on results page */
.streamlit-expanderHeader {{
    border-radius: 10px !important;
}}
"""


@st.cache_data(show_spinner=False)
def get_custom_css() -> str:
    return CUSTOM_CSS


def inject_styles() -> None:
    st.markdown(f"<style>{get_custom_css()}</style>", unsafe_allow_html=True)
