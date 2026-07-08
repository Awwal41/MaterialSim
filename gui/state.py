"""Session state and agent initialization."""

from __future__ import annotations

import streamlit as st

from gui.jarvis_panel import init_jarvis_state


@st.cache_resource(show_spinner="Initializing AI agent...")
def _create_agent():
    from materials_ai_agent import MaterialsAgent
    return MaterialsAgent()


def initialize_session_state() -> None:
    init_jarvis_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "simulation_params" not in st.session_state:
        st.session_state.simulation_params = {}
    if "last_sim_dir" not in st.session_state:
        st.session_state.last_sim_dir = None
    if "simulation_workflow" not in st.session_state:
        from materials_ai_agent.core.config import Config

        config = Config.from_env()
        st.session_state.simulation_workflow = {
            "step": 0,
            "material": "",
            "temperature": config.default_temperature,
            "ensemble": config.default_ensemble,
            "thermostat": config.default_thermostat,
            "timestep": config.default_timestep,
            "n_steps": config.default_n_steps,
            "force_field": config.default_force_field,
            "structure_source": config.default_structure_source,
            "structure_file": None,
            "mp_material_id": None,
            "explanations_shown": set(),
            "user_confirmations": {},
        }


def initialize_agent() -> bool:
    if st.session_state.agent_initialized:
        return True
    try:
        st.session_state.agent = _create_agent()
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.session_state.agent_initialized = False
        st.session_state._agent_init_error = str(e)
        return False


def get_agent():
    return st.session_state.get("agent")
