"""Chat page — native Streamlit chat UI."""

from __future__ import annotations

import streamlit as st

from gui import icons
from gui.simulation_workflow import (
    get_ai_response,
    handle_simulation_conversation,
    is_simulation_request,
    start_interactive_simulation_workflow,
)

WORKFLOW_STEPS = {
    1: "Material",
    2: "Temperature",
    3: "Ensemble",
    4: "Thermostat",
    5: "Timestep & steps",
    6: "Structure",
    7: "Force field",
    8: "Confirm",
    9: "Post-run",
}

EXAMPLE_PROMPTS = [
    "Simulate silicon at 300K",
    "Simulate copper at 500K with NVT",
    "What is a radial distribution function?",
]


def _render_workflow_stepper() -> None:
    step = st.session_state.simulation_workflow.get("step", 0)
    if step <= 0:
        return

    label = WORKFLOW_STEPS.get(step, f"Step {step}")
    progress = min(step / 9.0, 1.0)
    st.markdown(
        f'<div class="ms-stepper">'
        f'<div class="ms-stepper-label">Simulation setup · {label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(progress, text=f"Step {step} of 9 — {label}")


def _render_welcome_body() -> None:
    chips = "".join(f"<span class='ms-badge'>{p}</span>" for p in EXAMPLE_PROMPTS)
    st.markdown(
        f"""
<div class="ms-welcome">
  <p style="color:#8b9cb3; margin:0;">
    Describe a simulation or analysis task in plain language. I'll guide you through
    parameters, run molecular dynamics, and help interpret results.
  </p>
  <div class="ms-chip-row">{chips}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_welcome() -> None:
    st.markdown(f"#### {icons.SCIENCE} Welcome")
    _render_welcome_body()


def _render_messages() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        avatar = icons.AVATAR_USER if role == "user" else icons.AVATAR_AGENT
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            for plot_path in message.get("plots", []):
                st.image(plot_path)


def _process_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.simulation_workflow["step"] > 0:
        handle_simulation_conversation(prompt)
    elif is_simulation_request(prompt):
        start_interactive_simulation_workflow(prompt)
    else:
        get_ai_response(prompt)


@st.fragment
def _chat_fragment() -> None:
    _render_workflow_stepper()

    if not st.session_state.messages:
        st.markdown(f"#### {icons.SCIENCE} Welcome")
        _render_welcome_body()

    _render_messages()

    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        _process_prompt(pending)
        return

    if prompt := st.chat_input("Describe your materials science task…"):
        _process_prompt(prompt)


def render_chat_page() -> None:
    from gui.state import get_agent, initialize_agent

    if not initialize_agent() or not get_agent():
        st.error("Agent could not be initialized. Open **Settings** to configure API keys.")
        return

    _chat_fragment()
