"""Hands-free voice agent — talk like a PI, no buttons, no transcript wall."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gui import icons
from gui.jarvis_ui import AGENT_LABEL, jarvis_css, render_jarvis_hud
from gui.voice_stt import render_voice_input
from materials_ai_agent.voice.tts import synthesize_to_b64


def init_jarvis_state() -> None:
    """Initialize voice-agent session state keys."""
    defaults = {
        "jarvis_mode": True,
        "jarvis_state": "idle",
        "jarvis_status": "Say “Hey Material Sim” to begin",
        "jarvis_transcript": [],
        "auto_speak": True,
        "voice_always_on": True,
        "voice_wake_only": True,
        "voice_show_caption": False,
        "voice_mic_key": 0,
        "jarvis_pending_instruction": None,
        "jarvis_pending_action": None,
        "jarvis_playback_b64": None,
        "jarvis_last_event": None,
        "jarvis_last_plot": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def display_jarvis_mode(agent) -> None:
    """Immersive voice-first interface — always listening, speaks results aloud."""
    st.markdown(jarvis_css(), unsafe_allow_html=True)
    st.markdown(
        """
<style>
  section[data-testid="stSidebar"] { display: none; }
  .block-container { padding-top: 0.5rem; max-width: 960px; }
  .voice-page-tagline {
    text-align: center;
    color: #6b8f9c;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.25rem 0 0.5rem 0;
  }
  .voice-hint {
    text-align: center;
    color: #4a6270;
    font-size: 0.78rem;
    max-width: 520px;
    margin: 0.75rem auto 0 auto;
    line-height: 1.5;
  }
  .voice-stt-shell { display: flex; justify-content: center; margin-top: -0.5rem; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="voice-page-tagline">Hands-free · always listening</div>', unsafe_allow_html=True)

    st.markdown('<div class="voice-hero">', unsafe_allow_html=True)
    render_jarvis_hud(state=_hud_state(), status_text=st.session_state.jarvis_status)
    st.markdown("</div>", unsafe_allow_html=True)

    playback = st.session_state.get("jarvis_playback_b64")
    st.markdown('<div class="voice-stt-shell">', unsafe_allow_html=True)
    voice_payload = render_voice_input(
        always_on=st.session_state.get("voice_always_on", True),
        wake_only=st.session_state.get("voice_wake_only", True),
        show_caption=st.session_state.get("voice_show_caption", False),
        pause_listening=st.session_state.jarvis_state in ("thinking", "speaking"),
        agent_state=st.session_state.jarvis_state,
        status_text=st.session_state.jarvis_status,
        status_live=st.session_state.jarvis_state == "listening",
        audio_b64=playback,
        delay_ms=400,
        key=f"voice_stt_{st.session_state.voice_mic_key}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if playback:
        st.session_state.jarvis_playback_b64 = None

    st.markdown(
        '<p class="voice-hint">'
        "Talk like you would to a student in your group. "
        'Say <strong>“Hey Material Sim, run copper at 300 K for ten thousand steps”</strong> '
        "— I’ll run it and tell you how it went. No buttons. No typing."
        "</p>",
        unsafe_allow_html=True,
    )

    if plot := st.session_state.get("jarvis_last_plot"):
        if Path(plot).exists():
            st.image(plot, caption="Latest analysis", use_container_width=True)

    with st.expander("Optional: show last exchange", expanded=False):
        _render_optional_transcript()

    with st.expander("Voice settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.voice_always_on = st.toggle(
                "Always listening", value=st.session_state.get("voice_always_on", True)
            )
            st.session_state.voice_wake_only = st.toggle(
                "Require “Hey Material Sim”", value=st.session_state.get("voice_wake_only", True)
            )
        with c2:
            st.session_state.auto_speak = st.toggle(
                "Spoken replies", value=st.session_state.get("auto_speak", True)
            )
            st.session_state.voice_show_caption = st.toggle(
                "Live captions (debug)", value=st.session_state.get("voice_show_caption", False)
            )
        if st.button("Reset conversation", icon=icons.CLEAR, use_container_width=True):
            _reset_voice_session(agent_reset=True)
            st.rerun()

    _dispatch_voice_event(agent, voice_payload)


def _dispatch_voice_event(agent, voice_payload) -> None:
    if not isinstance(voice_payload, dict):
        return

    event = voice_payload.get("event")
    if event == st.session_state.get("jarvis_last_event"):
        return
    st.session_state.jarvis_last_event = event

    if event == "playback_finished":
        if st.session_state.get("jarvis_pending_action") == "run_simulation":
            _execute_deferred_simulation(agent)
        else:
            st.session_state.jarvis_state = "listening"
            st.session_state.jarvis_status = "Say “Hey Material Sim”…"
            st.session_state.voice_mic_key += 1
            st.rerun()
        return

    if event == "utterance":
        _handle_utterance(
            agent,
            utterance=(voice_payload.get("utterance") or "").strip(),
            raw=(voice_payload.get("raw") or "").strip(),
        )
        return

    if event == "wake_only":
        from materials_ai_agent.voice.wake import greeting_reply

        st.session_state.jarvis_transcript.append({"role": "assistant", "text": greeting_reply()})
        _speak(greeting_reply())
        return

    if event == "ready" and st.session_state.jarvis_state == "idle":
        st.session_state.jarvis_state = "listening"
        st.session_state.jarvis_status = "Say “Hey Material Sim”…"


def _hud_state() -> str:
    s = st.session_state.jarvis_state
    if s in {"listening", "thinking", "speaking"}:
        return s
    return "idle"


def _render_optional_transcript() -> None:
    if not st.session_state.jarvis_transcript:
        st.caption("Nothing yet — start talking.")
        return
    for entry in st.session_state.jarvis_transcript[-6:]:
        role = "You" if entry["role"] == "user" else AGENT_LABEL
        st.markdown(f"**{role}:** {entry['text']}")


def _handle_utterance(agent, *, utterance: str, raw: str) -> None:
    if not utterance:
        return

    last = st.session_state.jarvis_transcript[-1] if st.session_state.jarvis_transcript else None
    if last and last.get("role") == "user" and last.get("text") == utterance:
        return

    st.session_state.jarvis_transcript.append({"role": "user", "text": utterance})
    st.session_state.jarvis_state = "thinking"
    st.session_state.jarvis_status = "Working on it…"

    from materials_ai_agent.voice.orchestrator import VoiceOrchestrator

    response = VoiceOrchestrator(agent).process(utterance, raw=raw)
    st.session_state.jarvis_transcript.append({"role": "assistant", "text": response.spoken_text})

    if response.defer_action and response.action == "run_simulation":
        st.session_state.jarvis_pending_instruction = response.data.get("instruction", utterance)
        st.session_state.jarvis_pending_action = "run_simulation"
        _speak(response.spoken_text)
        return

    if response.action == "analyze" and response.data:
        _cache_analysis_plots(response.data)

    _speak(response.spoken_text)


def _execute_deferred_simulation(agent) -> None:
    instruction = st.session_state.pop("jarvis_pending_instruction", None)
    st.session_state.jarvis_pending_action = None
    if not instruction:
        st.session_state.jarvis_state = "listening"
        st.session_state.jarvis_status = "Say “Hey Material Sim”…"
        st.session_state.voice_mic_key += 1
        st.rerun()
        return

    st.session_state.jarvis_state = "thinking"
    st.session_state.jarvis_status = "Running molecular dynamics…"

    from materials_ai_agent.voice.orchestrator import VoiceOrchestrator

    with st.spinner("Running simulation…"):
        response = VoiceOrchestrator(agent).execute_simulation(instruction)

    st.session_state.jarvis_transcript.append({"role": "assistant", "text": response.spoken_text})

    if response.data and response.data.get("simulation_directory"):
        st.session_state.last_sim_dir = response.data["simulation_directory"]
        st.session_state.active_sim_dir = response.data["simulation_directory"]
    _cache_analysis_plots(response.data or {})
    _speak(response.spoken_text)


def _speak(text: str) -> None:
    st.session_state.jarvis_state = "speaking"
    st.session_state.jarvis_status = "Speaking…"

    if st.session_state.auto_speak and text:
        st.session_state.jarvis_playback_b64 = synthesize_to_b64(text)
    else:
        st.session_state.jarvis_state = "listening"
        st.session_state.jarvis_status = "Say “Hey Material Sim”…"

    st.session_state.voice_mic_key += 1
    st.rerun()


def _cache_analysis_plots(data: dict) -> None:
    for key in ("rdf", "msd", "thermodynamics"):
        section = data.get(key, {})
        plot = section.get("plot_file")
        if plot and Path(plot).exists():
            st.session_state.jarvis_last_plot = plot
            break


def _reset_voice_session(*, agent_reset: bool = False) -> None:
    st.session_state.jarvis_transcript = []
    st.session_state.jarvis_state = "idle"
    st.session_state.jarvis_status = "Say “Hey Material Sim” to begin"
    st.session_state.jarvis_pending_instruction = None
    st.session_state.jarvis_pending_action = None
    st.session_state.jarvis_playback_b64 = None
    st.session_state.jarvis_last_event = None
    st.session_state.voice_mic_key += 1
    if agent_reset:
        from gui.state import get_agent

        agent = get_agent()
        if agent and hasattr(agent, "reset_conversation"):
            agent.reset_conversation()
