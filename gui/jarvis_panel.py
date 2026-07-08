"""Voice agent panel — ChatGPT-style conversational flow."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from gui import icons
from gui.jarvis_ui import AGENT_LABEL, jarvis_css, render_jarvis_hud
from gui.voice_stt import render_voice_input


def init_jarvis_state() -> None:
    """Initialize voice-agent session state keys."""
    defaults = {
        "jarvis_mode": False,
        "jarvis_state": "idle",
        "jarvis_status": "Tap the microphone to begin",
        "jarvis_transcript": [],
        "voice_enabled": True,
        "auto_speak": True,
        "conversation_mode": True,
        "voice_auto_listen": True,
        "voice_turn": 0,
        "voice_mic_key": 0,
        "last_processed_audio": "",
        "pending_voice_reply": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def display_jarvis_mode(agent) -> None:
    """Render the Material Sim Agent voice-first interface."""
    st.markdown(jarvis_css(), unsafe_allow_html=True)

    _render_hud()
    _render_controls()
    _render_transcript()

    utterance = None

    # Primary: browser speech recognition (continuous ChatGPT-style flow).
    if st.session_state.voice_enabled:
        voice_payload = render_voice_input(
            auto_listen=st.session_state.get("voice_auto_listen", False),
            delay_ms=700 if st.session_state.get("pending_voice_reply") else 400,
            hint=(
                "Speak naturally. I'll listen, respond aloud, then listen again."
                if st.session_state.conversation_mode
                else "Tap the mic, speak, then pause."
            ),
            key=f"voice_stt_{st.session_state.voice_mic_key}",
        )
        st.session_state.voice_auto_listen = False
        st.session_state.pending_voice_reply = False

        if isinstance(voice_payload, dict) and voice_payload.get("event") == "utterance":
            utterance = (voice_payload.get("utterance") or "").strip()

        # Fallback: Streamlit native audio_input.
        if utterance is None:
            utterance = _try_streamlit_mic()

    # Text fallback always available.
    text_input = st.chat_input(f"Or type to {AGENT_LABEL}…")
    if text_input:
        utterance = text_input

    if utterance:
        _process_utterance(agent, utterance)


def _render_hud() -> None:
    st.markdown('<div class="voice-hero">', unsafe_allow_html=True)
    render_jarvis_hud(
        state=st.session_state.jarvis_state,
        status_text=st.session_state.jarvis_status,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_controls() -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.conversation_mode = st.toggle(
            f"{icons.NAV_VOICE} Conversation mode",
            value=st.session_state.get("conversation_mode", True),
            help="After each reply, automatically listen for your next message.",
        )
    with c2:
        st.session_state.voice_enabled = st.toggle(
            f"{icons.MIC} Voice input",
            value=st.session_state.get("voice_enabled", True),
        )
    with c3:
        st.session_state.auto_speak = st.toggle(
            "Spoken replies",
            value=st.session_state.get("auto_speak", True),
        )
    with c4:
        if st.button("New conversation", icon=icons.CLEAR, use_container_width=True):
            st.session_state.jarvis_transcript = []
            st.session_state.jarvis_state = "idle"
            st.session_state.jarvis_status = "Tap the microphone to begin"
            st.session_state.voice_mic_key += 1
            st.rerun()


def _render_transcript() -> None:
    if not st.session_state.jarvis_transcript:
        st.markdown(
            f"##### {icons.NAV_VOICE} Voice conversation\n\n"
            "Start speaking to run simulations, analyze results, or ask questions — "
            "just like a voice chat with an AI assistant."
        )
        return

    transcript = st.session_state.jarvis_transcript[-20:]
    for i, entry in enumerate(transcript):
        role = entry["role"]
        is_latest = i == len(transcript) - 1
        with st.chat_message(
            "user" if role == "user" else "assistant",
            avatar=icons.AVATAR_USER if role == "user" else icons.AVATAR_AGENT,
        ):
            st.markdown(entry["text"])
            if entry.get("audio_path") and Path(entry["audio_path"]).exists():
                st.audio(
                    entry["audio_path"],
                    autoplay=is_latest and role == "assistant" and st.session_state.auto_speak,
                )


def _try_streamlit_mic() -> str | None:
    """Fallback mic using st.audio_input when browser STT is unavailable."""
    try:
        audio = st.audio_input(
            f"{icons.MIC} Fallback microphone",
            key=f"jarvis_mic_{st.session_state.voice_mic_key}",
        )
        if audio is None:
            return None
        digest = hashlib.md5(audio.getvalue()).hexdigest()
        if digest == st.session_state.last_processed_audio:
            return None
        st.session_state.last_processed_audio = digest
        st.session_state.jarvis_state = "listening"
        st.session_state.jarvis_status = "Listening…"

        from materials_ai_agent.voice.stt import transcribe_audio

        return transcribe_audio(audio.getvalue())
    except Exception:
        st.caption("Native microphone fallback unavailable.")
        return None


def _process_utterance(agent, utterance: str) -> None:
    """Run an utterance through the voice orchestrator and speak the reply."""
    # Avoid double-processing on Streamlit reruns.
    last = st.session_state.jarvis_transcript[-1] if st.session_state.jarvis_transcript else None
    if last and last.get("role") == "user" and last.get("text") == utterance:
        return

    st.session_state.jarvis_transcript.append({"role": "user", "text": utterance})
    st.session_state.jarvis_state = "thinking"
    st.session_state.jarvis_status = "Thinking…"

    response = agent.process_command(utterance)

    assistant_entry = {
        "role": "assistant",
        "text": response.display_text,
        "audio_path": None,
    }

    st.session_state.jarvis_state = "speaking"
    st.session_state.jarvis_status = "Speaking…"

    if st.session_state.auto_speak and response.spoken_text:
        from materials_ai_agent.voice.tts import synthesize_speech

        audio_path = synthesize_speech(response.spoken_text)
        if audio_path and audio_path.exists():
            assistant_entry["audio_path"] = str(audio_path)

    st.session_state.jarvis_transcript.append(assistant_entry)

    if response.action == "analyze" and response.data:
        _show_analysis_plots(response.data)

    # Chain to next turn like ChatGPT voice.
    if st.session_state.conversation_mode and st.session_state.voice_enabled:
        st.session_state.jarvis_state = "idle"
        st.session_state.jarvis_status = "Listening for your reply…"
        st.session_state.voice_auto_listen = True
        st.session_state.pending_voice_reply = True
    else:
        st.session_state.jarvis_state = "idle"
        st.session_state.jarvis_status = "Tap the microphone to speak"

    st.session_state.voice_mic_key += 1
    st.rerun()


def _show_analysis_plots(data: dict) -> None:
    for key in ("rdf", "msd", "thermodynamics"):
        section = data.get(key, {})
        plot = section.get("plot_file")
        if plot and Path(plot).exists():
            st.image(plot, caption=key.upper())
