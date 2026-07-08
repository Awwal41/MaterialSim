"""Unified simulation workspace: configure -> run -> watch -> analyze.

This replaces the old 9-step keyword wizard. The LLM spec extractor pre-fills an
editable configuration panel from one plain-language sentence; the user tweaks
anything with real controls and runs. Results (3D structure, live-style
telemetry, and analysis) live on the same screen, driven by whichever run is
active.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import streamlit as st

from gui import icons, viewer
from gui.analysis import (
    perform_msd_analysis,
    perform_rdf_analysis,
    perform_thermodynamic_analysis,
)
from gui.simulation_workflow import (
    fetch_mp_structure_for_workflow,
    save_uploaded_structure,
)
from gui.utils import parse_sim_dir_name


# --------------------------------------------------------------------------- #
# Options (queried from the backend so the GUI matches installed capabilities)
# --------------------------------------------------------------------------- #
def _config():
    from materials_ai_agent.core.config import Config

    return Config.from_env()


def _capabilities() -> dict:
    caps = {
        "engines_registered": ["ase"],
        "engines_installed": [],
        "potentials_registered": [],
        "potentials_runnable": [],
        "protocols": ["equilibrium"],
    }
    try:
        from materials_ai_agent import bootstrap
        from materials_ai_agent.engines.registry import available_engines, list_engines
        from materials_ai_agent.potentials.registry import list_potentials
        from materials_ai_agent.protocols.registry import list_protocols

        bootstrap.ensure()
        caps["engines_registered"] = list_engines()
        caps["engines_installed"] = available_engines()
        caps["potentials_registered"] = list_potentials()
        caps["potentials_runnable"] = _config().runnable_force_fields()
        caps["protocols"] = list_protocols() or ["equilibrium"]
    except Exception:
        pass
    return caps


# --------------------------------------------------------------------------- #
# Session defaults for the configuration widgets
# --------------------------------------------------------------------------- #
def _ensure_defaults() -> None:
    ss = st.session_state
    if ss.get("_ws_defaults_ready"):
        return
    cfg = _config()
    defaults = {
        "ws_material": "",
        "ws_structure_source": cfg.default_structure_source,
        "ws_structure_file": None,
        "ws_mp_id": "",
        "ws_engine": "auto",
        "ws_force_field": cfg.default_force_field or "auto",
        "ws_protocol": "equilibrium",
        "ws_ensemble": cfg.default_ensemble,
        "ws_temperature": float(cfg.default_temperature),
        "ws_pressure": float(cfg.default_pressure),
        "ws_thermostat": cfg.default_thermostat or "auto",
        "ws_timestep": float(cfg.default_timestep),
        "ws_n_steps": int(cfg.default_n_steps),
        "ws_output_frequency": 100,
        "ws_target_atoms": 256,
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)
    ss._ws_defaults_ready = True


def _apply_spec(spec) -> None:
    """Fill the form from an extracted SimulationSpec (LLM/deterministic)."""
    ss = st.session_state
    cfg = _config()
    material = spec.system.material or spec.system.smiles or spec.system.mp_material_id or ""
    if str(material).lower() in {"unresolved", "custom", "user", "uploaded", "none"}:
        material = ""
    ss.ws_material = material
    ss.ws_temperature = max(cfg.min_temperature, min(spec.ensemble.temperature, cfg.max_temperature))
    ss.ws_pressure = float(spec.ensemble.pressure)
    ss.ws_ensemble = (spec.ensemble.name or "NVT").upper()
    ss.ws_thermostat = (spec.ensemble.thermostat or "auto").lower()
    ss.ws_force_field = (spec.potential.kind or "auto").lower()
    ss.ws_engine = (spec.engine or "auto").lower()
    ss.ws_protocol = spec.protocol.name or "equilibrium"
    ss.ws_n_steps = int(spec.run.n_steps)
    ss.ws_timestep = float(spec.run.timestep)
    ss.ws_output_frequency = int(spec.run.output_frequency)
    if spec.system.mp_material_id:
        ss.ws_mp_id = spec.system.mp_material_id


def _parse_nl(text: str) -> str:
    """Extract a spec from natural language and apply it to the form."""
    if not text.strip():
        return "Type a description first, e.g. 'thermal conductivity of silicon at 400 K via NEMD'."
    try:
        from materials_ai_agent.spec.extractor import extract_spec

        spec = extract_spec(text, _config())
        _apply_spec(spec)
        note = "" if st.session_state.ws_material else " I couldn't resolve a material — please set it below."
        return f"Parsed: {spec.summary()}.{note}"
    except Exception as exc:  # noqa: BLE001
        return f"Could not parse that: {exc}. Set parameters manually below."


# --------------------------------------------------------------------------- #
# Configuration rail
# --------------------------------------------------------------------------- #
def _selectbox(label, key, options, *, help=None, fmt=None):
    """A selectbox whose current session value is preserved even if off-list."""
    opts = list(options)
    current = st.session_state.get(key)
    if current is not None and current not in opts:
        opts = [current] + opts
    idx = opts.index(current) if current in opts else 0
    return st.selectbox(label, opts, index=idx, key=key, help=help, format_func=fmt or str)


def _render_config_rail(caps: dict) -> None:
    cfg = _config()
    # Sidebar quick-start requested a parse before any widgets are built.
    if st.session_state.pop("_ws_do_parse", False):
        st.session_state._ws_parse_note = _parse_nl(st.session_state.get("ws_nl", ""))

    st.markdown(f"**{icons.SCIENCE} Describe your simulation**")
    st.text_area(
        "Natural language",
        key="ws_nl",
        height=160,
        label_visibility="collapsed",
        placeholder="e.g. 'Simulate copper at 500 K in NPT for 50k steps' or "
        "'shock silicon with MSST' or 'thermal conductivity of Ar via NEMD'",
    )
    if st.button("Parse into form", icon=icons.TUNE, use_container_width=True, key="ws_parse"):
        st.session_state._ws_parse_note = _parse_nl(st.session_state.get("ws_nl", ""))
        st.rerun()
    if note := st.session_state.get("_ws_parse_note"):
        (st.success if "Parsed:" in note else st.info)(note)

    st.markdown("---")

    # System ---------------------------------------------------------------
    st.markdown(f"**{icons.FOLDER} System**")
    st.text_input("Material / formula / SMILES", key="ws_material", placeholder="e.g. Cu, SiO2, CCO")
    _selectbox("Structure source", "ws_structure_source", cfg.available_structure_sources)

    src = st.session_state.ws_structure_source
    if src in {"upload", "file"}:
        up = st.file_uploader(
            "Structure file", type=["xyz", "cif", "poscar", "vasp", "pdb"], key="ws_upload"
        )
        if up is not None:
            st.session_state.ws_structure_file = save_uploaded_structure(up)
            st.caption(f"Loaded {up.name}")
    elif src in {"material_project", "materials_project", "mp"}:
        st.text_input("Materials Project id (optional)", key="ws_mp_id", placeholder="mp-1234")

    # Method ---------------------------------------------------------------
    st.markdown(f"**{icons.TUNE} Method**")
    installed = set(caps["engines_installed"])
    engine_opts = ["auto"] + [e for e in caps["engines_registered"]]
    _selectbox(
        "Engine", "ws_engine", engine_opts,
        fmt=lambda e: e if e == "auto" else (f"{e} · installed" if e in installed else f"{e} · not installed"),
    )
    runnable = set(caps["potentials_runnable"])
    ff_opts = cfg.available_force_fields
    _selectbox(
        "Force field / potential", "ws_force_field", ff_opts,
        fmt=lambda k: k if k == "auto" else (f"{k} · ready" if k in runnable else f"{k} · needs install"),
    )
    _selectbox("Protocol", "ws_protocol", caps["protocols"])

    # Conditions -----------------------------------------------------------
    st.markdown(f"**{icons.THERMO} Conditions**")
    c1, c2 = st.columns(2)
    with c1:
        _selectbox("Ensemble", "ws_ensemble", cfg.available_ensembles)
    with c2:
        _selectbox(
            "Thermostat", "ws_thermostat",
            ["auto"] + [t.lower() for t in cfg.available_thermostats],
        )
    c3, c4 = st.columns(2)
    with c3:
        st.number_input(
            "Temperature (K)", key="ws_temperature",
            min_value=float(cfg.min_temperature), max_value=float(cfg.max_temperature), step=25.0,
        )
    with c4:
        st.number_input(
            "Pressure (atm)", key="ws_pressure", min_value=0.0, step=1.0,
            disabled=st.session_state.ws_ensemble != "NPT",
        )

    # Sampling -------------------------------------------------------------
    st.markdown(f"**{icons.MSD} Sampling**")
    c5, c6 = st.columns(2)
    with c5:
        st.number_input(
            "Timestep (ps)", key="ws_timestep",
            min_value=float(cfg.min_timestep), max_value=float(cfg.max_timestep),
            step=0.0005, format="%.4f",
        )
    with c6:
        st.number_input(
            "Steps", key="ws_n_steps",
            min_value=int(cfg.min_n_steps), max_value=int(cfg.max_n_steps), step=1000,
        )
    c7, c8 = st.columns(2)
    with c7:
        st.number_input("Output every", key="ws_output_frequency", min_value=1, step=50)
    with c8:
        st.number_input("Target atoms", key="ws_target_atoms", min_value=1, step=64)

    st.markdown("---")
    dt = st.session_state.ws_timestep * st.session_state.ws_n_steps
    st.caption(f"Total simulated time ≈ {dt:.2f} ps · {int(st.session_state.ws_n_steps):,} steps")
    if st.button(
        "Run simulation", type="primary", icon=icons.PLAY,
        use_container_width=True, key="ws_run",
        disabled=st.session_state.get("ws_busy", False),
    ):
        _run_simulation()

    with st.expander("What can this machine run?"):
        eng = ", ".join(caps["engines_installed"]) or "none installed"
        st.markdown(f"- **Engines**: {eng}  \n  _(registered: {', '.join(caps['engines_registered'])})_")
        st.markdown(
            f"- **Potentials ready**: {', '.join(runnable) or 'none'}  \n"
            f"  _(recognized: {', '.join(caps['potentials_registered']) or '—'})_"
        )
        st.markdown(f"- **Protocols**: {', '.join(caps['protocols'])}")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _run_simulation() -> None:
    ss = st.session_state
    if not ss.ws_material.strip():
        ss._ws_parse_note = "Please provide a material/formula/SMILES before running."
        st.rerun()
        return

    from materials_ai_agent.simple_simulation import run_simple_simulation
    from materials_ai_agent.structure_builder import normalize_structure_source

    run_kwargs = {
        "material": ss.ws_material.strip(),
        "temperature": float(ss.ws_temperature),
        "n_steps": int(ss.ws_n_steps),
        "timestep": float(ss.ws_timestep),
        "force_field": ss.ws_force_field,
        "ensemble": ss.ws_ensemble,
        "thermostat": None if ss.ws_thermostat == "auto" else ss.ws_thermostat,
        "engine": None if ss.ws_engine == "auto" else ss.ws_engine,
        "protocol": ss.ws_protocol,
        "pressure": float(ss.ws_pressure) if ss.ws_ensemble == "NPT" else None,
        "output_frequency": int(ss.ws_output_frequency),
        "target_atoms": int(ss.ws_target_atoms),
        "structure_source": normalize_structure_source(ss.ws_structure_source),
    }
    if ss.get("ws_structure_file"):
        run_kwargs["structure_file"] = ss.ws_structure_file
    if ss.get("ws_mp_id"):
        run_kwargs["mp_material_id"] = ss.ws_mp_id.strip()

    ss.ws_busy = True
    try:
        with st.spinner(f"Running {run_kwargs['protocol']} MD for {run_kwargs['material']}…"):
            result = run_simple_simulation(**run_kwargs)
    except Exception as exc:  # noqa: BLE001
        result = {"success": False, "error": str(exc)}
    finally:
        ss.ws_busy = False

    ss.ws_last_result = result
    if result.get("success"):
        sim_dir = result.get("simulation_directory")
        ss.last_sim_dir = sim_dir
        ss.active_sim_dir = sim_dir
    st.rerun()


# --------------------------------------------------------------------------- #
# Stage (right side): overview, 3D, telemetry, analysis, assistant
# --------------------------------------------------------------------------- #
def _list_runs() -> list[str]:
    dirs = [d for d in glob.glob("simulations/*") if os.path.isdir(d)]
    return sorted(dirs, key=os.path.getmtime, reverse=True)


def _active_dir() -> str | None:
    ss = st.session_state
    return ss.get("active_sim_dir") or ss.get("last_sim_dir")


def _render_result_banner() -> None:
    result = st.session_state.get("ws_last_result")
    if not result:
        return
    if result.get("success"):
        st.success(
            f"{icons.SUCCESS} {result.get('message', 'Simulation complete.')} "
            f"· {result.get('n_frames', '?')} frames · `{result.get('simulation_directory', '')}`"
        )
    elif result.get("needs_clarification"):
        st.warning(f"{icons.WARNING} Need more info: {result.get('error')}")
    else:
        st.error(f"{icons.ERROR} {result.get('error', 'Simulation failed.')}")


def _render_overview() -> None:
    ss = st.session_state
    runs = _list_runs()
    if runs:
        current = _active_dir()
        idx = runs.index(current) if current in runs else 0
        chosen = st.selectbox(
            "Active run", runs, index=idx,
            format_func=lambda d: Path(d).name, key="ws_active_select",
        )
        ss.active_sim_dir = chosen
    else:
        st.info("No runs yet — configure on the left and hit **Run simulation**.")

    st.markdown("**Current configuration**")
    m = st.columns(4)
    m[0].metric("Material", ss.ws_material or "—")
    m[1].metric("Ensemble", ss.ws_ensemble)
    m[2].metric("Temperature", f"{ss.ws_temperature:g} K")
    m[3].metric("Steps", f"{int(ss.ws_n_steps):,}")
    m2 = st.columns(4)
    m2[0].metric("Engine", ss.ws_engine)
    m2[1].metric("Potential", ss.ws_force_field)
    m2[2].metric("Protocol", ss.ws_protocol)
    m2[3].metric("Timestep", f"{ss.ws_timestep:g} ps")

    active = _active_dir()
    if active and Path(active).exists():
        files = sorted(p for p in Path(active).iterdir() if p.is_file())
        if files:
            st.markdown(f"**{icons.FOLDER} Files**")
            fcols = st.columns(3)
            for i, fpath in enumerate(files):
                with fcols[i % 3]:
                    with open(fpath, "rb") as fh:
                        st.download_button(
                            fpath.name, data=fh.read(), file_name=fpath.name,
                            key=f"ws_dl_{fpath.name}", use_container_width=True,
                            icon=icons.DOWNLOAD,
                        )


def _render_3d() -> None:
    active = _active_dir()
    if not active:
        st.info("Run or select a simulation to view its 3D structure/trajectory.")
        return
    vfile = viewer.find_viewable_file(active)
    if not vfile:
        st.caption("No .xyz structure/trajectory found in this run.")
        return
    animate = st.toggle("Animate trajectory", value=True, key="ws_animate")
    viewer.render_structure(str(vfile), animate=animate, height=480, key="ws_view3d")


def _render_telemetry() -> None:
    active = _active_dir()
    if not active:
        st.info("Run or select a simulation to see thermodynamic telemetry.")
        return
    viewer.render_telemetry(active)


def _render_analysis() -> None:
    active = _active_dir()
    if not active:
        st.info("Run or select a simulation to analyze.")
        return

    meta = parse_sim_dir_name(active)
    wf = st.session_state.simulation_workflow
    wf["material"] = meta["material"]
    if meta["temperature"] is not None:
        wf["temperature"] = meta["temperature"]
    if meta["n_steps"] is not None:
        wf["n_steps"] = meta["n_steps"]
    st.session_state.last_sim_dir = active

    a1, a2, a3 = st.columns(3)
    run = None
    with a1:
        if st.button("RDF", icon=icons.RDF, use_container_width=True, key="ws_rdf"):
            run = perform_rdf_analysis
    with a2:
        if st.button("MSD", icon=icons.MSD, use_container_width=True, key="ws_msd"):
            run = perform_msd_analysis
    with a3:
        if st.button("Thermodynamics", icon=icons.THERMO, use_container_width=True, key="ws_thermo"):
            run = perform_thermodynamic_analysis

    if run is not None:
        with st.spinner("Analyzing…"):
            st.session_state._ws_analysis = run()
    if out := st.session_state.get("_ws_analysis"):
        st.markdown(out)


def _render_assistant() -> None:
    from gui.state import get_agent, initialize_agent

    if not (initialize_agent() and get_agent()):
        st.info("Set an OpenAI API key in **Settings** to chat with the assistant.")
        return

    for msg in st.session_state.messages:
        avatar = icons.AVATAR_USER if msg["role"] == "user" else icons.AVATAR_AGENT
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about MD, methods, or your results…", key="ws_chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=icons.AVATAR_USER):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=icons.AVATAR_AGENT):
            with st.spinner("Thinking…"):
                reply = get_agent().chat(prompt)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def _render_stage() -> None:
    _render_result_banner()
    tabs = st.tabs(["Overview", "3D structure", "Telemetry", "Analysis", "Assistant"])
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_3d()
    with tabs[2]:
        _render_telemetry()
    with tabs[3]:
        _render_analysis()
    with tabs[4]:
        _render_assistant()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_workspace_page() -> None:
    _ensure_defaults()
    caps = _capabilities()

    left, right = st.columns([1.05, 2], gap="large")
    with left:
        _render_config_rail(caps)
    with right:
        _render_stage()
