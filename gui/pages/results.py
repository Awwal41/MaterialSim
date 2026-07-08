"""Simulation results browser."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import streamlit as st

from gui import icons
from gui.analysis import perform_msd_analysis, perform_rdf_analysis, perform_thermodynamic_analysis
from gui.utils import parse_sim_dir_name


PLOT_NAMES = {
    "rdf_plot.png": "RDF",
    "msd_analysis.png": "MSD",
    "thermodynamic_analysis.png": "Thermodynamics",
}


def _list_simulations() -> list[str]:
    dirs = [d for d in glob.glob("simulations/*") if os.path.isdir(d)]
    return sorted(dirs, key=os.path.getmtime, reverse=True)


def render_results_page() -> None:
    st.markdown(f"### {icons.NAV_RESULTS} Simulation results")

    sim_dirs = _list_simulations()
    if not sim_dirs:
        st.info("No simulations yet. Run one from **Chat** or **Voice Agent**.")
        return

    for sim_dir in sim_dirs:
        meta = parse_sim_dir_name(sim_dir)
        title = meta["material"]
        temp = meta["temperature"]
        steps = meta["n_steps"]
        subtitle = []
        if temp is not None:
            subtitle.append(f"{temp:g} K")
        if steps is not None:
            subtitle.append(f"{steps:,} steps")
        meta_line = " · ".join(subtitle) if subtitle else sim_dir

        with st.expander(f"{title} — {meta_line}", expanded=(sim_dir == sim_dirs[0])):
            st.caption(str(Path(sim_dir).resolve()))

            plots = [p for p in Path(sim_dir).glob("*.png")]
            if plots:
                cols = st.columns(min(len(plots), 3))
                for i, plot in enumerate(plots):
                    with cols[i % len(cols)]:
                        caption = PLOT_NAMES.get(plot.name, plot.stem)
                        st.image(str(plot), caption=caption, use_container_width=True)
            else:
                st.caption("No plot images yet.")

            files = sorted(f for f in Path(sim_dir).iterdir() if f.is_file())
            if files:
                st.markdown(f"**{icons.FOLDER} Files**")
                fcols = st.columns(3)
                for i, fpath in enumerate(files):
                    with fcols[i % 3]:
                        with open(fpath, "rb") as fh:
                            st.download_button(
                                fpath.name,
                                data=fh.read(),
                                file_name=fpath.name,
                                key=f"dl_{sim_dir}_{fpath.name}",
                                use_container_width=True,
                                icon=icons.DOWNLOAD,
                            )

            acol1, acol2, acol3 = st.columns(3)
            with acol1:
                if st.button("RDF", icon=icons.RDF, key=f"rdf_{sim_dir}", use_container_width=True):
                    _run_analysis(sim_dir, "rdf")
            with acol2:
                if st.button("MSD", icon=icons.MSD, key=f"msd_{sim_dir}", use_container_width=True):
                    _run_analysis(sim_dir, "msd")
            with acol3:
                if st.button("Thermo", icon=icons.THERMO, key=f"thermo_{sim_dir}", use_container_width=True):
                    _run_analysis(sim_dir, "thermo")


def _run_analysis(sim_dir: str, kind: str) -> None:
    meta = parse_sim_dir_name(sim_dir)
    wf = st.session_state.simulation_workflow
    wf["material"] = meta["material"]
    if meta["temperature"] is not None:
        wf["temperature"] = meta["temperature"]
    if meta["n_steps"] is not None:
        wf["n_steps"] = meta["n_steps"]
    st.session_state.last_sim_dir = sim_dir

    with st.spinner(f"Running {kind.upper()} analysis…"):
        if kind == "rdf":
            result = perform_rdf_analysis()
        elif kind == "msd":
            result = perform_msd_analysis()
        else:
            result = perform_thermodynamic_analysis()

    st.markdown(result)
