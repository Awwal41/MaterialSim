"""GUI utility helpers."""

from __future__ import annotations

import glob
import html
import os
import re
from pathlib import Path

import streamlit as st


def escape_html(text: str) -> str:
    return html.escape(str(text))


def resolve_sim_dir(material: str, temperature: float, n_steps: int) -> str | None:
    """Find the simulation output directory."""
    sim_dir = f"simulations/{material}_{temperature}K_{n_steps}steps"
    if os.path.exists(sim_dir):
        return sim_dir

    possible = glob.glob(f"simulations/{material}_*K_*steps")
    if possible:
        return possible[-1]

    if st_last := st.session_state.get("last_sim_dir"):
        if os.path.exists(st_last):
            return st_last

    all_dirs = sorted(glob.glob("simulations/*"), key=os.path.getmtime)
    return all_dirs[-1] if all_dirs else None


def parse_sim_dir_name(path: str) -> dict:
    """Parse simulations/Cu_300.0K_1000steps into metadata."""
    name = Path(path).name
    match = re.match(r"^(.+)_([\d.]+)K_(\d+)steps$", name)
    if match:
        return {
            "material": match.group(1),
            "temperature": float(match.group(2)),
            "n_steps": int(match.group(3)),
            "name": name,
        }
    return {"material": name, "temperature": None, "n_steps": None, "name": name}
