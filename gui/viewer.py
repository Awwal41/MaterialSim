"""Interactive 3D structure / trajectory viewer and run telemetry.

The viewer renders atomic structures and MD trajectories with py3Dmol (embedded
via Streamlit's HTML component, with an optional stmol path). Everything degrades
gracefully: if the optional viewer libraries are missing, the user gets a clear
install hint instead of a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

# Dark canvas that matches the app theme.
_BG = "#0f1419"


# --------------------------------------------------------------------------- #
# XYZ / trajectory parsing
# --------------------------------------------------------------------------- #
def _parse_xyz_frames(text: str, max_frames: int, max_atoms: int) -> List[str]:
    """Parse a (possibly multi-frame, possibly extended) XYZ document.

    Handles the project's custom dump format ``element x y z fx fy fz`` by
    keeping only ``element`` plus the first three coordinates, so py3Dmol's
    strict XYZ reader is happy. Returns a list of clean XYZ frame strings.
    """
    lines = text.splitlines()
    frames: List[str] = []
    i = 0
    n = len(lines)
    while i < n and len(frames) < max_frames:
        head = lines[i].strip()
        if not head.isdigit():
            i += 1
            continue
        n_atoms = int(head)
        atom_lines = lines[i + 2 : i + 2 + n_atoms]
        clean: List[str] = []
        for raw in atom_lines:
            parts = raw.split()
            if len(parts) >= 4:
                el, x, y, z = parts[0], parts[1], parts[2], parts[3]
                clean.append(f"{el} {x} {y} {z}")
            if len(clean) >= max_atoms:
                break
        if clean:
            frame = f"{len(clean)}\nframe {len(frames)}\n" + "\n".join(clean)
            frames.append(frame)
        i += 2 + n_atoms
    return frames


def _load_frames(source, max_frames: int, max_atoms: int) -> Tuple[List[str], int]:
    """Return (frames, total_frames_available) from a path or ASE Atoms."""
    # ASE Atoms (or list of Atoms) support.
    if not isinstance(source, (str, Path)):
        try:
            from ase.io import write  # noqa: WPS433
            import io

            atoms_list = source if isinstance(source, (list, tuple)) else [source]
            buf = io.StringIO()
            for atoms in atoms_list[:max_frames]:
                write(buf, atoms, format="xyz")
            frames = _parse_xyz_frames(buf.getvalue(), max_frames, max_atoms)
            return frames, len(atoms_list)
        except Exception:
            return [], 0

    path = Path(source)
    if not path.exists():
        return [], 0
    text = path.read_text(errors="ignore")
    # Count frames cheaply for the caption.
    total = sum(1 for ln in text.splitlines() if ln.strip().isdigit())
    frames = _parse_xyz_frames(text, max_frames, max_atoms)
    return frames, total


def find_viewable_file(sim_dir: str | Path) -> Optional[Path]:
    """Pick the best file to visualize from a simulation directory."""
    d = Path(sim_dir)
    for name in ("trajectory.xyz", "structure.xyz", "final_structure.xyz"):
        if (d / name).exists():
            return d / name
    xyz = sorted(d.glob("*.xyz"))
    return xyz[0] if xyz else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_structure(
    source,
    *,
    height: int = 460,
    animate: bool = True,
    max_frames: int = 60,
    max_atoms: int = 4000,
    caption: Optional[str] = None,
    key: Optional[str] = None,
) -> bool:
    """Render an interactive 3D structure/trajectory.

    Args:
        source: Path to an XYZ file, or an ASE ``Atoms``/list of ``Atoms``.
        animate: Animate multi-frame trajectories.

    Returns:
        True if something was rendered, False otherwise.
    """
    try:
        import py3Dmol  # noqa: WPS433
    except Exception:
        st.info(
            "3D viewer needs `py3Dmol`. Install it with "
            "`pip install py3Dmol stmol` (or `pip install -r requirements_gui.txt`)."
        )
        return False

    frames, total = _load_frames(source, max_frames, max_atoms)
    if not frames:
        st.caption("No viewable atomic coordinates found.")
        return False

    view = py3Dmol.view(width="100%", height=height)
    view.setBackgroundColor(_BG)

    is_traj = animate and len(frames) > 1
    if is_traj:
        view.addModelsAsFrames("\n".join(frames), "xyz")
    else:
        view.addModel(frames[0], "xyz")

    view.setStyle(
        {
            "sphere": {"scale": 0.30},
            "stick": {"radius": 0.14},
        }
    )
    view.zoomTo()
    if is_traj:
        view.animate({"loop": "forward", "interval": 120})

    html = view._make_html()  # noqa: SLF001 - public-enough for embedding
    try:
        import streamlit.components.v1 as components  # noqa: WPS433

        components.html(html, height=height + 10, scrolling=False)
    except Exception:
        st.markdown(html, unsafe_allow_html=True)

    if caption:
        st.caption(caption)
    elif is_traj:
        shown = min(len(frames), max_frames)
        st.caption(f"Animated trajectory — {shown} of {total} frames · drag to rotate, scroll to zoom")
    else:
        st.caption("Static structure · drag to rotate, scroll to zoom")
    return True


# --------------------------------------------------------------------------- #
# Telemetry (thermodynamics over the run)
# --------------------------------------------------------------------------- #
_THERMO_COLUMNS = ["Step", "Temp", "PotEng", "KinEng", "TotEng", "Press", "Volume"]


def load_thermo_dataframe(sim_dir: str | Path):
    """Parse ``output.log`` (Step Temp PotEng KinEng TotEng Press Volume) to a DataFrame."""
    import pandas as pd

    log = Path(sim_dir) / "output.log"
    if not log.exists():
        return None

    rows: List[List[float]] = []
    header: Optional[List[str]] = None
    for line in log.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Step") and "Temp" in s:
            header = s.split()
            continue
        parts = s.split()
        # Numeric data rows only.
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        if header and len(values) == len(header):
            rows.append(values)
        elif not header and len(values) == len(_THERMO_COLUMNS):
            rows.append(values)

    if not rows:
        return None
    cols = header if header else _THERMO_COLUMNS
    return pd.DataFrame(rows, columns=cols)


def render_telemetry(sim_dir: str | Path) -> bool:
    """Render live-style thermodynamic charts for a run. Returns True if plotted."""
    df = load_thermo_dataframe(sim_dir)
    if df is None or df.empty or "Step" not in df.columns:
        st.caption("No thermodynamic log (`output.log`) to chart yet.")
        return False

    df = df.set_index("Step")

    c1, c2 = st.columns(2)
    if "Temp" in df.columns:
        with c1:
            st.markdown("**Temperature (K)**")
            st.line_chart(df[["Temp"]], height=220)
    energy_cols = [c for c in ("PotEng", "KinEng", "TotEng") if c in df.columns]
    if energy_cols:
        with c2:
            st.markdown("**Energy (eV)**")
            st.line_chart(df[energy_cols], height=220)

    c3, c4 = st.columns(2)
    if "Press" in df.columns:
        with c3:
            st.markdown("**Pressure (bar)**")
            st.line_chart(df[["Press"]], height=200)
    if "Volume" in df.columns:
        with c4:
            st.markdown("**Volume (Å³)**")
            st.line_chart(df[["Volume"]], height=200)
    return True
