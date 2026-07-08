"""Post-processing analysis for MD simulation output.

Operates on the trajectory/log files produced by ``simple_simulation`` (and the
compatible LAMMPS format).
"""

import glob
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def find_simulation_dir(
    material: Optional[str] = None,
    temperature: Optional[float] = None,
    n_steps: Optional[int] = None,
    base: str = "simulations",
) -> Optional[Path]:
    """Locate a simulation directory, tolerant of formatting differences."""
    base_path = Path(base)
    if material is not None and temperature is not None and n_steps is not None:
        exact = base_path / f"{material}_{temperature}K_{n_steps}steps"
        if exact.exists():
            return exact

    if material is not None:
        matches = sorted(glob.glob(str(base_path / f"{material}_*K_*steps")))
        if matches:
            return Path(max(matches, key=os.path.getmtime))

    matches = sorted(glob.glob(str(base_path / "*")))
    dirs = [Path(m) for m in matches if Path(m).is_dir()]
    if dirs:
        return max(dirs, key=os.path.getmtime)
    return None


def _read_xyz_frames(data_file: Path) -> List[np.ndarray]:
    """Read all frames from an XYZ-style trajectory into a list of Nx3 arrays."""
    with open(data_file, "r") as fh:
        lines = fh.readlines()

    frames: List[np.ndarray] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        stripped = lines[i].strip()
        if stripped.isdigit():
            n_atoms = int(stripped)
            i += 2  # skip count + comment line
            positions = []
            for j in range(n_atoms):
                if i + j >= n_lines:
                    break
                parts = lines[i + j].strip().split()
                if len(parts) >= 4:
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if positions:
                frames.append(np.array(positions))
            i += n_atoms
        else:
            i += 1
    return frames


def _resolve_data_file(sim_dir: Path) -> Optional[Path]:
    for name in ("trajectory.xyz", "structure.xyz"):
        candidate = sim_dir / name
        if candidate.exists():
            return candidate
    return None


def compute_rdf(sim_dir: Path, r_max: float = 10.0, n_bins: int = 200) -> Dict[str, Any]:
    """Compute the radial distribution function from the first frame."""
    data_file = _resolve_data_file(sim_dir)
    if data_file is None:
        return {"success": False, "error": "No trajectory.xyz or structure.xyz found."}

    frames = _read_xyz_frames(data_file)
    if not frames:
        return {"success": False, "error": "Could not parse any atomic positions."}

    positions = frames[0]
    n_atoms = len(positions)
    if n_atoms < 2:
        return {"success": False, "error": "Need at least 2 atoms to compute an RDF."}

    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    dr = r_bins[1] - r_bins[0]

    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n_atoms, k=1)
    distances = dist_matrix[iu]
    distances = distances[distances < r_max]

    hist, _ = np.histogram(distances, bins=r_bins)
    rho = n_atoms / (4.0 / 3.0 * np.pi * r_max ** 3)
    shell = 4 * np.pi * r_centers ** 2 * dr
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = hist / (shell * rho * n_atoms)
        g_r = np.nan_to_num(g_r)

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(g_r, height=0.1)
    peak_distances = r_centers[peaks]

    plt.figure(figsize=(10, 6))
    plt.plot(r_centers, g_r, "b-", linewidth=2, label="g(r)")
    if len(peaks):
        plt.scatter(peak_distances, g_r[peaks], color="red", s=40, zorder=5, label="peaks")
    plt.xlabel("Distance (Angstrom)")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function - {sim_dir.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_file = sim_dir / "rdf_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "success": True,
        "n_atoms": n_atoms,
        "first_peak": float(peak_distances[0]) if len(peak_distances) else None,
        "n_peaks": int(len(peaks)),
        "max_g_r": float(np.max(g_r)),
        "plot_file": str(plot_file),
    }


def compute_msd(sim_dir: Path) -> Dict[str, Any]:
    """Compute mean squared displacement across trajectory frames."""
    data_file = _resolve_data_file(sim_dir)
    if data_file is None:
        return {"success": False, "error": "No trajectory data found."}

    frames = _read_xyz_frames(data_file)
    if len(frames) < 2:
        return {
            "success": False,
            "error": "Need at least 2 trajectory frames to compute MSD.",
        }

    n_atoms = min(len(f) for f in frames)
    positions = np.array([f[:n_atoms] for f in frames])  # (T, N, 3)
    initial = positions[0]
    disp = positions - initial[None, :, :]
    msd = (disp ** 2).sum(axis=-1).mean(axis=1)  # (T,)
    time_steps = np.arange(len(msd))

    diffusion_coeff = 0.0
    if len(msd) > 10:
        from scipy import stats

        slope, *_ = stats.linregress(time_steps[10:], msd[10:])
        diffusion_coeff = slope / 6.0

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, msd, "b-", linewidth=2, label="Average MSD")
    plt.xlabel("Frame")
    plt.ylabel("MSD (Angstrom^2)")
    plt.title(f"Mean Squared Displacement - {sim_dir.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_file = sim_dir / "msd_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "success": True,
        "n_frames": len(frames),
        "n_atoms": n_atoms,
        "diffusion_coefficient": float(diffusion_coeff),
        "final_msd": float(msd[-1]),
        "plot_file": str(plot_file),
    }


def compute_thermodynamics(sim_dir: Path) -> Dict[str, Any]:
    """Parse the thermodynamic log and summarize production-window statistics."""
    from .simulation_quality import assess_simulation_quality

    log_file = sim_dir / "output.log"
    if not log_file.exists():
        return {"success": False, "error": "output.log not found."}

    meta: dict = {}
    meta_file = sim_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    with open(log_file, "r") as fh:
        lines = fh.readlines()

    header = "Step Temp PotEng KinEng TotEng Press Volume"
    data_start = next((i + 1 for i, ln in enumerate(lines) if header in ln), None)
    if data_start is None:
        return {"success": False, "error": "Thermodynamic header not found in log."}

    rows = []
    for line in lines[data_start:]:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                rows.append([float(p) for p in parts[:7]])
            except ValueError:
                continue
    if not rows:
        return {"success": False, "error": "No numeric thermodynamic data found."}

    data = np.array(rows)
    steps, temps, poteng, kineng, toteng, press, volume = (data[:, i] for i in range(7))

    prod_start = meta.get("production_start_step")
    if prod_start is not None:
        mask = steps >= prod_start
    else:
        mask = np.zeros(len(steps), dtype=bool)
        mask[len(steps) // 2 :] = True

    if not mask.any():
        mask = np.ones(len(steps), dtype=bool)

    prod_temps = temps[mask]
    prod_press = press[mask]
    prod_steps = steps[mask]

    quality = assess_simulation_quality(
        sim_dir, meta.get("target_temperature", float(np.mean(prod_temps)))
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax1, ax2), (ax3, ax4) = axes
    ax1.plot(steps, temps, "b-", alpha=0.5, label="all frames")
    ax1.plot(prod_steps, prod_temps, "c-", linewidth=2, label="production")
    ax1.axhline(prod_temps.mean(), color="r", ls="--", alpha=0.7,
                label=f"prod avg {prod_temps.mean():.1f} K")
    target = meta.get("target_temperature")
    if target:
        ax1.axhline(target, color="lime", ls=":", alpha=0.8, label=f"target {target:g} K")
    ax1.set(xlabel="Step", ylabel="Temperature (K)", title="Temperature")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, poteng, "r-", label="PE")
    ax2.plot(steps, kineng, "g-", label="KE")
    ax2.plot(steps, toteng, "b-", label="Total")
    ax2.set(xlabel="Step", ylabel="Energy (eV)", title="Energy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(steps, press, "g-", alpha=0.6)
    ax3.plot(prod_steps, prod_press, "g-", linewidth=2)
    ax3.set(xlabel="Step", ylabel="Pressure (bar)", title="Pressure")
    ax3.grid(True, alpha=0.3)

    ax4.plot(steps, volume, "m-")
    ax4.set(xlabel="Step", ylabel="Volume (Angstrom^3)", title="Volume")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = sim_dir / "thermodynamic_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "success": True,
        "avg_temperature": float(prod_temps.mean()),
        "std_temperature": float(prod_temps.std()),
        "avg_pressure": float(prod_press.mean()),
        "std_pressure": float(prod_press.std()),
        "avg_total_energy": float(toteng[mask].mean()),
        "n_points": int(len(prod_temps)),
        "n_points_total": int(len(steps)),
        "target_temperature": meta.get("target_temperature"),
        "production_only": True,
        "converged": quality.get("converged", False),
        "pressure_reliable": quality.get("pressure_reliable", False),
        "quality_warnings": quality.get("warnings", []),
        "recommendations": quality.get("recommendations", []),
        "plot_file": str(plot_file),
    }


def analyze_all(sim_dir: Path) -> Dict[str, Any]:
    """Run every available analysis and return a combined result."""
    from .simulation_quality import assess_simulation_quality, format_quality_report

    sim_dir = Path(sim_dir)
    if not sim_dir.exists():
        return {"success": False, "error": f"Simulation directory not found: {sim_dir}"}

    quality = assess_simulation_quality(sim_dir)
    thermo = compute_thermodynamics(sim_dir)

    return {
        "success": True,
        "simulation_directory": str(sim_dir),
        "quality": quality,
        "quality_report": format_quality_report(quality),
        "rdf": compute_rdf(sim_dir),
        "msd": compute_msd(sim_dir),
        "thermodynamics": thermo,
    }
