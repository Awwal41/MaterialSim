"""Assess whether an MD run equilibrated and is safe to interpret."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HEADER = "Step Temp PotEng KinEng TotEng Press Volume"


def _parse_log(log_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, temperatures, pressures from output.log."""
    lines = log_file.read_text(encoding="utf-8").splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if HEADER in ln), None)
    if start is None:
        return np.array([]), np.array([]), np.array([])

    rows = []
    for line in lines[start:]:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                rows.append([float(p) for p in parts[:7]])
            except ValueError:
                continue
    if not rows:
        return np.array([]), np.array([]), np.array([])

    data = np.array(rows)
    return data[:, 0], data[:, 1], data[:, 5]


def _production_slice(
    steps: np.ndarray,
    temps: np.ndarray,
    meta: Optional[dict],
) -> slice:
    """Choose indices for production (post-equilibration) statistics."""
    n = len(temps)
    if n == 0:
        return slice(0, 0)

    prod_start = None
    if meta:
        prod_start = meta.get("production_start_step")

    if prod_start is not None:
        idx = int(np.searchsorted(steps, prod_start, side="left"))
        if idx < n:
            return slice(idx, n)

    # Drop duplicate step-0 rows and use latter half as production window.
    if n >= 4:
        return slice(n // 2, n)
    return slice(0, n)


def assess_simulation_quality(
    sim_dir: Path,
    target_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate equilibration and flag unreliable thermodynamic output."""
    sim_dir = Path(sim_dir)
    log_file = sim_dir / "output.log"
    meta: dict = {}
    meta_file = sim_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    target_temperature = float(
        target_temperature
        if target_temperature is not None
        else meta.get("target_temperature", 300.0)
    )

    warnings: List[str] = []
    recommendations: List[str] = []
    warnings.extend(meta.get("warnings", []))

    if not log_file.exists():
        return {
            "success": False,
            "converged": False,
            "warnings": ["output.log not found."],
            "recommendations": ["Re-run the simulation."],
        }

    steps, temps, press = _parse_log(log_file)
    if len(temps) == 0:
        return {
            "success": False,
            "converged": False,
            "warnings": ["No thermodynamic data in output.log."],
            "recommendations": ["Check that the simulation completed."],
        }

    prod = _production_slice(steps, temps, meta)
    prod_temps = temps[prod]
    prod_press = press[prod] if len(press) == len(temps) else press[prod]

    avg_t = float(np.mean(prod_temps))
    std_t = float(np.std(prod_temps))
    max_t = float(np.max(temps))
    avg_p = float(np.mean(prod_press)) if len(prod_press) else 0.0

    temp_error = abs(avg_t - target_temperature) / max(target_temperature, 1.0)
    spike_ratio = max_t / max(target_temperature, 1.0)

    converged = True

    if spike_ratio > 2.5:
        converged = False
        warnings.append(
            f"Large temperature spike detected (max {max_t:.0f} K vs target "
            f"{target_temperature:.0f} K). Early frames are not equilibrated."
        )
        recommendations.append(
            "Increase equilibration time, reduce the timestep (try 0.0005 ps), "
            "or use a material supported by the EMT potential (Cu, Al, Ni, …)."
        )

    if temp_error > 0.20:
        converged = False
        warnings.append(
            f"Production temperature {avg_t:.1f} ± {std_t:.1f} K deviates from "
            f"target {target_temperature:.0f} K by more than 20%."
        )
        recommendations.append(
            "Run longer, increase thermostat coupling (Langevin friction), or "
            "verify the interatomic potential is appropriate for this material."
        )

    if std_t > max(0.35 * target_temperature, 80.0):
        converged = False
        warnings.append(
            f"Temperature fluctuations are very large (σ = {std_t:.1f} K), "
            "indicating poor NVT control or a melting/unstable structure."
        )

    pressure_reliable = abs(avg_p) < 5000.0
    if not pressure_reliable:
        warnings.append(
            f"Average pressure {avg_p:.0f} bar is unphysical for a condensed-phase "
            "NVT run; pressure values should not be interpreted."
        )
        recommendations.append(
            "Pressure from simplified potentials (LJ/EMT) in small cells is often "
            "not meaningful. Focus on temperature and structural metrics (RDF, MSD)."
        )

    ff_label = (meta.get("force_field") or "").lower()
    is_lj = "lj" in ff_label or "lennard" in ff_label
    if is_lj and meta.get("material") in {"Si", "C", "Ge"}:
        converged = False
        warnings.append(
            f"{meta.get('material')} was simulated with Lennard-Jones, which does not "
            "reproduce covalent crystal physics. Structural and transport properties "
            "are qualitative at best."
        )
        recommendations.append(
            "For silicon, use Cu or Al demonstrations with EMT, or integrate a "
            "Tersoff/Stillinger–Weber potential for production-quality covalent MD."
        )

    if converged and not warnings:
        recommendations.append(
            "Simulation appears equilibrated in the production window. "
            "You can trust temperature and RDF trends for this potential."
        )

    return {
        "success": True,
        "converged": converged,
        "target_temperature": target_temperature,
        "avg_temperature": avg_t,
        "std_temperature": std_t,
        "max_temperature": max_t,
        "avg_pressure": avg_p,
        "pressure_reliable": pressure_reliable,
        "production_points": int(len(prod_temps)),
        "warnings": warnings,
        "recommendations": recommendations,
        "force_field": meta.get("force_field"),
        "material": meta.get("material"),
        "ensemble": meta.get("ensemble"),
    }


def format_quality_report(quality: Dict[str, Any]) -> str:
    """Human-readable quality summary for chat/voice surfaces."""
    if not quality.get("success"):
        return "Could not assess simulation quality."

    lines = []
    if quality.get("converged"):
        lines.append("**Equilibration:** appears acceptable in the production window.")
    else:
        lines.append("**Equilibration:** **not converged** — interpret results with caution.")

    lines.append(
        f"- Production T: **{quality['avg_temperature']:.1f} ± "
        f"{quality['std_temperature']:.1f} K** (target {quality['target_temperature']:.0f} K)"
    )
    if quality.get("pressure_reliable"):
        lines.append(f"- Production P: **{quality['avg_pressure']:.1f} bar**")
    else:
        lines.append("- Production P: **unreliable** (simplified potential / small cell)")

    for w in quality.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
    if quality.get("recommendations"):
        lines.append("\n**Suggestions:**")
        for r in quality["recommendations"]:
            lines.append(f"- {r}")
    return "\n".join(lines)
