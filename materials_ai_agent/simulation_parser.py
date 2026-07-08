"""Parse natural-language instructions into :class:`SimulationSpec`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .core.config import Config
from .core.materials_database import MaterialsDatabase
from .simulation_spec import SimulationSpec
from .structure_builder import parse_alloy_notation


def parse_simulation_instruction(instruction: str, config: Optional[Config] = None) -> SimulationSpec:
    """Extract a full simulation spec from natural language."""
    config = config or Config.from_env()
    text = instruction
    lower = instruction.lower()
    db = MaterialsDatabase()

    alloy_elements = None
    alloy_fractions = None
    # Compounds like Al2O3 must be detected before alloy notation.
    compound_material = _extract_compound(instruction)
    if compound_material:
        material = compound_material
    else:
        alloy = parse_alloy_notation(instruction)
        if alloy:
            alloy_elements, alloy_fractions = alloy
            material = "".join(alloy_elements)
        else:
            material = _extract_material(instruction, lower, db)

    temperature = _extract_temperature(lower, config)
    pressure = _extract_pressure(lower, config)
    timestep = _extract_timestep(lower, config)
    n_steps = _extract_n_steps(lower, config, timestep)
    ensemble = _extract_ensemble(lower, config)
    thermostat = _extract_thermostat(lower, config)
    force_field = _extract_force_field(lower, config)
    structure_file = _extract_structure_file(instruction)
    mp_material_id = _extract_mp_material_id(instruction)
    structure_source = _extract_structure_source(lower, structure_file, mp_material_id)
    supercell_reps = _extract_supercell(lower)
    target_atoms = _extract_target_atoms(lower)
    output_frequency = _extract_output_frequency(lower)

    return SimulationSpec(
        material=material,
        temperature=temperature,
        pressure=pressure,
        n_steps=n_steps,
        timestep=timestep,
        ensemble=ensemble,
        thermostat=thermostat,
        force_field=force_field,
        structure_source=structure_source,
        structure_file=structure_file,
        mp_material_id=mp_material_id,
        supercell_reps=supercell_reps,
        target_atoms=target_atoms,
        output_frequency=output_frequency,
        alloy_elements=alloy_elements,
        alloy_fractions=alloy_fractions,
        description=instruction.strip()[:500],
    )


def _extract_compound(instruction: str) -> Optional[str]:
    for compound in ("Al2O3", "SiO2", "TiO2", "MgO", "Fe2O3"):
        if re.search(rf"\b{re.escape(compound)}\b", instruction, re.I):
            return compound
    return None


def _extract_material(instruction: str, lower: str, db: MaterialsDatabase) -> str:
    mp_id = _extract_mp_material_id(instruction)
    if mp_id:
        return mp_id

    if _uses_user_structure(lower):
        return "custom"

    for formula in sorted(db.get_all_materials(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(formula)}\b", instruction, re.I):
            return formula

    keyword_map = {
        "silicon": "Si", "aluminum": "Al", "aluminium": "Al",
        "copper": "Cu", "iron": "Fe", "water": "H2O",
        "carbon": "C", "gold": "Au", "nickel": "Ni",
        "magnesium": "Mg", "titanium": "Ti", "platinum": "Pt",
    }
    for keyword, formula in keyword_map.items():
        if keyword in lower:
            return formula

    m = re.search(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b", instruction)
    if m:
        return m.group(1)

    # No material could be resolved. Never silently substitute a default
    # (previously this returned "Cu"), which produced confidently-wrong runs.
    # Callers must detect the sentinel and ask the user to clarify.
    return "unresolved"


def _extract_temperature(lower: str, config: Config) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*k(?:elvin)?\b", lower)
    if m:
        return max(config.min_temperature, min(float(m.group(1)), config.max_temperature))
    if "room temperature" in lower or "room temp" in lower:
        return 300.0
    return config.default_temperature


def _extract_pressure(lower: str, config: Config) -> float:
    # GPa -> atm (1 GPa ≈ 9869 atm)
    m = re.search(r"(\d+(?:\.\d+)?)\s*gpa", lower)
    if m:
        return float(m.group(1)) * 9869.23
    m = re.search(r"(\d+(?:\.\d+)?)\s*bar", lower)
    if m:
        return float(m.group(1)) * 0.986923
    m = re.search(r"(\d+(?:\.\d+)?)\s*atm", lower)
    if m:
        return float(m.group(1))
    if "ambient pressure" in lower or "standard pressure" in lower:
        return 1.0
    return config.default_pressure


def _extract_timestep(lower: str, config: Config) -> float:
    m = re.search(r"timestep\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:ps|fs)", lower)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if "fs" in m.group(0) and "ps" not in m.group(0) else val
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ps|fs)\s+timestep", lower)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if "fs" in m.group(0) else val
    m = re.search(r"dt\s*=\s*(\d+(?:\.\d+)?)\s*(?:ps|fs)?", lower)
    if m:
        return float(m.group(1))
    return config.default_timestep


def _extract_n_steps(lower: str, config: Config, timestep_ps: float) -> int:
    m = re.search(r"(\d+)\s*steps?", lower)
    if m:
        return max(config.min_n_steps, min(int(m.group(1)), config.max_n_steps))

    # Duration in ps or ns.
    m = re.search(r"(\d+(?:\.\d+)?)\s*ns", lower)
    if m:
        duration_ps = float(m.group(1)) * 1000.0
        return max(config.min_n_steps, min(int(duration_ps / timestep_ps), config.max_n_steps))

    m = re.search(r"(\d+(?:\.\d+)?)\s*ps\b", lower)
    if m and "timestep" not in lower[: m.start()]:
        duration_ps = float(m.group(1))
        return max(config.min_n_steps, min(int(duration_ps / timestep_ps), config.max_n_steps))

    return config.default_n_steps


def _extract_ensemble(lower: str, config: Config) -> str:
    for ens in config.available_ensembles:
        if ens.lower() in lower or f"{ens.lower()} ensemble" in lower:
            return ens.upper()
    return config.default_ensemble.upper()


def _extract_thermostat(lower: str, config: Config) -> str:
    if "berendsen" in lower:
        return "Berendsen"
    if "nose" in lower or "hoover" in lower:
        return "Nose-Hoover"
    if "langevin" in lower:
        return "Langevin"
    if "nve" in lower and "ensemble" in lower:
        return "None"
    return config.default_thermostat


def _extract_force_field(lower: str, config: Config) -> Optional[str]:
    for ff in config.available_force_fields:
        if ff.lower() in lower:
            return ff
    if "emt" in lower:
        return "emt"
    return config.default_force_field


def _extract_structure_file(instruction: str) -> Optional[str]:
    quoted = re.search(
        r'(?:file|structure|poscar|cif|xyz)[:\s]+["\']([^"\']+\.(?:xyz|cif|poscar|vasp|pdb|json))["\']',
        instruction,
        re.I,
    )
    if quoted and Path(quoted.group(1)).exists():
        return quoted.group(1)

    m = re.search(
        r"(?:file|structure|poscar|cif|xyz|from)[:\s]+([^\s,;]+\.(?:xyz|cif|poscar|vasp|pdb|json))",
        instruction,
        re.I,
    )
    if m and Path(m.group(1)).exists():
        return m.group(1)
    m = re.search(r"\b([\w./\\-]+\.(?:xyz|cif|poscar|vasp|pdb))\b", instruction, re.I)
    if m and Path(m.group(1)).exists():
        return m.group(1)
    return None


def _extract_mp_material_id(instruction: str) -> Optional[str]:
    m = re.search(r"\b(mp-\d+)\b", instruction, re.I)
    return m.group(1) if m else None


def _uses_user_structure(lower: str) -> bool:
    markers = (
        "upload",
        "my structure",
        "user structure",
        "structure file",
        "from file",
        "from cif",
        "from poscar",
        "from xyz",
        "materials project",
        "material project",
    )
    return any(marker in lower for marker in markers)


def _extract_structure_source(
    lower: str,
    structure_file: Optional[str],
    mp_material_id: Optional[str],
) -> str:
    if structure_file:
        return "file"
    if mp_material_id:
        return "material_project"
    if any(k in lower for k in ("materials project", "material project", "from mp", "mp structure")):
        return "material_project"
    if any(k in lower for k in ("upload", "my structure", "user structure", "structure file", "from file")):
        return "file"
    return "generate"


def _extract_supercell(lower: str) -> Optional[tuple[int, int, int]]:
    m = re.search(r"supercell\s*(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)", lower)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)\s+supercell", lower)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _extract_target_atoms(lower: str) -> int:
    m = re.search(r"(\d+)\s*atoms?", lower)
    if m:
        return max(8, min(int(m.group(1)), 4096))
    return 64


def _extract_output_frequency(lower: str) -> int:
    m = re.search(r"(?:every|output|write|save)\s*(\d+)\s*steps?", lower)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"output\s+frequency\s*(\d+)", lower)
    if m:
        return max(1, int(m.group(1)))
    return 100
