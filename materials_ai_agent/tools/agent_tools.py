"""LangChain-callable tools that wrap the simulation and analysis engines.

These tools have no heavy optional dependencies (no torch/pymatgen/mp-api), so
they can always be registered with the conversational agent.
"""

from typing import List, Optional

from langchain_core.tools import tool

from ..analysis_engine import analyze_all, find_simulation_dir
from ..orchestrator import run_spec
from ..simple_simulation import run_simple_simulation
from ..spec.extractor import extract_spec


def create_agent_tools(config) -> List:
    """Build the list of tools exposed to the LLM agent."""

    @tool
    def run_md_simulation(
        material: str,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        n_steps: Optional[int] = None,
        force_field: Optional[str] = None,
        ensemble: Optional[str] = None,
        thermostat: Optional[str] = None,
        timestep: Optional[float] = None,
        target_atoms: Optional[int] = None,
        structure_file: Optional[str] = None,
        structure_source: Optional[str] = None,
        mp_material_id: Optional[str] = None,
        supercell_reps: Optional[str] = None,
        engine: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> str:
        """Run molecular dynamics simulation and return a summary.

        Args:
            material: Formula ('Cu', 'Al2O3', 'CuNi'), SMILES, or 'custom' with a file.
            temperature: Temperature in Kelvin.
            pressure: Pressure in atm (for NPT).
            n_steps: Number of MD steps to run.
            force_field: Potential kind ('auto', 'emt', 'lj', 'eam', 'tersoff',
                'meam', 'reaxff', 'mace', 'chgnet', 'm3gnet', 'opls', 'gaff').
            ensemble: Thermodynamic ensemble ('NVE', 'NVT', 'NPT').
            thermostat: Thermostat ('auto', 'langevin', 'berendsen', 'nose-hoover').
            timestep: Integration timestep in ps.
            target_atoms: Approximate number of atoms in the generated supercell.
            structure_file: Path to XYZ/CIF/POSCAR/PDB to load instead of building.
            structure_source: 'generate', 'file'/'upload', or 'material_project'.
            mp_material_id: Materials Project id (e.g. 'mp-1234') when MP is the source.
            supercell_reps: Supercell as '4x4x4' when replicating a loaded structure.
            engine: 'auto', 'ase', 'lammps', or 'openmm'.
            protocol: 'equilibrium', 'nemd', 'msst', or 'deformation'.
        """
        reps = None
        if supercell_reps:
            parts = supercell_reps.lower().replace("×", "x").split("x")
            if len(parts) == 3:
                reps = tuple(int(p) for p in parts)

        resolved_source = structure_source
        if not resolved_source:
            if structure_file:
                resolved_source = "file"
            elif mp_material_id:
                resolved_source = "material_project"
            else:
                resolved_source = "generate"

        result = run_simple_simulation(
            material=material,
            temperature=temperature,
            pressure=pressure,
            n_steps=n_steps,
            force_field=force_field,
            ensemble=ensemble,
            thermostat=thermostat,
            timestep=timestep,
            target_atoms=target_atoms or 64,
            structure_file=structure_file,
            structure_source=resolved_source,
            mp_material_id=mp_material_id,
            supercell_reps=reps,
            engine=engine,
            protocol=protocol,
        )
        if result.get("success"):
            return (
                f"{result['message']}. Output written to "
                f"{result['simulation_directory']}."
            )
        if result.get("needs_clarification"):
            return f"I need more information before running: {result.get('error')}"
        return f"Simulation failed: {result.get('error')}"

    @tool
    def run_simulation_from_instruction(instruction: str) -> str:
        """Parse a natural-language MD request and run the simulation.

        Handles engine/potential/protocol/system selection automatically.
        Examples: 'NPT CuNi alloy at 800 K and 50 bar for 20 ps',
        'MSST shock of Cu along z at 8 km/s', 'NEMD thermal conductivity of Si',
        'MACE simulation of LiFePO4 at 300 K', 'tensile test of Al along x'.
        """
        spec = extract_spec(instruction, config)
        result = run_spec(spec, mp_api_key=getattr(config, "mp_api_key", None))
        if result.get("success"):
            return (
                f"Parsed: {spec.summary()}. {result['message']} "
                f"Output: {result['simulation_directory']}."
            )
        if result.get("needs_clarification"):
            return f"I need more information before running: {result.get('error')}"
        return f"Simulation failed: {result.get('error')}"

    @tool
    def list_simulation_capabilities() -> str:
        """List which MD engines, potentials, and protocols are available here."""
        from ..bootstrap import ensure
        from ..engines.registry import available_engines, list_engines
        from ..protocols.registry import list_protocols

        ensure()
        engines = available_engines()
        installed = ", ".join(engines) if engines else "none"
        return (
            f"Engines installed: {installed} (registered: {', '.join(list_engines())}). "
            f"Runnable potentials: {', '.join(config.runnable_force_fields())}. "
            f"Protocols: {', '.join(list_protocols())}."
        )

    @tool
    def analyze_simulation(simulation_directory: str) -> str:
        """Analyze a completed simulation (RDF, MSD, thermodynamics).

        Args:
            simulation_directory: Path to a simulation output directory.
        """
        result = analyze_all(simulation_directory)
        if not result.get("success"):
            return f"Analysis failed: {result.get('error')}"

        parts = []
        if result.get("quality_report"):
            parts.append(result["quality_report"])
        parts.append(f"Analysis of {result['simulation_directory']}:")
        thermo = result.get("thermodynamics", {})
        if thermo.get("success"):
            parts.append(
                f"Production temperature {thermo['avg_temperature']:.1f} K "
                f"(+/- {thermo['std_temperature']:.1f} K)."
            )
            if thermo.get("pressure_reliable"):
                parts.append(f"Production pressure {thermo['avg_pressure']:.1f} bar.")
            else:
                parts.append("Pressure values are unreliable for this potential.")
            if not thermo.get("converged"):
                parts.append("WARNING: simulation did not equilibrate.")
        rdf = result.get("rdf", {})
        if rdf.get("success") and rdf.get("first_peak"):
            parts.append(f"RDF first peak at {rdf['first_peak']:.2f} Angstrom.")
        msd = result.get("msd", {})
        if msd.get("success") and thermo.get("converged"):
            parts.append(
                f"Diffusion coefficient ~ {msd['diffusion_coefficient']:.3e} "
                "Angstrom^2/frame."
            )
        for rec in thermo.get("recommendations", [])[:2]:
            parts.append(f"Suggestion: {rec}")
        return " ".join(parts)

    @tool
    def list_available_materials() -> str:
        """List materials known to the built-in materials database."""
        from ..core.materials_database import MaterialsDatabase

        db = MaterialsDatabase()
        formulas = ", ".join(sorted(db.get_all_materials().keys()))
        return f"Available materials: {formulas}."

    @tool
    def locate_latest_simulation() -> str:
        """Find the most recent simulation directory on disk."""
        sim_dir = find_simulation_dir()
        return str(sim_dir) if sim_dir else "No simulation directories found yet."

    @tool
    def research_literature_and_models(task_description: str) -> str:
        """Search arXiv papers and the web to find the best LLM/approach for a task.

        Args:
            task_description: The materials-science task to research.
        """
        from ..core.literature_search import research_task
        from ..core.model_router import ModelRouter

        research = research_task(task_description)
        rec = ModelRouter().recommend(task_description, search_online=True)
        parts = [
            f"Recommended model: {rec.model_id} ({rec.rationale})",
            f"Papers found: {research['paper_count']}, web results: {research['web_count']}.",
        ]
        for p in research.get("papers", [])[:2]:
            parts.append(f"Paper: {p['title'][:80]}")
        return " ".join(parts)

    return [
        run_md_simulation,
        run_simulation_from_instruction,
        list_simulation_capabilities,
        analyze_simulation,
        list_available_materials,
        locate_latest_simulation,
        research_literature_and_models,
    ]
