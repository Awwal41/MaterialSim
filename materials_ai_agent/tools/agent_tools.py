"""LangChain-callable tools that wrap the simulation and analysis engines.

These tools have no heavy optional dependencies (no torch/pymatgen/mp-api), so
they can always be registered with the conversational agent.
"""

from typing import List, Optional

from langchain_core.tools import tool

from ..analysis_engine import analyze_all, find_simulation_dir
from ..simple_simulation import run_simple_simulation


def create_agent_tools(config) -> List:
    """Build the list of tools exposed to the LLM agent."""

    @tool
    def run_md_simulation(
        material: str,
        temperature: Optional[float] = None,
        n_steps: Optional[int] = None,
        force_field: Optional[str] = None,
        ensemble: Optional[str] = None,
    ) -> str:
        """Run molecular dynamics simulation and return a summary.

        Args:
            material: Material formula, e.g. 'Cu', 'Al', 'Si', 'H2O'.
            temperature: Temperature in Kelvin.
            n_steps: Number of MD steps to run.
            force_field: Preferred potential ('emt', 'lj', 'eam', 'tersoff').
            ensemble: Thermodynamic ensemble ('NVE', 'NVT', 'NPT').
        """
        result = run_simple_simulation(
            material=material,
            temperature=temperature,
            n_steps=n_steps,
            force_field=force_field,
            ensemble=ensemble,
        )
        if result.get("success"):
            return (
                f"{result['message']}. Output written to "
                f"{result['simulation_directory']}."
            )
        return f"Simulation failed: {result.get('error')}"

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
        analyze_simulation,
        list_available_materials,
        locate_latest_simulation,
        research_literature_and_models,
    ]
