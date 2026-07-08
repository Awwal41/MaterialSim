"""Main Materials AI Agent class."""

import logging
from typing import Any, Dict, List, Optional

from .config import Config
from .exceptions import MaterialsAgentError
from .materials_database import MaterialsDatabase
from .model_router import ModelRouter

logger = logging.getLogger(__name__)


class MaterialsAgent:
    """Autonomous agent for computational materials science.

    Core capabilities (simulation and analysis) run deterministically against the MD engine and do not require an LLM. The optional conversational
    ``chat`` interface uses an LLM with tool access when an API key is present.
    A :class:`ModelRouter` selects the best model per task, optionally
    researching benchmarks and papers online first.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the Materials AI Agent.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.config.create_directories()

        self.logger = logging.getLogger(__name__)
        self.materials_db = MaterialsDatabase()
        self.model_router = ModelRouter(default_model=self.config.model_name)

        # Build the toolset (real, usable tools).
        self.tools = self._initialize_tools()

        # The LLM/agent are optional: only needed for free-form chat.
        self.llm = None
        self.agent = None
        self._current_model: Optional[str] = None
        self._init_llm_agent()

        self.logger.info(
            "Materials AI Agent initialized with %d tool(s)", len(self.tools)
        )

    def _initialize_tools(self) -> List:
        """Initialize the tools the agent can call.

        Simulation and analysis tools are always available. Database and ML
        tools are optional and only loaded when their dependencies are present.
        """
        tools: List = []
        try:
            from ..tools.agent_tools import create_agent_tools

            tools = create_agent_tools(self.config)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to build agent tools: %s", exc)
        return tools

    def _init_llm_agent(self, model_name: Optional[str] = None) -> None:
        """Create the LLM-backed conversational agent when possible."""
        if not self.config.openai_api_key:
            self.logger.warning(
                "No OpenAI API key configured; conversational chat is disabled. "
                "Simulation and analysis still work."
            )
            return

        model = model_name or self.config.model_name
        try:
            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            self.llm = ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openai_api_key,
            )
            self.agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=self._system_prompt(),
            )
            self._current_model = model
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not initialize LLM agent: %s", exc)
            self.llm = None
            self.agent = None

    def reconfigure_llm(self, task_description: str) -> str:
        """Pick the best model for *task_description* and re-init if needed.

        Returns the model ID now in use.
        """
        rec = self.model_router.recommend(task_description, search_online=True)
        if rec.model_id != self._current_model and rec.provider == "openai":
            self.logger.info(
                "Switching LLM: %s -> %s (%s)",
                self._current_model,
                rec.model_id,
                rec.rationale,
            )
            self._init_llm_agent(rec.model_id)
        return rec.model_id

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the Material Sim Agent, an expert in computational "
            "materials science and molecular dynamics. You can run MD simulations, "
            "analyze their results, search literature, and recommend the best "
            "approaches. When a user asks you to run a simulation, call the "
            "simulation tool rather than only describing what to do. "
            "Users may supply their own structure files (XYZ, CIF, POSCAR) or "
            "Materials Project ids (mp-####); pass structure_file, structure_source, "
            "or mp_material_id to run_md_simulation when they do. "
            "For complex requests (alloys, NPT, pressure, supercells, compounds, "
            "duration in ps/ns), prefer run_simulation_from_instruction. "
            "Always check simulation quality: if temperature, pressure, or "
            "equilibration warnings are present, explain clearly that the run "
            "did not converge and suggest fixes (longer equilibration, smaller "
            "timestep, or EMT-supported metals like Cu/Al instead of Si with LJ). "
            "Never present unphysical averages from the equilibration phase as "
            "final results. Explain results using correct scientific terminology "
            "and units. Be concise when responding via voice."
        )

    def process_command(self, utterance: str):
        """Process a voice or text command through the voice orchestrator."""
        from ..voice.orchestrator import VoiceOrchestrator

        return VoiceOrchestrator(self).process(utterance)

    def research_best_model(self, task: str) -> Dict[str, Any]:
        """Research online and return the best LLM recommendation for *task*."""
        rec = self.model_router.recommend(task, search_online=True)
        from .literature_search import research_task

        research = research_task(task)
        return {
            "model_id": rec.model_id,
            "provider": rec.provider,
            "task_type": rec.task_type,
            "rationale": rec.rationale,
            "confidence": rec.confidence,
            "sources": rec.sources,
            "papers": research.get("papers", []),
            "web_results": research.get("web_results", []),
        }

    # ------------------------------------------------------------------
    # Core deterministic capabilities (no LLM required)
    # ------------------------------------------------------------------
    def run_simulation(self, instruction: str) -> Dict[str, Any]:
        """Run a simulation from a natural-language instruction.

        Args:
            instruction: Natural-language description of the simulation.

        Returns:
            Dictionary with ``success`` and, on success, simulation details.
        """
        self.logger.info("Running simulation: %s", instruction)
        try:
            params = self._parse_simulation_instruction(instruction)
            from ..simple_simulation import run_simple_simulation

            return run_simple_simulation(**params)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Simulation failed")
            return {"success": False, "error": str(exc)}

    def _parse_simulation_instruction(self, instruction: str) -> Dict[str, Any]:
        """Extract simulation parameters from a natural-language instruction."""
        from ..simulation_parser import parse_simulation_instruction

        spec = parse_simulation_instruction(instruction, self.config)
        return spec.to_run_kwargs()

    def analyze_results(self, simulation_path: str) -> Dict[str, Any]:
        """Analyze real simulation output (RDF, MSD, thermodynamics)."""
        self.logger.info("Analyzing results from: %s", simulation_path)
        try:
            from ..analysis_engine import analyze_all

            result = analyze_all(simulation_path)
            result.setdefault("simulation_path", simulation_path)
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Analysis failed")
            return {"success": False, "simulation_path": simulation_path, "error": str(exc)}

    def query_database(self, query: str) -> Dict[str, Any]:
        """Query materials databases (Materials Project) if configured."""
        self.logger.info("Querying database: %s", query)
        try:
            from ..tools.database import DatabaseTool

            tool = DatabaseTool(self.config)
            results = tool.query_materials_project(query)
            return {"success": results.get("success", True), "query": query, "results": results}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Database query unavailable: %s", exc)
            return {
                "success": False,
                "query": query,
                "error": (
                    "Database querying requires the optional 'mp-api' and "
                    "'pymatgen' packages and a Materials Project API key. "
                    f"({exc})"
                ),
            }

    def predict_properties(self, material: str, properties: List[str]) -> Dict[str, Any]:
        """Predict material properties using the ML tool if available."""
        self.logger.info("Predicting properties for %s: %s", material, properties)
        try:
            from ..tools.ml import MLTool  # noqa: F401

            return {
                "success": False,
                "material": material,
                "properties": properties,
                "error": (
                    "Property prediction requires a trained model. Train one via "
                    "the ML tool before requesting predictions."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "material": material,
                "properties": properties,
                "error": (
                    "ML prediction requires the optional 'torch'/'scikit-learn' "
                    f"stack. ({exc})"
                ),
            }

    # ------------------------------------------------------------------
    # Optional conversational interface
    # ------------------------------------------------------------------
    def chat(self, message: str) -> str:
        """Chat with the agent (requires a configured LLM)."""
        self.reconfigure_llm(message)
        if self.agent is None:
            return (
                "Conversational chat is unavailable because no OpenAI API key is "
                "configured. You can still run simulations and analyses directly."
            )
        try:
            result = self.agent.invoke({"messages": [("user", message)]})
            messages = result.get("messages", [])
            if messages:
                return getattr(messages[-1], "content", str(messages[-1]))
            return ""
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Chat failed")
            return f"I encountered an error: {exc}"
