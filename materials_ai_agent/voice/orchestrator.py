"""Voice command orchestrator — routes spoken/text commands to agent actions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent import MaterialsAgent

logger = logging.getLogger(__name__)


@dataclass
class VoiceResponse:
    """Structured response from a voice command."""

    spoken_text: str
    display_text: str
    action: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None


class VoiceOrchestrator:
    """Interpret voice/text commands and drive the full materials workflow."""

    INTENT_PATTERNS: Dict[str, List[str]] = {
        "run_simulation": [
            r"\b(simulate|simulation|run\s+(a\s+)?md|molecular\s+dynamics)\b",
            r"\brun\s+(a\s+)?simulation\b",
        ],
        "analyze": [
            r"\b(analyz|analysis|analyse)\b",
            r"\b(rdf|msd|thermodynamic|results?)\b",
        ],
        "predict": [
            r"\b(predict|prediction|forecast)\b",
            r"\b(property|properties|elastic|conductivity)\b",
        ],
        "research": [
            r"\b(research|paper|literature|arxiv|best\s+model|benchmark)\b",
        ],
        "database": [
            r"\b(database|materials\s+project|query|lookup)\b",
        ],
        "status": [
            r"\b(status|ready|online|hello|hey)\b",
        ],
    }

    def __init__(self, agent: "MaterialsAgent"):
        self.agent = agent

    def process(self, utterance: str) -> VoiceResponse:
        """Process a voice or text utterance end-to-end."""
        utterance = utterance.strip()
        if not utterance:
            return VoiceResponse(
                spoken_text="I didn't catch that. Please try again.",
                display_text="No input received.",
                action="none",
                success=False,
            )

        intent = self._detect_intent(utterance)
        router = self.agent.model_router
        rec = router.recommend(utterance, search_online=(intent == "research"))
        model_id = rec.model_id

        if intent == "run_simulation":
            return self._handle_simulation(utterance, model_id)
        if intent == "analyze":
            return self._handle_analysis(utterance, model_id)
        if intent == "predict":
            return self._handle_predict(utterance, model_id)
        if intent == "research":
            return self._handle_research(utterance, model_id)
        if intent == "database":
            return self._handle_database(utterance, model_id)
        if intent == "status":
            return self._handle_status(model_id)

        return self._handle_chat(utterance, model_id)

    def _detect_intent(self, text: str) -> str:
        lower = text.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(p, lower) for p in patterns):
                return intent
        return "chat"

    def _handle_simulation(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance)
        result = self.agent.run_simulation(utterance)
        if result.get("success"):
            spoken = (
                f"Simulation complete. {result['message']}. "
                f"The output is in {result['simulation_directory']}."
            )
            display = f"**Simulation complete**\n\n{result['message']}\n\nDirectory: `{result['simulation_directory']}`"
            quality = result.get("quality") or {}
            if quality and not quality.get("converged"):
                from ..simulation_quality import format_quality_report
                display += f"\n\n{format_quality_report(quality)}"
                spoken = (
                    "Simulation finished, but it did not equilibrate properly. "
                    + spoken
                )
            for w in result.get("warnings", []):
                display += f"\n- ⚠️ {w}"
            return VoiceResponse(
                spoken_text=spoken,
                display_text=display,
                action="run_simulation",
                success=True,
                data=result,
                model_used=model_id,
            )
        return VoiceResponse(
            spoken_text=f"Simulation failed. {result.get('error', 'Unknown error')}.",
            display_text=f"Simulation failed: {result.get('error')}",
            action="run_simulation",
            success=False,
            data=result,
            model_used=model_id,
        )

    def _handle_analysis(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance)
        from ..analysis_engine import find_simulation_dir

        sim_dir = find_simulation_dir()
        if sim_dir is None:
            return VoiceResponse(
                spoken_text="No simulation results found. Run a simulation first.",
                display_text="No simulation directories found.",
                action="analyze",
                success=False,
                model_used=model_id,
            )

        result = self.agent.analyze_results(str(sim_dir))
        if not result.get("success"):
            return VoiceResponse(
                spoken_text="Analysis failed.",
                display_text=f"Analysis failed: {result.get('error')}",
                action="analyze",
                success=False,
                model_used=model_id,
            )

        parts = []
        spoken_parts = []
        quality = result.get("quality", {})
        if quality.get("success"):
            from ..simulation_quality import format_quality_report
            parts.append(format_quality_report(quality))
            if not quality.get("converged"):
                spoken_parts.append(
                    "Warning: this simulation did not equilibrate. Results may be unreliable."
                )
            else:
                spoken_parts.append("Simulation quality looks acceptable.")

        parts.append(f"**Analysis of {sim_dir.name}**")
        rdf = result.get("rdf", {})
        if rdf.get("success") and rdf.get("first_peak"):
            parts.append(f"- RDF first peak: **{rdf['first_peak']:.2f} Å**")
            spoken_parts.append(
                f"RDF first peak at {rdf['first_peak']:.2f} angstroms."
            )
        thermo = result.get("thermodynamics", {})
        if thermo.get("success"):
            parts.append(
                f"- Production temperature: **{thermo['avg_temperature']:.1f} ± "
                f"{thermo['std_temperature']:.1f} K**"
            )
            spoken_parts.append(
                f"Production temperature {thermo['avg_temperature']:.0f} kelvin."
            )
            if thermo.get("pressure_reliable"):
                parts.append(f"- Production pressure: **{thermo['avg_pressure']:.1f} bar**")
            else:
                parts.append("- Production pressure: **unreliable** (do not interpret)")
        msd = result.get("msd", {})
        if msd.get("success"):
            parts.append(f"- Final MSD: **{msd['final_msd']:.4f} Å²**")
            if quality.get("converged"):
                spoken_parts.append(f"Final MSD {msd['final_msd']:.4f}.")
            else:
                spoken_parts.append(
                    "MSD values are shown but may be unreliable until the run equilibrates."
                )

        if not spoken_parts:
            spoken_parts = ["Analysis complete."]

        return VoiceResponse(
            spoken_text=" ".join(spoken_parts),
            display_text="\n".join(parts),
            action="analyze",
            success=True,
            data=result,
            model_used=model_id,
        )

    def _handle_predict(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance)
        material = self.agent._parse_simulation_instruction(utterance)["material"]
        props = re.findall(
            r"\b(elastic|conductivity|band\s*gap|thermal|modulus)\b",
            utterance.lower(),
        ) or ["general properties"]
        result = self.agent.predict_properties(material, props)
        spoken = (
            f"Property prediction for {material} is not yet available "
            "without a trained model."
            if not result.get("success")
            else f"Predictions ready for {material}."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text=result.get("error") or str(result),
            action="predict",
            success=result.get("success", False),
            data=result,
            model_used=model_id,
        )

    def _handle_research(self, utterance: str, model_id: str) -> VoiceResponse:
        from ..core.literature_search import research_task

        research = research_task(utterance)
        rec = self.agent.model_router.recommend(utterance, search_online=True)

        lines = [
            f"**Research: {utterance}**",
            f"Recommended model: **{rec.model_id}** ({rec.rationale})",
            "",
        ]
        if research["papers"]:
            lines.append("**Papers (arXiv):**")
            for p in research["papers"][:3]:
                lines.append(f"- [{p['title'][:80]}]({p['url']})")
        if research["web_results"]:
            lines.append("\n**Web sources:**")
            for r in research["web_results"][:3]:
                lines.append(f"- {r['title'][:80]}")

        spoken = (
            f"I researched your question and recommend {rec.model_id}. "
            f"I found {research['paper_count']} papers and "
            f"{research['web_count']} web sources."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text="\n".join(lines),
            action="research",
            success=True,
            data=research,
            model_used=rec.model_id,
        )

    def _handle_database(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance)
        result = self.agent.query_database(utterance)
        spoken = (
            "Database query complete."
            if result.get("success")
            else "Database query failed. Check your Materials Project API key."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text=str(result.get("results") or result.get("error")),
            action="database",
            success=result.get("success", False),
            data=result,
            model_used=model_id,
        )

    def _handle_status(self, model_id: str) -> VoiceResponse:
        n_tools = len(self.agent.tools)
        spoken = (
            f"MaterialSim online. {n_tools} tools available. "
            "I can run simulations, analyze results, and search literature."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text=(
                f"**Material Sim Agent — Status**\n\n"
                f"- Tools: {n_tools}\n"
                f"- Active model: `{model_id}`\n"
                f"- Voice: enabled\n"
                f"- MD engine: ASE (real physics)"
            ),
            action="status",
            success=True,
            model_used=model_id,
        )

    def _handle_chat(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance)
        reply = self.agent.chat(utterance)
        spoken = _voice_summary(reply)
        return VoiceResponse(
            spoken_text=spoken,
            display_text=reply,
            action="chat",
            success=True,
            model_used=model_id,
        )


def _voice_summary(text: str, max_chars: int = 500) -> str:
    """Shorten assistant text for natural spoken replies."""
    import re

    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"#+\s*", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"
