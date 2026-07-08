"""Voice command orchestrator — PI-style spoken dialogue for hands-free control."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .wake import contains_wake_phrase, greeting_reply, strip_wake_phrase

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
    defer_action: bool = False  # speak first, execute on playback_finished


class VoiceOrchestrator:
    """Interpret spoken commands and drive the materials workflow like a PI would."""

    INTENT_PATTERNS: Dict[str, List[str]] = {
        "run_simulation": [
            r"\b(simulate|simulation|run\s+(a\s+)?md|molecular\s+dynamics)\b",
            r"\brun\s+(a\s+)?simulation\b",
            r"\b(do|start|launch|kick\s+off)\b.*\b(run|simulation|md)\b",
            r"\brun\b.*\b(at|for|with)\b",
        ],
        "analyze": [
            r"\b(analyz|analysis|analyse|interpret)\b",
            r"\b(how\s+did|how\s+does|tell\s+me\s+about)\b.*\b(result|run|simulation)\b",
            r"\b(rdf|msd|thermodynamic|results?)\b",
        ],
        "predict": [
            r"\b(predict|prediction|forecast)\b",
        ],
        "research": [
            r"\b(research|paper|literature|arxiv|best\s+model|benchmark)\b",
        ],
        "database": [
            r"\b(database|materials\s+project|query|lookup)\b",
        ],
        "status": [
            r"^(status|ready|online|hello|hi)\b",
            r"^(what\s+can\s+you\s+do|who\s+are\s+you)\b",
        ],
    }

    def __init__(self, agent: "MaterialsAgent"):
        self.agent = agent

    def process(self, utterance: str, *, raw: str = "") -> VoiceResponse:
        """Process a voice utterance end-to-end."""
        raw = (raw or utterance).strip()
        utterance = strip_wake_phrase(utterance or raw).strip()

        if not utterance and contains_wake_phrase(raw):
            return VoiceResponse(
                spoken_text=greeting_reply(),
                display_text=greeting_reply(),
                action="greeting",
                success=True,
            )

        if not utterance:
            return VoiceResponse(
                spoken_text="I didn't catch that. Try saying hey Material Sim, then your request.",
                display_text="No command heard.",
                action="none",
                success=False,
            )

        intent = self._detect_intent(utterance)
        router = self.agent.model_router
        rec = router.recommend(utterance, search_online=(intent == "research"))
        model_id = rec.model_id

        if intent == "run_simulation":
            return self._plan_simulation(utterance, model_id)
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

    def execute_simulation(self, utterance: str) -> VoiceResponse:
        """Run a queued simulation and return a spoken PI-style debrief."""
        self.agent.reconfigure_llm(utterance, search_online=False)
        result = self.agent.run_simulation(utterance)
        return self._simulation_debrief(result, model_id=None)

    def _detect_intent(self, text: str) -> str:
        lower = text.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(p, lower) for p in patterns):
                return intent
        # Natural imperatives without keywords: "copper at 300 K, ten thousand steps"
        if re.search(r"\b(at\s+\d+\s*k|\d+\s*steps?|nvt|npt|nve)\b", lower):
            return "run_simulation"
        return "chat"

    def _plan_simulation(self, utterance: str, model_id: str) -> VoiceResponse:
        """Acknowledge like a PI, then defer the heavy run until after TTS."""
        try:
            from ..spec.extractor import extract_spec

            spec = extract_spec(utterance, self.agent.config)
            summary = spec.summary()
        except Exception:
            summary = utterance[:120]

        spoken = (
            f"Got it. {self._speakable_summary(summary)} "
            "I'll set that up and run it now — give me a moment."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text=f"**Starting simulation**\n\n{summary}",
            action="run_simulation",
            success=True,
            data={"instruction": utterance, "summary": summary},
            model_used=model_id,
            defer_action=True,
        )

    def _simulation_debrief(self, result: Dict[str, Any], model_id: Optional[str]) -> VoiceResponse:
        if result.get("needs_clarification"):
            err = result.get("error") or "I need a bit more detail."
            return VoiceResponse(
                spoken_text=f"I couldn't start yet. {err}",
                display_text=f"**Need clarification**\n\n{err}",
                action="run_simulation",
                success=False,
                data=result,
                model_used=model_id,
            )

        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return VoiceResponse(
                spoken_text=f"The run failed. {err}",
                display_text=f"Simulation failed: {err}",
                action="run_simulation",
                success=False,
                data=result,
                model_used=model_id,
            )

        quality = result.get("quality") or {}
        spoken_parts = ["Alright, the run finished."]
        display = f"**Simulation complete**\n\n{result.get('message', '')}"

        if quality.get("success"):
            if quality.get("converged"):
                spoken_parts.append("Thermodynamically, it looks equilibrated.")
            else:
                spoken_parts.append(
                    "Heads up — it didn't fully equilibrate, so treat the numbers cautiously."
                )
            if quality.get("avg_temperature") is not None:
                t = quality["avg_temperature"]
                spoken_parts.append(f"Production temperature landed near {t:.0f} kelvin.")

        sim_dir = result.get("simulation_directory")
        if sim_dir:
            from pathlib import Path as _Path

            spoken_parts.append(f"Outputs are saved under {_Path(sim_dir).name}.")
            display += f"\n\nDirectory: `{sim_dir}`"

        spoken_parts.append("Want me to walk through the RDF, MSD, or energy trace?")

        for w in result.get("warnings", []):
            display += f"\n- ⚠️ {w}"

        return VoiceResponse(
            spoken_text=" ".join(spoken_parts),
            display_text=display,
            action="run_simulation",
            success=True,
            data=result,
            model_used=model_id,
        )

    def _handle_analysis(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance, search_online=False)
        from ..analysis_engine import find_simulation_dir

        sim_dir = find_simulation_dir()
        if sim_dir is None:
            return VoiceResponse(
                spoken_text="I don't see a finished run yet. Tell me what to simulate first.",
                display_text="No simulation directories found.",
                action="analyze",
                success=False,
                model_used=model_id,
            )

        result = self.agent.analyze_results(str(sim_dir))
        if not result.get("success"):
            return VoiceResponse(
                spoken_text="I couldn't analyze those results.",
                display_text=f"Analysis failed: {result.get('error')}",
                action="analyze",
                success=False,
                model_used=model_id,
            )

        spoken_parts = [f"Here's what I'm seeing in {sim_dir.name}."]
        parts = [f"**Analysis — {sim_dir.name}**"]

        quality = result.get("quality", {})
        if quality.get("success") and not quality.get("converged"):
            spoken_parts.append("The run didn't equilibrate cleanly, so interpret with care.")

        rdf = result.get("rdf", {})
        if rdf.get("success") and rdf.get("first_peak"):
            parts.append(f"- RDF first peak: **{rdf['first_peak']:.2f} Å**")
            spoken_parts.append(
                f"Nearest-neighbor shell shows up around {rdf['first_peak']:.2f} angstroms."
            )

        thermo = result.get("thermodynamics", {})
        if thermo.get("success"):
            parts.append(
                f"- Production T: **{thermo['avg_temperature']:.1f} ± "
                f"{thermo['std_temperature']:.1f} K**"
            )
            spoken_parts.append(
                f"Average production temperature is {thermo['avg_temperature']:.0f} kelvin."
            )

        msd = result.get("msd", {})
        if msd.get("success"):
            parts.append(f"- Final MSD: **{msd['final_msd']:.4f} Å²**")
            if quality.get("converged"):
                spoken_parts.append(f"Mean squared displacement ends at {msd['final_msd']:.4f}.")

        if len(spoken_parts) == 1:
            spoken_parts.append("Analysis finished — check the workspace plots for detail.")

        return VoiceResponse(
            spoken_text=" ".join(spoken_parts),
            display_text="\n".join(parts),
            action="analyze",
            success=True,
            data=result,
            model_used=model_id,
        )

    def _handle_predict(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance, search_online=False)
        material = self.agent._parse_simulation_instruction(utterance)["material"]
        props = re.findall(
            r"\b(elastic|conductivity|band\s*gap|thermal|modulus)\b",
            utterance.lower(),
        ) or ["general properties"]
        result = self.agent.predict_properties(material, props)
        spoken = (
            f"I don't have a trained model for {material} yet."
            if not result.get("success")
            else f"Predictions for {material} are ready."
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
        spoken = (
            f"I looked that up — found {research['paper_count']} papers and "
            f"{research['web_count']} web sources. For this task I'd reach for {rec.model_id}."
        )
        lines = [f"**Research: {utterance}**", f"Model: **{rec.model_id}**"]
        return VoiceResponse(
            spoken_text=spoken,
            display_text="\n".join(lines),
            action="research",
            success=True,
            data=research,
            model_used=rec.model_id,
        )

    def _handle_database(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance, search_online=False)
        result = self.agent.query_database(utterance)
        spoken = (
            "Database query complete."
            if result.get("success")
            else "Database lookup failed — check your Materials Project key in Settings."
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
            f"Material Sim is online with {n_tools} tools. "
            "Say hey Material Sim, then tell me what to run or analyze."
        )
        return VoiceResponse(
            spoken_text=spoken,
            display_text=f"**Online** — {n_tools} tools, model `{model_id}`",
            action="status",
            success=True,
            model_used=model_id,
        )

    def _handle_chat(self, utterance: str, model_id: str) -> VoiceResponse:
        self.agent.reconfigure_llm(utterance, search_online=False)
        reply = self.agent.chat(utterance)
        spoken = _voice_summary(reply)
        return VoiceResponse(
            spoken_text=spoken,
            display_text=reply,
            action="chat",
            success=True,
            model_used=model_id,
        )

    @staticmethod
    def _speakable_summary(summary: str) -> str:
        s = summary.replace("protocol=", "using ")
        s = s.replace("potential=", "with ")
        s = s.replace("engine=", "on ")
        s = s.replace("dt=", "timestep ")
        s = s.replace("steps", " steps")
        s = s.replace("@", " at ")
        s = s.replace("K", " kelvin")
        s = s.replace(",", ", ")
        return s


def _voice_summary(text: str, max_chars: int = 480) -> str:
    """Shorten assistant text for natural spoken replies."""
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
