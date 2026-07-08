"""Intelligent LLM router: picks the best model per task using web + paper research."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .literature_search import research_task

logger = logging.getLogger(__name__)


@dataclass
class ModelRecommendation:
    """A recommended LLM for a specific task type."""

    model_id: str
    provider: str
    task_type: str
    rationale: str
    confidence: float
    sources: List[str]


# Curated fallbacks when online search is unavailable.
_TASK_DEFAULTS: Dict[str, ModelRecommendation] = {
    "simulation_planning": ModelRecommendation(
        model_id="gpt-4o",
        provider="openai",
        task_type="simulation_planning",
        rationale="Strong structured reasoning for MD parameter selection.",
        confidence=0.85,
        sources=["built-in benchmark table"],
    ),
    "analysis_interpretation": ModelRecommendation(
        model_id="gpt-4o-mini",
        provider="openai",
        task_type="analysis_interpretation",
        rationale="Fast, cost-effective interpretation of RDF/MSD/thermo data.",
        confidence=0.80,
        sources=["built-in benchmark table"],
    ),
    "literature_review": ModelRecommendation(
        model_id="gpt-4o",
        provider="openai",
        task_type="literature_review",
        rationale="Long-context synthesis of papers and web sources.",
        confidence=0.82,
        sources=["built-in benchmark table"],
    ),
    "property_prediction": ModelRecommendation(
        model_id="gpt-4o",
        provider="openai",
        task_type="property_prediction",
        rationale="Reliable numeric reasoning for property estimation.",
        confidence=0.78,
        sources=["built-in benchmark table"],
    ),
    "voice_conversation": ModelRecommendation(
        model_id="gpt-4o-mini",
        provider="openai",
        task_type="voice_conversation",
        rationale="Low latency for real-time Jarvis-style dialogue.",
        confidence=0.88,
        sources=["built-in benchmark table"],
    ),
    "general_chat": ModelRecommendation(
        model_id="gpt-4o-mini",
        provider="openai",
        task_type="general_chat",
        rationale="Balanced speed and quality for general Q&A.",
        confidence=0.75,
        sources=["built-in benchmark table"],
    ),
}

# Keywords used to classify a user utterance into a task type.
_TASK_KEYWORDS: Dict[str, List[str]] = {
    "simulation_planning": [
        "simulate", "simulation", "molecular dynamics", "md", "lammps", "run",
        "nvt", "npt", "nve", "ensemble", "timestep",
    ],
    "analysis_interpretation": [
        "analyze", "analysis", "rdf", "msd", "radial", "displacement",
        "thermodynamic", "plot", "result", "interpret",
    ],
    "literature_review": [
        "paper", "literature", "arxiv", "research", "cite", "publication",
        "review", "benchmark", "state of the art",
    ],
    "property_prediction": [
        "predict", "prediction", "property", "elastic", "conductivity",
        "band gap", "ml", "machine learning",
    ],
    "voice_conversation": [
        "jarvis", "hey", "hello", "status", "what can you",
    ],
}


class ModelRouter:
    """Select the best LLM for a given materials-science task."""

    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self._cache: Dict[str, ModelRecommendation] = {}

    def classify_task(self, text: str) -> str:
        """Classify *text* into a task type using keyword matching."""
        lower = text.lower()
        scores: Dict[str, int] = {}
        for task_type, keywords in _TASK_KEYWORDS.items():
            scores[task_type] = sum(1 for kw in keywords if kw in lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general_chat"

    def recommend(
        self,
        task_description: str,
        *,
        search_online: bool = True,
    ) -> ModelRecommendation:
        """Return the best model for *task_description*, optionally researching online."""
        task_type = self.classify_task(task_description)
        cache_key = f"{task_type}:{task_description[:80]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        rec = _TASK_DEFAULTS.get(task_type, _TASK_DEFAULTS["general_chat"])
        sources = list(rec.sources)

        if search_online:
            research = research_task(task_description)
            sources.extend(p["title"] for p in research.get("papers", [])[:2])
            sources.extend(r["title"] for r in research.get("web_results", [])[:2])

            upgraded = self._upgrade_from_research(task_type, research)
            if upgraded:
                rec = upgraded
                rec.sources = sources

        self._cache[cache_key] = rec
        return rec

    @staticmethod
    def _upgrade_from_research(
        task_type: str, research: Dict[str, Any]
    ) -> Optional[ModelRecommendation]:
        """Heuristically upgrade model choice based on web/paper snippets."""
        snippets = " ".join(
            r.get("snippet", "") + r.get("title", "")
            for r in research.get("web_results", [])
        ).lower()
        paper_text = " ".join(
            p.get("summary", "") + p.get("title", "")
            for p in research.get("papers", [])
        ).lower()
        combined = snippets + " " + paper_text

        if "gpt-4o" in combined and task_type in {
            "literature_review", "simulation_planning", "property_prediction"
        }:
            return ModelRecommendation(
                model_id="gpt-4o",
                provider="openai",
                task_type=task_type,
                rationale="Online benchmarks favour gpt-4o for this task category.",
                confidence=0.90,
                sources=[],
            )
        if "claude" in combined and "reasoning" in combined:
            return ModelRecommendation(
                model_id="claude-sonnet-4-20250514",
                provider="anthropic",
                task_type=task_type,
                rationale="Recent benchmarks highlight Claude for scientific reasoning.",
                confidence=0.85,
                sources=[],
            )
        if task_type == "voice_conversation" or "low latency" in combined:
            return ModelRecommendation(
                model_id="gpt-4o-mini",
                provider="openai",
                task_type=task_type,
                rationale="Low-latency model selected for voice interaction.",
                confidence=0.88,
                sources=[],
            )
        return None

    def get_model_id(self, task_description: str, *, search_online: bool = True) -> str:
        """Convenience: return just the model ID string."""
        return self.recommend(task_description, search_online=search_online).model_id
