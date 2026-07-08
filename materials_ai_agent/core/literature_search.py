"""Search scientific literature (arXiv) and the open web for model/task research."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search arXiv for papers related to *query*."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_API, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("arXiv search failed: %s", exc)
        return []

    root = ET.fromstring(resp.text)
    papers: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        summary = re.sub(r"\s+", " ", summary)[:400]
        link = ""
        for link_el in entry.findall("atom:link", ATOM_NS):
            if link_el.get("title") == "pdf":
                link = link_el.get("href", "")
                break
        if not link:
            link = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        papers.append({"title": title, "summary": summary, "url": link})
    return papers


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the open web via DuckDuckGo (no API key required)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", "")[:300],
                        "url": item.get("href", ""),
                    }
                )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return []


def research_task(
    task_description: str,
    *,
    include_papers: bool = True,
    include_web: bool = True,
) -> Dict[str, Any]:
    """Gather literature and web context for a materials-science task."""
    papers: List[Dict[str, str]] = []
    web: List[Dict[str, str]] = []

    if include_papers:
        papers = search_arxiv(
            f"molecular dynamics OR materials informatics {task_description}",
            max_results=4,
        )
    if include_web:
        web = search_web(
            f"best LLM model {task_description} materials science 2024 2025",
            max_results=4,
        )

    return {
        "task": task_description,
        "papers": papers,
        "web_results": web,
        "paper_count": len(papers),
        "web_count": len(web),
    }
