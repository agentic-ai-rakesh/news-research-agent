"""Entry point for one question: validate, load the index, run the graph, return a result."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from news_research.agent.graph import build_research_graph, timed_invoke
from news_research.agent.prompts import PROMPT_VERSION
from news_research.agent.tools import make_tools
from news_research.config import Settings
from news_research.vector_store import load_index

logger = logging.getLogger(__name__)


@dataclass
class AgentAnswer:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    grounded: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0
    model: str = ""
    prompt_version: str = PROMPT_VERSION


def validate_question(question: str, settings: Settings) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    if len(question) > settings.max_question_chars:
        raise ValueError(f"The question is too long (max {settings.max_question_chars} characters).")
    return question


def ask(question: str, llm: BaseChatModel, embeddings: Embeddings, settings: Settings) -> AgentAnswer:
    """Answer one question with the research agent. Raises IndexNotReadyError / ValueError."""
    question = validate_question(question, settings)
    store, manifest = load_index(embeddings, settings)
    tools = make_tools(store, manifest, settings.top_k)
    graph = build_research_graph(llm, tools, manifest, settings.max_tool_rounds)
    final_state, latency = timed_invoke(graph, {"messages": [HumanMessage(question)], "tool_rounds": 0})
    result = final_state["result"]
    logger.info(
        "Answered in %.2fs, %d steps, sources=%s, tokens=%s",
        latency,
        len(result["steps"]),
        [s["article_id"] for s in result["sources"]],
        result["usage"].get("total_tokens"),
    )
    return AgentAnswer(
        question=question,
        latency_seconds=latency,
        model=getattr(llm, "model_name", "") or getattr(llm, "model", "") or type(llm).__name__,
        **result,
    )
