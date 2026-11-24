"""The research agent as an explicit LangGraph loop.

    START -> agent --(tool calls)--> tools --> agent ... --(no tool calls)--> finalize -> END

- agent: the chat model plans and either calls tools or writes the answer. After
  max_tool_rounds the tools are taken away, so the loop always ends.
- tools: runs list_articles / search_articles and counts the round.
- finalize: pulls out the answer, the [A#] citations, the sources, the agent steps and
  token usage.
"""

from __future__ import annotations

import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from news_research.agent.prompts import FINAL_ROUND_NOTE, RESEARCH_AGENT_SYSTEM_PROMPT
from news_research.vector_store import Manifest

CITATION = re.compile(r"\[(A\d+)\]")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    result: dict[str, Any]


def build_research_graph(llm: BaseChatModel, tools: list[BaseTool], manifest: Manifest, max_tool_rounds: int):
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        system = [SystemMessage(RESEARCH_AGENT_SYSTEM_PROMPT)]
        if rounds >= max_tool_rounds:
            system.append(SystemMessage(FINAL_ROUND_NOTE))
            model = llm  # no tools: the model must answer now
        else:
            model = llm_with_tools
        response = model.invoke(system + state["messages"])
        return {"messages": [response]}

    def tools(state: AgentState) -> dict:
        update = tool_node.invoke(state)
        return {"messages": update["messages"], "tool_rounds": state.get("tool_rounds", 0) + 1}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"

    def finalize(state: AgentState) -> dict:
        return {"result": summarize_run(state["messages"], manifest)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "finalize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def summarize_run(messages: list[AnyMessage], manifest: Manifest) -> dict[str, Any]:
    """Answer, cited sources, retrieved passages, steps and token usage from one run."""
    answer = messages[-1].text if messages else ""
    articles = {a.article_id: a for a in manifest.articles}

    cited_ids = list(dict.fromkeys(CITATION.findall(answer)))
    unknown_citations = [cid for cid in cited_ids if cid not in articles]
    sources = [
        {"article_id": cid, "title": articles[cid].title, "source": articles[cid].source}
        for cid in cited_ids
        if cid in articles
    ]

    steps: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tool_results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for key in usage:
            usage[key] += (message.usage_metadata or {}).get(key, 0)
        for call in message.tool_calls:
            result = tool_results.get(call["id"])
            artifact = (result.artifact or []) if result is not None else []
            retrieved.extend(artifact)
            steps.append(
                {
                    "tool": call["name"],
                    "args": call["args"],
                    "results": len(artifact) if call["name"] == "search_articles" else None,
                    "status": getattr(result, "status", "success") if result else "missing",
                }
            )

    return {
        "answer": answer,
        "sources": sources,
        "unknown_citations": unknown_citations,
        "retrieved": retrieved,
        "steps": steps,
        "usage": usage,
        "grounded": bool(sources),
    }


def timed_invoke(graph, state: AgentState) -> tuple[AgentState, float]:
    started = time.perf_counter()
    final_state = graph.invoke(state)
    return final_state, round(time.perf_counter() - started, 3)
