"""Assemble the NewsGenie LangGraph workflow.

Topology
--------
            ┌──────────────┐
            │   classify   │
            └──────┬───────┘
        intent ====╪====================
        news │     │ websearch    │ general
             ▼     ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────────┐
       │fetch_news│  │web_search│  │ general_chat │──► END
       └────┬─────┘  └────┬─────┘  └──────────────┘
   articles?│             │
     yes│ no│ (fallback)  │
        │   └────────────►│
        ▼                 ▼
     ┌────────────────────────┐
     │        generate        │──► END
     └────────────────────────┘

A `MemorySaver` checkpointer persists state per `thread_id`, which is how the
assistant keeps conversation context across turns within a session.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import nodes
from state import GraphState


# ── Conditional edge functions ─────────────────────────────────────────
def _route_by_intent(state: GraphState) -> str:
    """First fork: send the turn to the right data/answer node."""
    return state.get("intent", "general")


def _route_after_news(state: GraphState) -> str:
    """Second fork: if news came back empty, fall back to web search;
    otherwise synthesise the answer."""
    news = state.get("news") or {}
    if news.get("ok") and news.get("items"):
        return "generate"
    return "web_search"  # fallback path


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the workflow once, with an in-memory checkpointer."""
    g = StateGraph(GraphState)

    g.add_node("classify", nodes.classify)
    g.add_node("fetch_news", nodes.fetch_news)
    g.add_node("web_search", nodes.web_search)
    g.add_node("general_chat", nodes.general_chat)
    g.add_node("generate", nodes.generate)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route_by_intent,
        {
            "news": "fetch_news",
            "websearch": "web_search",
            "general": "general_chat",
        },
    )
    g.add_conditional_edges(
        "fetch_news",
        _route_after_news,
        {"generate": "generate", "web_search": "web_search"},
    )
    g.add_edge("web_search", "generate")
    g.add_edge("general_chat", END)
    g.add_edge("generate", END)

    return g.compile(checkpointer=MemorySaver())


def run_turn(query: str, category: str, thread_id: str) -> dict:
    """Run one user turn through the graph and return the final state.

    `thread_id` ties the turn to a conversation thread so the checkpointer can
    supply prior context. The returned dict includes `response` and `notices`.
    """
    from langchain_core.messages import HumanMessage

    graph = build_graph()
    config_arg = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "query": query,
        "category": category,
        "messages": [HumanMessage(content=query)],
        "notices": [],
    }
    final_state = graph.invoke(inputs, config=config_arg)
    return final_state
