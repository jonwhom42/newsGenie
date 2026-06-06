"""The shared state object that flows through the LangGraph workflow.

Each node reads from and writes to this typed dict. `messages` uses LangGraph's
`add_messages` reducer so conversation turns accumulate across the session
(this is what gives NewsGenie its memory / context).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["news", "websearch", "general"]


class GraphState(TypedDict, total=False):
    # Running conversation history (persisted per session via the checkpointer).
    messages: Annotated[list, add_messages]

    # The current user turn.
    query: str
    # News category selected in the UI sidebar (friendly label, e.g. "Finance").
    category: str

    # Filled in by the classifier node.
    intent: Intent
    topic: str            # the specific subject to search for, if any

    # Filled in by the data nodes.
    news: dict[str, Any]      # uniform tool dict from tools.fetch_news
    search: dict[str, Any]    # uniform tool dict from tools.web_search
    used_fallback: bool       # True if news was empty and we fell back to search

    # The final assistant answer + any user-facing notices.
    response: str
    notices: list[str]
