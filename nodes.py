"""LangGraph node functions for NewsGenie.

The graph is: classify -> (news | websearch | general) -> generate -> END,
with news falling back to websearch when no articles are found. Each function
takes the current `GraphState` and returns a partial state update.

Design goals:
* Query differentiation lives in `classify` (LLM with a deterministic
  keyword fallback).
* Every node is defensive — if the LLM or an API is unavailable, the node
  produces a useful response instead of raising.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
import tools
from llm import get_llm

# ── Keyword fallback for classification (used when the LLM is unavailable) ──
_NEWS_HINTS = re.compile(
    r"\b(news|headline|headlines|breaking|latest|update|updates|today|"
    r"happening|story|stories|report)\b", re.I)
_SEARCH_HINTS = re.compile(
    r"\b(who|what|when|where|how|why|price|stock|score|result|weather|"
    r"current|now|2024|2025|2026|define|explain)\b", re.I)


# ── 1. Classify intent ─────────────────────────────────────────────────
def classify(state: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the turn is a news request, a web-search question, or a
    general/conversational query — and extract the topic to look up."""
    query = (state.get("query") or "").strip()
    category = state.get("category") or config.DEFAULT_CATEGORY
    llm = get_llm()

    if llm is None:
        return _keyword_classify(query)

    system = (
        "You are the router for NewsGenie, a news + information assistant. "
        "Classify the user's message into exactly one intent and extract the "
        "topic to look up.\n\n"
        "Intents:\n"
        "- news: the user wants current news/headlines (e.g. 'latest tech "
        "news', 'what's happening in sports'). Use this for requests for "
        "recent developments in a domain.\n"
        "- websearch: a factual question whose answer needs up-to-date or "
        "external info (prices, scores, definitions, 'who is...', 'how to...').\n"
        "- general: greetings, chit-chat, opinions, or anything answerable "
        "from general knowledge without fresh data.\n\n"
        "Respond with ONLY a JSON object: "
        '{"intent": "news|websearch|general", "topic": "<short search topic '
        'or empty string>"}'
    )
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=query),
        ])
        data = _extract_json(resp.content)
        intent = data.get("intent", "general")
        if intent not in ("news", "websearch", "general"):
            intent = "general"
        topic = (data.get("topic") or "").strip()
        return {"intent": intent, "topic": topic}
    except Exception:
        # Any LLM/parse failure -> deterministic fallback, never crash.
        return _keyword_classify(query)


def _keyword_classify(query: str) -> dict[str, Any]:
    if _NEWS_HINTS.search(query):
        return {"intent": "news", "topic": _strip_news_words(query)}
    if _SEARCH_HINTS.search(query) or "?" in query:
        return {"intent": "websearch", "topic": query}
    return {"intent": "general", "topic": ""}


def _strip_news_words(query: str) -> str:
    topic = _NEWS_HINTS.sub("", query)
    return re.sub(r"\s+", " ", topic).strip()


# ── 2. Fetch news ──────────────────────────────────────────────────────
def fetch_news(state: dict[str, Any]) -> dict[str, Any]:
    """Pull live news for the selected category, or a topic if one was found."""
    topic = (state.get("topic") or "").strip()
    category_label = state.get("category") or config.DEFAULT_CATEGORY
    category = config.CATEGORY_MAP.get(category_label, "general")

    # A specific topic -> targeted search; otherwise category headlines.
    result = tools.fetch_news(category=category, query=topic or None)

    notices: list[str] = []
    if not result["ok"]:
        notices.append(result.get("message", "News fetch failed."))
    return {"news": result, "notices": notices}


# ── 3. Web search ──────────────────────────────────────────────────────
def web_search(state: dict[str, Any]) -> dict[str, Any]:
    """Search the web. Reached directly for factual questions, or as the
    fallback when news came back empty."""
    topic = (state.get("topic") or state.get("query") or "").strip()
    used_fallback = bool(state.get("news"))  # we only have news if we tried it
    result = tools.web_search(topic)

    notices = list(state.get("notices", []))
    if used_fallback:
        notices.append("No live news found — answering from a web search.")
    if not result["ok"]:
        notices.append(result.get("message", "Web search failed."))
    return {"search": result, "used_fallback": used_fallback,
            "notices": notices}


# ── 4. General chat ────────────────────────────────────────────────────
def general_chat(state: dict[str, Any]) -> dict[str, Any]:
    """Answer a conversational/general-knowledge query directly with Gemini,
    using the running message history for context."""
    query = state.get("query") or ""
    llm = get_llm()
    if llm is None:
        return {"response": _no_llm_message(),
                "messages": [AIMessage(content=_no_llm_message())]}

    system = SystemMessage(content=(
        "You are NewsGenie, a friendly and concise news & information "
        "assistant. Answer the user's question helpfully. If the question "
        "needs very recent facts you don't have, say so and suggest they ask "
        "for the latest news on the topic. Keep answers focused."
    ))
    # Include prior turns for conversational context.
    history = state.get("messages", [])
    try:
        resp = llm.invoke([system, *history, HumanMessage(content=query)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"response": text, "messages": [AIMessage(content=text)]}
    except Exception as exc:
        notice = _llm_error_notice(exc)
        text = ("I couldn't generate an answer just now. " + notice)
        return {"response": text, "messages": [AIMessage(content=text)],
                "notices": list(state.get("notices", [])) + [notice]}


# ── 5. Generate (synthesise news / search results) ─────────────────────
def generate(state: dict[str, Any]) -> dict[str, Any]:
    """Compose the final answer from whatever data we gathered, with sources.

    Works for both the news path and the web-search path. Falls back to a
    plain formatted list if the LLM is unavailable.
    """
    query = state.get("query") or ""
    news = state.get("news") or {}
    search = state.get("search") or {}

    news_items = news.get("items", []) if news.get("ok") else []
    search_items = search.get("items", []) if search.get("ok") else []

    # Nothing at all came back -> honest, helpful dead-end message.
    if not news_items and not search_items:
        msg = _nothing_found_message(state)
        return {"response": msg, "messages": [AIMessage(content=msg)]}

    context = _format_context(news_items, search_items)
    llm = get_llm()

    if llm is None:
        # No LLM: present the curated results directly.
        text = _format_plain(query, news_items, search_items)
        return {"response": text, "messages": [AIMessage(content=text)]}

    system = SystemMessage(content=(
        "You are NewsGenie. Summarise the retrieved items below into a clear, "
        "trustworthy answer for the user's request. Rules:\n"
        "- Base your answer ONLY on the provided items; do not invent facts.\n"
        "- Lead with a one-line summary, then 3-6 concise bullet points.\n"
        "- Attribute claims to their source name and keep a neutral tone "
        "(this helps the user judge reliability).\n"
        "- End with a short 'Sources' list of the article/site titles.\n"
        "- If items look thin or potentially unreliable, say so honestly."
    ))
    human = HumanMessage(content=(
        f"User request: {query}\n\nRetrieved items:\n{context}"
    ))
    try:
        resp = llm.invoke([system, human])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"response": text, "messages": [AIMessage(content=text)]}
    except Exception as exc:
        # LLM failed (e.g. no credits / bad key) — show curated raw results
        # and tell the user why the AI summary is missing.
        text = _format_plain(query, news_items, search_items)
        return {"response": text, "messages": [AIMessage(content=text)],
                "notices": list(state.get("notices", [])) + [_llm_error_notice(exc)]}


# ── Formatting helpers ─────────────────────────────────────────────────
def _format_context(news_items: list, search_items: list) -> str:
    lines: list[str] = []
    if news_items:
        lines.append("=== NEWS ARTICLES ===")
        for i, a in enumerate(news_items, 1):
            lines.append(
                f"{i}. {a['title']} — {a['source']} ({a['published_at']})\n"
                f"   {a['description']}\n   {a['url']}")
    if search_items:
        lines.append("=== WEB RESULTS ===")
        for i, s in enumerate(search_items, 1):
            lines.append(f"{i}. {s['title']}\n   {s['snippet']}\n   {s['url']}")
    return "\n".join(lines)


def _format_plain(query: str, news_items: list, search_items: list) -> str:
    """Used when the LLM is unavailable — present the raw sources cleanly.

    External text (titles, descriptions, snippets) is markdown-escaped and
    whitespace-collapsed so stray characters like a leading '#' in a snippet
    can't hijack rendering (e.g. turning into a huge heading).
    """
    out = [f"Here is what I found for **{_md_escape(query) or 'your request'}**:\n"]
    for a in news_items:
        out.append(
            f"- **{_md_escape(a['title'])}** — {_md_escape(a['source'])} "
            f"({a['published_at']})  \n"
            f"  {_md_escape(a['description'])}  \n"
            f"  <{a['url']}>")
    for s in search_items:
        out.append(
            f"- **{_md_escape(s['title'])}**  \n"
            f"  {_md_escape(s['snippet'])}  \n"
            f"  <{s['url']}>")
    out.append("\n_(Showing raw results — an AI summary couldn't be generated.)_")
    return "\n".join(out)


# Characters that have structural meaning in Markdown and must be neutralised
# when we render text that came from an external source.
_MD_SPECIAL = ("\\", "`", "*", "_", "{", "}", "[", "]", "<", ">", "#", "|")


def _md_escape(text: Any) -> str:
    """Collapse whitespace and escape Markdown specials in untrusted text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    for ch in _MD_SPECIAL:  # backslash is first, so it's escaped before the rest
        text = text.replace(ch, "\\" + ch)
    return text


def _llm_error_notice(exc: Exception) -> str:
    """Turn an LLM exception into a clear, actionable user-facing message."""
    s = str(exc).lower()
    if "api_key_invalid" in s or "api key not valid" in s or "invalid" in s and "key" in s:
        return ("AI summary unavailable: the GOOGLE_API_KEY is invalid — get a "
                "free key at aistudio.google.com/apikey and set it in your .env.")
    if "permission" in s or "403" in s:
        return ("AI summary unavailable: the Gemini API isn't enabled for this "
                "key — enable it in Google AI Studio.")
    if "quota" in s or "resource_exhausted" in s or "429" in s or "rate" in s:
        return ("AI summary unavailable: hit the Gemini free-tier rate limit — "
                "wait a moment and try again.")
    if "not found" in s or "404" in s:
        return (f"AI summary unavailable: model '{config.GEMINI_MODEL}' wasn't "
                "found — try GEMINI_MODEL=gemini-2.0-flash in your .env.")
    return f"AI summary unavailable ({type(exc).__name__})."


def _nothing_found_message(state: dict[str, Any]) -> str:
    parts = ["I couldn't find anything relevant for that right now."]
    for n in state.get("notices", []):
        parts.append(f"- {n}")
    parts.append("Try rephrasing, picking a different category, or asking for "
                 "the latest news on a broader topic.")
    return "\n".join(parts)


def _no_llm_message() -> str:
    return ("Gemini isn't configured (missing or invalid GOOGLE_API_KEY), "
            "so I can't answer general questions. You can still browse live "
            "news by selecting a category and asking for the latest headlines.")


def _extract_json(content: Any) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response, defensively."""
    text = content if isinstance(content, str) else str(content)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}
