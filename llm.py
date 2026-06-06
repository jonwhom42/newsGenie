"""Gemini LLM access for NewsGenie.

We drive Google Gemini through `langchain_google_genai.ChatGoogleGenerativeAI`
because the rest of the system is a LangGraph workflow, and LangGraph nodes
compose naturally with LangChain chat models. The model is created lazily and
cached so we only pay construction cost once per process.
"""
from __future__ import annotations

from functools import lru_cache

import config

_INIT_ERROR: str | None = None


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatGoogleGenerativeAI instance, or None if unavailable.

    Returns None when the API key is missing or the package can't be
    imported, so callers can fall back to a non-LLM path instead of crashing.
    """
    global _INIT_ERROR
    if not config.has_google_key():
        _INIT_ERROR = "GOOGLE_API_KEY is not set."
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            max_output_tokens=config.MAX_TOKENS,
            temperature=0.3,
            timeout=30,
            max_retries=2,
        )
    except Exception as exc:  # missing package, bad model id, etc.
        _INIT_ERROR = f"Could not initialise Gemini: {exc}"
        return None


def init_error() -> str | None:
    """Human-readable reason the LLM is unavailable (or None)."""
    return _INIT_ERROR


def llm_available() -> bool:
    return get_llm() is not None
