"""External data tools: real-time news (NewsAPI) and web search (DuckDuckGo).

Every function returns a small, uniform dict so the LangGraph nodes never have
to reason about HTTP details or library quirks:

    {"ok": bool, "items": [...], "error": str | None, "source": str}

`items` is a list of normalised records:
    news:   {title, source, url, published_at, description}
    search: {title, url, snippet}

Both tools are written to *never raise* — a missing key, a network blip, or an
empty result set comes back as a structured response the caller can branch on.
That is the backbone of NewsGenie's fallback strategy.
"""
from __future__ import annotations

from typing import Any

import requests

import config

NEWSAPI_TOP_HEADLINES = "https://newsapi.org/v2/top-headlines"
NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"
HTTP_TIMEOUT = 10  # seconds


# ── NewsAPI ────────────────────────────────────────────────────────────
def fetch_news(
    category: str | None = None,
    query: str | None = None,
    page_size: int = config.NEWS_PAGE_SIZE,
) -> dict[str, Any]:
    """Fetch real-time news from NewsAPI.

    - If `query` is given, search *everything* (topic-specific news).
    - Otherwise pull *top-headlines* for the given `category`.
    Returns the uniform tool dict; `ok=False` with an `error` code on any
    failure (missing_key, http_error, network_error, no_results).
    """
    if not config.has_newsapi_key():
        return _err("missing_key", "news",
                    "NEWSAPI_KEY is not set — skipping live news.")

    try:
        if query:
            url = NEWSAPI_EVERYTHING
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": config.NEWSAPI_KEY,
            }
        else:
            url = NEWSAPI_TOP_HEADLINES
            params = {
                "category": category or "general",
                "country": config.NEWS_COUNTRY,
                "pageSize": page_size,
                "apiKey": config.NEWSAPI_KEY,
            }

        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)

        # NewsAPI signals problems both via HTTP status and a JSON status field.
        if resp.status_code != 200:
            detail = _safe_json(resp).get("message", resp.text[:200])
            return _err("http_error", "news",
                        f"NewsAPI returned {resp.status_code}: {detail}")

        data = resp.json()
        if data.get("status") != "ok":
            return _err("api_error", "news",
                        data.get("message", "NewsAPI reported an error."))

        articles = [_normalise_article(a) for a in data.get("articles", [])]
        articles = [a for a in articles if a["title"]]  # drop [Removed] etc.

        if not articles:
            return _err("no_results", "news",
                        "No matching news articles were found.")

        return {"ok": True, "items": articles, "error": None, "source": "news"}

    except requests.exceptions.Timeout:
        return _err("network_error", "news", "NewsAPI request timed out.")
    except requests.exceptions.RequestException as exc:
        return _err("network_error", "news", f"Network error: {exc}")
    except Exception as exc:  # last-resort guard
        return _err("unknown_error", "news", f"Unexpected error: {exc}")


def _normalise_article(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": (a.get("title") or "").strip(),
        "source": ((a.get("source") or {}).get("name") or "Unknown").strip(),
        "url": a.get("url") or "",
        "published_at": (a.get("publishedAt") or "")[:10],  # YYYY-MM-DD
        "description": (a.get("description") or "").strip(),
    }


# ── DuckDuckGo web search ──────────────────────────────────────────────
def web_search(query: str,
               max_results: int = config.WEB_SEARCH_RESULTS) -> dict[str, Any]:
    """Search the web via DuckDuckGo (no API key required).

    Handles the package rename from `duckduckgo-search` to `ddgs` and never
    raises — returns the uniform tool dict instead.
    """
    DDGS = _import_ddgs()
    if DDGS is None:
        return _err("missing_dependency", "search",
                    "DuckDuckGo search library is not installed "
                    "(pip install ddgs).")

    try:
        results: list[dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": (r.get("title") or "").strip(),
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": (r.get("body") or "").strip(),
                })

        if not results:
            return _err("no_results", "search",
                        "Web search returned no results.")

        return {"ok": True, "items": results, "error": None, "source": "search"}

    except Exception as exc:  # the lib raises a variety of network errors
        return _err("network_error", "search", f"Web search failed: {exc}")


def _import_ddgs():
    """Return the DDGS class from whichever package is installed, or None."""
    try:
        from ddgs import DDGS  # current package name
        return DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS  # legacy package name
            return DDGS
        except Exception:
            return None


# ── helpers ────────────────────────────────────────────────────────────
def _err(code: str, source: str, message: str) -> dict[str, Any]:
    return {"ok": False, "items": [], "error": code,
            "source": source, "message": message}


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {}
