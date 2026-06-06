"""Central configuration for NewsGenie.

Loads environment variables (from a `.env` file if present) and exposes
the settings the rest of the app relies on. Keeping this in one place makes
the fallback / error-handling story explicit: every consumer can ask
`config.has_google_key()` / `config.has_newsapi_key()` and degrade
gracefully instead of crashing.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at runtime
    pass


# ── Credentials ────────────────────────────────────────────────────────
# Google AI Studio API key (free tier). Get one at https://aistudio.google.com/apikey
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "").strip()

# ── Model / behaviour ──────────────────────────────────────────────────
# Gemini model. gemini-2.5-flash is fast and on the free tier. Alternatives:
# gemini-2.0-flash, gemini-2.5-pro (set GEMINI_MODEL in your .env to switch).
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

NEWS_COUNTRY: str = os.getenv("NEWS_COUNTRY", "us").strip() or "us"

# How many items to pull from each source.
NEWS_PAGE_SIZE: int = 8
WEB_SEARCH_RESULTS: int = 5


# ── News categories ────────────────────────────────────────────────────
# Maps the friendly label shown in the UI to the NewsAPI `category` value.
# NewsAPI supports: business, entertainment, general, health, science,
# sports, technology. We surface the three the brief calls out (technology,
# finance, sports) plus a few extras. "Finance" maps to NewsAPI "business".
CATEGORY_MAP: dict[str, str] = {
    "Technology": "technology",
    "Finance": "business",
    "Sports": "sports",
    "Business": "business",
    "Health": "health",
    "Science": "science",
    "Entertainment": "entertainment",
    "General": "general",
}

# Default category selected in the UI sidebar.
DEFAULT_CATEGORY: str = "Technology"


def has_google_key() -> bool:
    return bool(GOOGLE_API_KEY)


def has_newsapi_key() -> bool:
    return bool(NEWSAPI_KEY)
