"""NewsGenie — Streamlit front end.

A chat interface with a category sidebar. It manages a per-session conversation
thread (so the LangGraph checkpointer can keep context), surfaces system status
(which API keys are configured), shows fallback/error notices, and keeps the UI
responsive while the workflow runs.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

import config
from graph import run_turn
from llm import init_error, llm_available

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsGenie",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Session state ──────────────────────────────────────────────────────
def _init_session() -> None:
    if "thread_id" not in st.session_state:
        # One conversation thread per browser session.
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        # List of (role, text) tuples for rendering the transcript.
        st.session_state.history = []
    if "category" not in st.session_state:
        st.session_state.category = config.DEFAULT_CATEGORY


_init_session()


# ── Sidebar: category, status, controls ────────────────────────────────
with st.sidebar:
    st.header("📰 NewsGenie")
    st.caption("AI-powered news & information assistant")

    st.subheader("News category")
    st.session_state.category = st.selectbox(
        "Pick a category for live headlines",
        options=list(config.CATEGORY_MAP.keys()),
        index=list(config.CATEGORY_MAP.keys()).index(st.session_state.category),
        help="Used when you ask for the latest news. You can still ask any "
             "general question in the chat.",
    )

    st.divider()
    st.subheader("System status")
    # LLM status
    if llm_available():
        st.success(f"Gemini ready · `{config.GEMINI_MODEL}`")
    else:
        st.error("Gemini unavailable")
        st.caption(init_error() or "Set GOOGLE_API_KEY in your .env.")
    # News API status
    if config.has_newsapi_key():
        st.success("NewsAPI key detected")
    else:
        st.warning("No NewsAPI key — news falls back to web search.")
    # Web search is always available (no key needed)
    st.info("Web search (DuckDuckGo) · no key required")

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    with st.expander("Try asking…"):
        st.markdown(
            "- *Latest technology news*\n"
            "- *What's happening in finance?*\n"
            "- *Show me sports headlines*\n"
            "- *Who won the last Formula 1 race?*\n"
            "- *Explain what an ETF is*\n"
            "- *Hello! What can you do?*"
        )


# ── Main: title + transcript ───────────────────────────────────────────
st.title("📰 NewsGenie")
st.caption(
    "Ask for the latest news in a category, ask a factual question, or just "
    "chat. NewsGenie routes each query to the right tool automatically."
)

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)


# ── Chat input + workflow run ──────────────────────────────────────────
prompt = st.chat_input("Ask NewsGenie anything…")
if prompt:
    # Echo the user's message immediately.
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking & gathering sources…"):
            try:
                state = run_turn(
                    query=prompt,
                    category=st.session_state.category,
                    thread_id=st.session_state.thread_id,
                )
                answer = state.get("response") or "Sorry, I had nothing to say."
                notices = state.get("notices") or []
            except Exception as exc:
                answer = (
                    "⚠️ Something went wrong while processing that. "
                    f"Details: `{exc}`"
                )
                notices = []

        # Show any fallback / error notices above the answer.
        for n in notices:
            st.info(n)
        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
