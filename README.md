# 📰 NewsGenie — An AI-Powered Information & News Assistant

NewsGenie is a unified assistant that **answers general queries** and **delivers
real-time, curated news** in a single chat interface. It distinguishes news
requests from conversational/factual questions, fetches live headlines from a
news API, complements answers with a web-search tool, and orchestrates
everything through a **LangGraph** workflow behind a **Streamlit** UI.

This repository is the deliverable for the *NewsGenie* course-end project.

---

## ✨ What it does

- **Conversational chatbot** — chat with Gemini, with conversation memory per session.
- **Query differentiation** — every message is routed to one of three intents:
  `news`, `websearch`, or `general`.
- **Real-time news** — top headlines by category (Technology, Finance, Sports, …)
  via [NewsAPI.org](https://newsapi.org).
- **Web search** — live external info via DuckDuckGo (no API key required).
- **LangGraph workflow** — a graph routes the query, gathers data, and synthesises
  a sourced answer.
- **Robust fallbacks** — missing keys, failed API calls, and empty results are all
  handled gracefully (news → web-search fallback, LLM-off degraded mode, etc.).

---

## 🏗️ Architecture

```
Streamlit UI (app.py)
        │  query + selected category + session thread_id
        ▼
LangGraph workflow (graph.py)
   classify ──► fetch_news ──(empty?)──► web_search ──► generate ──► answer
        │                                   ▲
        ├──► web_search ────────────────────┘
        └──► general_chat ──► answer
        ▲
        └── Gemini via langchain-google-genai (llm.py)
            NewsAPI + DuckDuckGo tools (tools.py)
```

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit chat UI, session management, status & notices |
| `graph.py` | Builds/compiles the LangGraph workflow + `run_turn()` entry point |
| `nodes.py` | Node functions: classify, fetch_news, web_search, general_chat, generate |
| `state.py` | Typed graph state (`GraphState`) with `add_messages` memory |
| `tools.py` | NewsAPI + DuckDuckGo wrappers (never raise; uniform result dicts) |
| `llm.py` | Cached `ChatGoogleGenerativeAI` (Gemini) instance + availability checks |
| `config.py` | Env/config, category map, key-presence helpers |

See **REPORT.md** for the full design write-up (chatbot design, sample outputs,
workflow & error-handling) — that is the graded submission document.

---

## 🚀 Quick start

### 1. Prerequisites
- Python **3.10–3.12**
- A **Google AI Studio API key** (free, no card) — https://aistudio.google.com/apikey
- *(Recommended)* a **NewsAPI key** (free) — https://newsapi.org/register
  *(Without it, NewsGenie still runs and falls back to web search.)*

### 2. Install
```bash
# from the project folder
python -m venv .venv

# activate it
source .venv/bin/activate          # macOS / Linux
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env               # macOS / Linux
Copy-Item .env.example .env        # Windows PowerShell
```
Then edit `.env` and set at least `GOOGLE_API_KEY` (and ideally `NEWSAPI_KEY`).

> 💡 **Model tip:** the default is `gemini-2.5-flash` (fast, free tier). You can
> switch to `gemini-2.0-flash` or `gemini-2.5-pro` via `GEMINI_MODEL` in `.env`.

### 4. Run
```bash
streamlit run app.py
```
Streamlit opens `http://localhost:8501`. Pick a category in the sidebar and start
chatting.

---

## 🧪 Try these

| You type | NewsGenie does |
|----------|----------------|
| `Latest technology news` | Routes → `news`, pulls Technology headlines, summarises with sources |
| `What's happening in finance?` | Routes → `news`, category = Finance/business |
| `Show me sports headlines` | Routes → `news`, category = Sports |
| `Who won the last F1 race?` | Routes → `websearch`, DuckDuckGo + summary |
| `Explain what an ETF is` | Routes → `websearch` / `general` knowledge answer |
| `Hi, what can you do?` | Routes → `general`, conversational reply |

---

## 🛟 Fallback & error handling (summary)

| Situation | Behaviour |
|-----------|-----------|
| `GOOGLE_API_KEY` missing | Sidebar shows "Gemini unavailable"; news still browsable; raw results shown without AI summary |
| `NEWSAPI_KEY` missing | Sidebar warns; news intent falls back to web search |
| NewsAPI returns 0 articles / error | Auto-fallback to DuckDuckGo, with a notice |
| Web search fails | Honest "couldn't find anything" message + suggestions |
| Network timeout / exception | Caught per-tool; user sees a notice, app stays alive |
| LLM/JSON classify error | Deterministic keyword-based routing kicks in |

Full details and rationale are in **REPORT.md** (§3 Workflow & Error Handling).

---

## 📁 Project layout
```
newsGenie/
├── app.py            # Streamlit UI
├── graph.py          # LangGraph workflow
├── nodes.py          # graph node functions
├── state.py          # graph state schema
├── tools.py          # NewsAPI + DuckDuckGo
├── llm.py            # Gemini (ChatGoogleGenerativeAI) wrapper
├── config.py         # settings & categories
├── requirements.txt
├── .env.example
├── README.md
└── REPORT.md         # ← graded submission document
```
