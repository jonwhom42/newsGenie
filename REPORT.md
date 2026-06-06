# NewsGenie — Project Report
### An AI-Powered Information and News Assistant

**Stack:** Python · LangGraph · Streamlit · Google Gemini (`langchain-google-genai`) · NewsAPI.org · DuckDuckGo
**Submission date:** 2026-06-05

---

## 0. Executive summary

NewsGenie is a single, unified assistant that does two jobs that are normally
split across different apps:

1. **Answers general and factual queries** like a chatbot, and
2. **Curates real-time, source-attributed news** by category.

The hard part is doing both *in one conversation* — knowing when a message means
"chat with me" versus "fetch me the news" versus "look this fact up." NewsGenie
solves this with a **LangGraph workflow** whose first node is an **intent
classifier**, after which the query is routed to the right tool (news API, web
search, or the model's own knowledge) and a final node **synthesises a
trustworthy, sourced answer**. A **Streamlit** UI wraps it with category
selection, session memory, live status, and graceful error notices.

This document covers the three required deliverables:

- **§1 — AI chatbot design:** conversation management & query differentiation.
- **§2 — Real-time news integration:** sample outputs for Technology, Finance, Sports.
- **§3 — Workflow & error handling:** API integration, fallback mechanisms, and
  end-to-end query processing.

> ✅ **Verification status:** the system is fully built and was run end-to-end on
> 2026-06-05 against live Gemini (`gemini-2.5-flash`) and NewsAPI. All sample
> outputs and routing tables in this report are **real captures from that run**,
> not mock-ups.

---

## 1. AI Chatbot Design — Conversation Management & Query Differentiation

### 1.1 Conversation management

**Goal:** keep each user's session coherent (remember context), responsive, and
isolated from other users.

| Concern | How NewsGenie handles it |
|---|---|
| **Memory / context** | Graph state carries a `messages` list reduced with LangGraph's `add_messages`. A **`MemorySaver` checkpointer** persists this per **`thread_id`**, so follow-up turns ("and in finance?") see prior context. |
| **Session isolation** | Each browser session gets a `uuid4` `thread_id` stored in `st.session_state`. Two users never share a thread. |
| **Reset** | A "New conversation" button mints a fresh `thread_id` and clears the transcript. |
| **Responsiveness** | The UI echoes the user's message instantly and runs the workflow inside a `st.spinner`, so the app never appears frozen. The compiled graph and the Gemini client are **cached** (`lru_cache`) so they're built once per process. |
| **Transparency** | The sidebar shows live system status (Gemini ready? NewsAPI key present?) and every answer can be preceded by **notices** explaining fallbacks (e.g. "No live news found — answering from a web search"). |

**Conversation-design guidelines applied:**

1. **One responsibility per turn.** Each message is classified and answered; the
   assistant doesn't try to be news + chat + search simultaneously — it *routes*.
2. **Always ground claims.** For news/search answers the model is instructed to
   use **only** the retrieved items and to **attribute each claim to a source**,
   ending with a "Sources" list. This is the core anti-misinformation lever.
3. **Be honest about limits.** If data is thin, missing, or potentially
   unreliable, the assistant says so rather than fabricating.
4. **Neutral, concise tone.** Lead line + 3–6 bullets keeps answers scannable —
   the antidote to "information overload."
5. **Degrade, never crash.** Every failure mode has a defined user-facing
   behaviour (see §3).

### 1.2 Query differentiation (the router)

The first graph node, `classify`, sorts every message into exactly one **intent**:

| Intent | Meaning | Example | Path taken |
|---|---|---|---|
| `news` | Wants current headlines / recent developments in a domain | *"latest technology news"*, *"what's happening in sports"* | → `fetch_news` |
| `websearch` | A factual question needing fresh/external info | *"who won the last F1 race?"*, *"current price of gold"* | → `web_search` |
| `general` | Greetings, opinions, general knowledge | *"hi, what can you do?"*, *"explain inflation"* | → `general_chat` |

**Primary method — LLM classification.** Gemini receives a strict system prompt
and returns a tiny JSON object:

```json
{"intent": "news|websearch|general", "topic": "<short search topic>"}
```

It returns both the **intent** and the **topic** to look up, so a request like
*"any news on the Nvidia earnings?"* yields `{"intent":"news","topic":"Nvidia
earnings"}` — which drives a targeted NewsAPI *everything* search rather than a
generic category pull.

**Fallback method — deterministic keywords.** If Gemini is unavailable or returns
unparseable text, a regex-based classifier takes over (news keywords like
*latest/headline/breaking*; question/search keywords like *who/price/score/?*).
This guarantees the router *always* produces a valid route.

**Live LLM routing** — actual intents the running app assigned (2026-06-05):

| User message | Intent chosen | Route taken |
|---|---|---|
| "Latest technology news" | `news` | NewsAPI (Technology) → summary |
| "What's happening in finance?" | `news` | NewsAPI (Finance) → summary |
| "Any news on Nvidia earnings?" | `news` | NewsAPI topic search `q=Nvidia earnings` |
| "Explain what an ETF is in simple terms" | `websearch` | DuckDuckGo → summary |
| "Hi! What can you do?" | `general` | Direct conversational reply (no tools) |

**Keyword fallback** — the deterministic backup (used if the LLM is unavailable),
verified independently:

```
news       <- 'latest technology news'
websearch  <- 'who won the F1 race?'
general    <- 'hello there'
websearch  <- 'explain ETFs'
news       <- 'sports headlines today'
```

**Why a separate router node (not one mega-prompt)?** Separation makes the system
inspectable and cheap to test, lets each downstream node specialise, and means a
classification failure can fall back independently of answer generation.

---

## 2. Real-Time News Integration — Sample Outputs

### 2.1 How it works
- The user picks a **category** in the sidebar (mapped to a NewsAPI category;
  *Finance* → `business`).
- On a `news` intent, `fetch_news` calls **NewsAPI top-headlines** for that
  category (or **everything** if a specific topic was extracted).
- Articles are normalised to `{title, source, url, published_at, description}`,
  empty/`[Removed]` entries are dropped, and the `generate` node asks Gemini to
  summarise **using only those articles**, with per-claim attribution + a Sources
  list.

> The blocks below are **actual, unedited outputs captured from the running app
> on 2026-06-05** (Gemini `gemini-2.5-flash` + live NewsAPI). Live headlines
> change every run, so re-running will show different stories — but the routing,
> grounding, and source attribution are exactly as shown.

### 2.2 Technology
**User:** `Latest technology news`  ·  **Category:** Technology  ·  **Route:** `news`

```
Recent technology news highlights the impact of AI on employment and corporate
spending, shifts in the fintech market, and emerging trends in device
preferences and deeptech investment.

• AI's Impact on Workforce and Investment: Some companies, including Teradata
  and TTEC, are reportedly cutting worker pay or pausing retirement benefits to
  fund AI investments (Livemint). India's top IT companies are experiencing
  "jobless growth" — revenue rising with stable headcount due to automation
  (The Times of India).
• Fintech Market Evolution: Payment aggregators are increasingly targeting
  startups and small merchants amid UPI's zero-MDR regime (Livemint).
• Deeptech Investment Growth: Silicon Valley's South Park Commons is expanding
  into India's deeptech sector — semiconductors and robotics (The Times of India).
• Emerging Consumer Tech Preference: A trend toward "smaller tech" suggests
  consumers favouring more compact devices (Gottabemobile.com).
• New Smart Soundbar Release: Bose introduced a Dolby Atmos Smart Soundbar with
  Bluetooth/Wi-Fi and Alexa/Google Assistant integration (Amazon.co.jp).

Sources:
• Livemint — Companies are now cutting worker pay to fund AI investment
• The Times of India — Jobless growth in IT: Revenue grows, headcount stagnates
• Livemint — Fintechs intensify merchant push as payment aggregator market crowds
• The Times of India — Silicon Valley's South Park Commons ramps up India deeptech
• Gottabemobile.com — Why Smaller Tech Is Making a Big Comeback
• Amazon.co.jp — Bose Smart Soundbar (Dolby Atmos)
```

### 2.3 Finance
**User:** `What's happening in finance?`  ·  **Category:** Finance (`business`)  ·  **Route:** `news`

```
Recent financial news includes proposed changes to bank deposit insurance in
India, significant corporate acquisition proposals, and developments around the
SpaceX IPO.

• India's finance ministry has proposed increasing the bank deposit insurance
  limit from ₹5 lakh to ₹7.5 lakh, pending PMO approval (Livemint).
• Fidelity has reportedly lowered the barrier for retail investors to
  participate in the SpaceX IPO, which is seeking a record $75 billion
  (Slashdot.org, citing Barron's / Yahoo Finance / NYT).
• UWM Holdings (UWMC) reaffirmed its proposal to acquire Two Harbors Investment
  Corp. for $12.50 per share in cash, with a stock-election option (AP).
• A Sebi order raised concerns for Rajesh Exports' battery PLI project, alleging
  ~₹15.15 trillion in revenue was materially misrepresented FY21–FY25 (Livemint).
• In Nigeria, FCMB is collaborating with the Federal Government and IITA to
  establish a National Youth in Agribusiness Land Trust Fund (The Punch).

Sources:
• Livemint — Govt may protect bank deposits up to ₹7.5 lakh
• Slashdot.org — How Much Do I Need to Buy the SpaceX IPO? (via Barron's)
• Associated Press — UWMC reaffirms proposal to acquire Two Harbors at $12.50/share
• Livemint — Sebi order casts shadow on Rajesh Exports' battery PLI project
• The Punch — FCMB partners FG, IITA on agribusiness drive
```

### 2.4 Sports
**User:** `Show me sports headlines`  ·  **Category:** Sports  ·  **Route:** `news`

```
Recent sports headlines include backlash against Buffalo Wild Wings, updated
college basketball rankings, an NBA pre-draft workout, WNBA commentary, a sports
analyst's health transformation, USMNT players' interest in the Knicks, and an
MLB hitting streak.

• Buffalo Wild Wings faced criticism — including from retired NBA star Blake
  Griffin — for offering refuge to a fan who stormed the NBA Finals court
  (Sporting News).
• In men's college basketball, St. John's entered the Top 10 and Kentucky joined
  the Top 25; Louisville's Isaac McKneely had a pre-draft workout with the OKC
  Thunder (Fox Sports, USA Today).
• Jason Whitlock suggested "Caitlin Clark fever" has subsided in Indianapolis
  after recent Indiana Fever losses (The Comeback).
• Stephen A. Smith showcased a significant body transformation after nearly
  becoming diabetic (BroBible).
• San Francisco Giants outfielder Jung Hoo Lee extended his MLB-best active hit
  streak to 12 games (USA Today).

Sources:
• Sporting News — Buffalo Wild Wings faces backlash from Blake Griffin and others
• Fox Sports — College Basketball Rankings: St. John's into Top 10, Kentucky Top 25
• The Comeback — Jason Whitlock says Caitlin Clark fever is over in Indianapolis
• BroBible — Stephen A. Smith shows off body transformation
• USA Today — Giants' Jung Hoo Lee has longest active hit streak in MLB
```

### 2.5 Bonus — topic-targeted news **and anti-hallucination grounding** (real capture)
**User:** `Any news on Nvidia earnings?`  ·  **Route:** `news` (topic = "Nvidia earnings")

NewsGenie's router extracted the topic `Nvidia earnings`, so `fetch_news` called
NewsAPI **everything** with `q=Nvidia earnings` instead of a category pull. The
feed had Nvidia-adjacent stories but nothing about the earnings specifically —
and the model **said so rather than inventing numbers**:

```
Based on the provided articles, there is no direct news regarding Nvidia's
earnings. The articles mention Nvidia in other contexts, such as its chips being
used in Apple's new Siri and its CEO influencing another company's market value.

• Apple's new Siri, set to launch in September, will be powered by Google cloud
  and Nvidia chips (Macdailynews.com).
• A comment from Nvidia CEO Jensen Huang reportedly added tens of billions of
  dollars to Marvell Technology's market value (TheStreet).
```

This is the **anti-misinformation design working live**: the model is instructed
to use *only* retrieved items, so when the exact story isn't there it admits the
gap instead of fabricating — exactly the behaviour that builds user trust.

---

## 3. Workflow & Error Handling

### 3.1 End-to-end query processing (the LangGraph workflow)

```
                         ┌──────────────┐
   user query  ─────────►│   classify   │  (LLM → JSON; keyword fallback)
   + category            └──────┬───────┘
                  intent ───────┼─────────────────────────────
              news │            │ websearch         │ general
                   ▼            ▼                    ▼
            ┌────────────┐ ┌────────────┐    ┌──────────────┐
            │ fetch_news │ │ web_search │    │ general_chat │──► END
            └─────┬──────┘ └─────┬──────┘    └──────────────┘
       articles?  │              │
         yes│  no │ (fallback)   │
            │     └─────────────►│
            ▼                    ▼
         ┌─────────────────────────┐
         │        generate         │  (summarise + cite sources)──► END
         └─────────────────────────┘
```

| Node | Input | Action | Output |
|---|---|---|---|
| `classify` | query | LLM/keyword → intent + topic | `intent`, `topic` |
| `fetch_news` | topic/category | NewsAPI call (uniform result dict) | `news`, `notices` |
| `web_search` | topic/query | DuckDuckGo call | `search`, `used_fallback`, `notices` |
| `general_chat` | query + history | Gemini direct answer w/ context | `response` |
| `generate` | news + search | Gemini synthesis with sources | `response` |

Conditional edges encode the logic: `classify` forks on intent; `fetch_news`
forks on *"did we get articles?"* (yes → `generate`, no → `web_search`).

### 3.2 API integration

| Capability | Service | Auth | Endpoint / lib | Notes |
|---|---|---|---|---|
| LLM / chatbot | Google Gemini | `GOOGLE_API_KEY` | `langchain-google-genai` `ChatGoogleGenerativeAI` | cached; chosen because LangGraph composes with LangChain chat models |
| Real-time news | NewsAPI.org | `NEWSAPI_KEY` | `/v2/top-headlines`, `/v2/everything` | category headlines or topic search |
| Web search | DuckDuckGo | **none** | `ddgs` (falls back to legacy `duckduckgo_search`) | complements/falls back for news |

All three are wrapped so they **return a uniform dict** —
`{ok, items, error, source, message}` — and **never raise**. The graph branches
on `ok`/`items`, never on exceptions.

### 3.3 Fallback mechanisms

The brief specifically calls out *API failures, missing keys, and no-news
situations.* Each is handled:

| Failure | Detection | Fallback behaviour |
|---|---|---|
| **Missing `GOOGLE_API_KEY`** | `config.has_google_key()` / `llm.get_llm()` returns `None` | App runs in **degraded mode**: keyword routing, and `generate` prints raw curated results without an AI summary. Sidebar shows "Gemini unavailable". |
| **Missing `NEWSAPI_KEY`** | `fetch_news` returns `error="missing_key"` | News intent **falls back to web search**; sidebar warns. |
| **NewsAPI HTTP/API error** | non-200 status or `status != "ok"` | Structured error → `_route_after_news` sends flow to `web_search`. |
| **No articles found** | `items == []` | Same fallback to web search, with a user notice. |
| **Network timeout / exception** | `requests` exceptions caught per-call | Returns `network_error`; flow continues; user sees a notice. |
| **Web search empty/fails** | `web_search` returns `no_results`/`network_error` | `generate` emits an honest "couldn't find anything" message + suggestions. |
| **LLM classify returns junk** | JSON parse fails | Deterministic **keyword classifier** used instead. |
| **LLM answer call errors** | exception in `generate`/`general_chat` | Falls back to plain formatted source list; UI shows a contained error. |
| **`ddgs` not installed** | import guard returns `None` | Returns `missing_dependency` cleanly. |

**Layered safety:** tool layer (never raises) → graph layer (routes on data, not
exceptions) → UI layer (try/except around `run_turn`, renders notices). A failure
at any layer is absorbed by the next.

### 3.4 Performance & UX optimisations
- **Caching:** compiled graph and Gemini client built once (`lru_cache`).
- **Bounded calls:** HTTP timeouts (10s), `max_retries`, capped `page_size`/result
  counts to keep latency and token use predictable.
- **Instant feedback:** user message echoed immediately; spinner during work.
- **Session memory** via checkpointer avoids re-sending full history manually.

### 3.5 Anti-misinformation design (ties it together)
1. **Trustworthy sources by construction** — reputable news outlets via NewsAPI;
   each item carries its **source name + publish date + URL**.
2. **Grounded generation** — the model may only use retrieved items and must cite
   them; it's told to flag thin/unreliable material.
3. **Transparency** — fallback notices tell the user *where* an answer came from
   (live news vs. web search vs. model knowledge), so they can judge reliability.

---

## 4. Results / What was delivered

✅ Interactive AI assistant: instant general answers **and** real-time curated news.
✅ Integrated **real-time news API + dynamic web search + LangGraph workflow**.
✅ **Streamlit** UI with session management, category selection, status & responsive design.
✅ Documented **fallback mechanisms and optimisations** for reliable performance under failures.

## 5. How to run
See **README.md**. In short: `pip install -r requirements.txt`, copy
`.env.example` → `.env` and add keys, then `streamlit run app.py`.

## 6. Possible extensions
- Persist sessions to disk/DB (swap `MemorySaver` for `SqliteSaver`).
- Add a source-credibility score and explicit fact-check cross-referencing.
- Stream tokens to the UI; add multi-language and personalised feeds.
- Cache repeated category pulls for a short TTL to cut API calls.
