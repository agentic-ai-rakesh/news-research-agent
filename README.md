# RockyBot: News Research Agent

Paste news article URLs, and a **LangGraph research agent** answers your questions from
those articles only, citing each fact as `[A1]`, `[A2]` and so on. A FastAPI backend runs
the agent, and a Streamlit app is the UI.

![](rockybot.jpg)

## Summary

```
URLs → fetch + extract article text → chunk (1000 / 200 overlap) → OpenAI embeddings → FAISS
Question → 🤖 agent: plan → search_articles / list_articles (up to 4 rounds) → answer with [A#]
        → finalize: sources, agent steps, tokens, latency
```

- **Agent:** an explicit LangGraph loop (`agent ⇄ tools → finalize`). The model plans
  its searches, runs one per part of a comparison, rephrases when a search misses, and
  answers only from retrieved passages. It says so when the articles don't contain the
  answer.
- **Tools:**
  - `list_articles`: what is indexed.
  - `search_articles(query, article_id?)`: semantic search, optionally limited to one article.
- **Grounding:** every fact is cited `[A#]`. Sources are the articles the answer actually
  cites. Citations that don't match an indexed article are flagged, not shown as sources.
- **Backend:** FastAPI with `/api/ingest`, `/api/index`, `/api/ask` and `/api/config`.
  It returns clear errors with the right status codes, and removes API keys from error text.
- **UI:** Streamlit, a client of the API:
  - URLs as 3 boxes, a free-text list or an uploaded `.txt` file
  - the indexed-articles panel
  - chat with sources
  - a "How the agent worked" panel with steps, tokens and latency
- **Observability:**
  - per-answer agent steps, token usage, latency, model and prompt version
  - a rotating log file in `logs/`
  - optional LangSmith tracing
- **Tests:** 37 pytest tests with a scripted LLM and keyword embeddings. They need no
  API key and no network.
- **Tech:** Python 3.11 · LangGraph · LangChain (core, OpenAI, text splitters) · OpenAI
  GPT-5.5 + text-embedding-3-small · FAISS (faiss-cpu) · BeautifulSoup · FastAPI ·
  Streamlit · pytest · ruff

## Quick start

Needs Python 3.11 or newer. `make setup` uses [uv](https://docs.astral.sh/uv/) if it is
installed; otherwise it uses `python3.11 -m venv` and pip.

```bash
git clone https://github.com/agentic-ai-rakesh/news-research-agent.git && cd news-research-agent
make setup            # create .venv, install requirements.txt, create .env from .env.example
# edit .env and set OPENAI_API_KEY
make test             # 37 tests, no API key needed
make run              # backend on :8000 + Streamlit on :8501 (Ctrl+C stops both)
```

Or run them in two terminals: `make backend`, then `make frontend`. The API docs are at
http://127.0.0.1:8000/docs. Other targets: `make lint`, `make format`, `make clean` and
`make setup-notebooks`.

Without make:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
.venv/bin/uvicorn api.main:app --port 8000
.venv/bin/streamlit run frontend/app.py
```

## How it works

### 1. Ingestion (`POST /api/ingest`, deterministic, no LLM)
- **Parse URLs:** blank boxes are skipped, duplicates removed, and only `http(s)` URLs
  accepted (max `MAX_URLS`). Invalid ones are reported back.
- **Fetch:** each page is downloaded with `requests` (timeout `FETCH_TIMEOUT_SECONDS`,
  with a User-Agent). Every failure is reported for that URL and doesn't stop the batch.
- **Extract:** BeautifulSoup keeps headings, paragraphs and list items from `<article>`
  (or `<body>`), and drops navigation, scripts, headers and footers. Pages with too
  little text are reported, for example when they need JavaScript or block bots.
- **Article ids:** the loaded articles are numbered `A1`, `A2`, … in input order.
- **Chunk:** `RecursiveCharacterTextSplitter` makes chunks of 1000 characters with 200
  overlap, splitting at paragraphs, then lines, sentences, commas and words. Each chunk
  keeps its article id, title, URL and chunk number.
- **Embed and index:** OpenAI `text-embedding-3-small`, stored in a FAISS inner-product
  index over normalized vectors, which gives cosine similarity. The index is written to
  a temporary folder and then swapped in, so a failed build never leaves half an index.
  `manifest.json` records the embedding model, chunk settings and articles.

### 2. Research agent (`POST /api/ask`)

```
START → agent ──tool calls──▶ tools ──▶ agent … ──no tool calls──▶ finalize → END
```

- **Input check:** an empty question, or one over 1000 characters, is rejected before
  any model call. With no index, or an index built with another embedding model, the
  API asks you to process URLs (HTTP 409).
- **agent node:** GPT-5.5 with the research system prompt (`agent/prompts.py`, versioned)
  decides whether to call tools or answer.
- **tools node:** runs `list_articles` / `search_articles` and counts the round. After
  `AGENT_MAX_TOOL_ROUNDS` (4), the agent loses its tools and must answer, so the loop
  always ends.
- **finalize node:** collects the answer, cited sources, unknown citations, every tool
  call with its arguments and result count, the retrieved passages with similarity
  scores, and the token usage summed over all model calls.

### 3. UI (Streamlit → API)
- **Sidebar:**
  - 3 URL boxes, a box for more URLs and a `.txt` upload, then **Process URLs** with live status
  - the indexed articles with links, and **Clear index**
  - the active model and retrieval settings
- **Chat:** answer, then **Sources**, then **How the agent worked**: steps, tokens,
  latency, model and prompt version. Questions are disabled until something is indexed.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/api/config` | Non-secret settings and whether `OPENAI_API_KEY` is set |
| GET | `/api/index` | The indexed articles (manifest), or `indexed: false` |
| POST | `/api/ingest` | `{"urls": [...]}` → index built; returns failures, invalid and duplicate URLs |
| DELETE | `/api/index` | Remove the index |
| POST | `/api/ask` | `{"question": "..."}` → answer, sources, steps, retrieved, usage, latency |

**Error codes:**
- **422:** bad input or no article text found.
- **409:** nothing indexed yet.
- **503:** `OPENAI_API_KEY` is missing.
- **502:** the model or embedding call failed. Keys are removed from the message.

## Configuration

Everything is set in `.env` (see `.env.example`); only `OPENAI_API_KEY` is required.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `gpt-5.5` | Agent model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings (changing it requires re-processing) |
| `LLM_TEMPERATURE` | empty (model default) | 0–2; leave empty for GPT-5.x reasoning models |
| `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | 1500 / 60 / 2 | Model call limits |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | Splitter settings (overlap < size) |
| `TOP_K` | 4 | Passages per search |
| `AGENT_MAX_TOOL_ROUNDS` | 4 | Tool rounds before the agent must answer |
| `MAX_URLS` / `FETCH_TIMEOUT_SECONDS` | 10 / 20 | Ingestion limits |
| `INDEX_DIR` / `LOG_DIR` | `data/faiss_index` / `logs` | Where the index and logs go (git-ignored) |
| `NEWS_RESEARCH_API_URL` | `http://127.0.0.1:8000` | Backend address used by Streamlit |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | off | Optional LangSmith traces of every step |

Invalid values stop startup with a message naming the variable.

## Tests

```bash
make test
```

The tests use a scripted chat model, so the agent's tool calls are predictable, and
keyword-hashing embeddings, so retrieval ranks for real. Page fetching is faked too: no
API key or network is needed.

| File | Covers |
|---|---|
| `test_config.py` | Defaults, environment overrides, invalid values, overlap < size, missing or placeholder key |
| `test_ingestion.py` | URL cleaning (blanks, invalid, duplicates, limit), article extraction without nav/footer, per-URL fetch failures, chunk metadata and overlap |
| `test_vector_store.py` | Index and manifest build, atomic swap, search relevance, stale embedding model refused, clear, JSON round trip, cosine scores, filter, mismatched files |
| `test_agent.py` | Search then cited answer, multi-search comparison, tool-round cap, unknown citations, input validation, no index, tool filtering |
| `test_api.py` | Health, config, ingest → ask → clear flow, 422/409/503/502, API key removed from errors |

**Latest results:**
- `make test`: **37 passed** (full list below). `make lint`: clean.
- Live run with GPT-5.5 on the three sample articles below (3 articles, 14 chunks):
  - A single question was answered with every fact cited `[A2]`.
  - A comparison made the agent list the articles, then search A3 and A1 separately,
    and cite both.
  - An off-topic question (Tesla) got "the indexed articles do not say…" with no sources.

### Test cases and results

Verified from scratch on a clean copy of the repository (Python 3.11.18, `make setup`, then
`make test` and `make lint`): **37 passed in 1.92s**, lint clean.

<details>
<summary>All 37 test cases</summary>

| File | Test case | Result |
|---|---|---|
| `test_agent` | Agent searches then answers with cited sources | ✅ passed |
| `test_agent` | Comparison question runs several searches | ✅ passed |
| `test_agent` | Tool rounds are capped and the model must answer | ✅ passed |
| `test_agent` | Unknown citations are reported not shown as sources | ✅ passed |
| `test_agent` | Invalid questions never reach the model (blank question) | ✅ passed |
| `test_agent` | Invalid questions never reach the model (question over 1000 characters) | ✅ passed |
| `test_agent` | Asking before indexing explains what to do | ✅ passed |
| `test_agent` | Tools list articles and filter by article | ✅ passed |
| `test_api` | Health and config | ✅ passed |
| `test_api` | Full flow ingest ask clear | ✅ passed |
| `test_api` | Ingest rejects input without valid urls | ✅ passed |
| `test_api` | Ingest reports when no page has article text | ✅ passed |
| `test_api` | Empty question is 422 | ✅ passed |
| `test_api` | Model errors are 502 with keys redacted | ✅ passed |
| `test_api` | Missing api key is 503 with the fix | ✅ passed |
| `test_config` | Defaults fix the old settings | ✅ passed |
| `test_config` | Environment overrides | ✅ passed |
| `test_config` | Invalid values are rejected with the variable name (`TOP_K=many`) | ✅ passed |
| `test_config` | Invalid values are rejected with the variable name (`TOP_K=0`) | ✅ passed |
| `test_config` | Invalid values are rejected with the variable name (`LLM_TEMPERATURE=3`) | ✅ passed |
| `test_config` | Overlap must be smaller than chunk size | ✅ passed |
| `test_config` | Missing or placeholder API key is refused (empty) | ✅ passed |
| `test_config` | Missing or placeholder API key is refused (spaces only) | ✅ passed |
| `test_config` | Missing or placeholder API key is refused (placeholder value) | ✅ passed |
| `test_ingestion` | Parse urls skips blanks and reports invalid and duplicates | ✅ passed |
| `test_ingestion` | Parse urls enforces the limit | ✅ passed |
| `test_ingestion` | Extract article keeps text and drops navigation scripts and footer | ✅ passed |
| `test_ingestion` | Fetch reports each failure and numbers only loaded articles | ✅ passed |
| `test_ingestion` | Chunks overlap and keep article metadata | ✅ passed |
| `test_ingestion` | Long paragraphs are split with overlap | ✅ passed |
| `test_vector_store` | Build writes index and manifest | ✅ passed |
| `test_vector_store` | Load returns a searchable store | ✅ passed |
| `test_vector_store` | Missing index is reported | ✅ passed |
| `test_vector_store` | Index from another embedding model is refused | ✅ passed |
| `test_vector_store` | Clear index | ✅ passed |
| `test_vector_store` | Faiss store round trip scores and filter | ✅ passed |
| `test_vector_store` | Faiss store rejects mismatched files | ✅ passed |

</details>

## Project structure

```
news_research_tool_project/
├── api/main.py                  FastAPI backend
├── frontend/app.py              Streamlit UI (API client)
├── news_research/
│   ├── config.py                Settings from .env, validated
│   ├── logging_setup.py         Console + rotating file log
│   ├── ingestion.py             URL parsing, fetching, extraction, chunking
│   ├── faiss_store.py           FAISS store (cosine, JSON metadata, no pickle)
│   ├── vector_store.py          Index build/load/clear + manifest
│   ├── models.py                OpenAI chat model and embeddings
│   └── agent/
│       ├── prompts.py           Versioned system prompt
│       ├── tools.py             list_articles, search_articles
│       ├── graph.py             LangGraph agent ⇄ tools → finalize
│       └── research_agent.py    ask(): validate, load index, run graph
├── tests/                       37 tests + fakes
├── notebooks/                   Learning notebooks (loaders, splitters, FAISS, retrieval)
├── requirements.txt             App + dev dependencies (pinned ranges)
├── requirements-notebooks.txt   Extra packages for the notebooks
├── Makefile · pyproject.toml · .env.example · .gitignore
```

Sample articles used in the live run:
- https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
- https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
- https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html

## What changed from the first version

- **Agent:** the single `RetrievalQAWithSourcesChain` call is now a LangGraph research
  agent with tools, multi-step search and a capped loop.
- **Imports:** `langchain.chains` and `langchain.text_splitter` no longer exist in
  LangChain 1.x, so a fresh install crashed. The app now uses `langchain-core`,
  `langchain-openai` and `langchain-text-splitters`.
- **Model:** the old completion model `gpt-3.5-turbo-instruct` at temperature 0.9 is
  replaced by a configurable chat model with the model's default temperature.
- **Inputs:** empty URL boxes are no longer sent to the loader. Invalid URLs and failed
  pages are reported individually. The `.txt` URL upload the old README promised now
  exists.
- **Chunking:** chunks now overlap by 200 characters (it was 0).
- **Status messages:** the misleading message order and the 2-second `sleep` are gone;
  progress is real.
- **Empty index:** asking before indexing now explains what to do instead of showing
  nothing.
- **Index storage:** no pickle (`allow_dangerous_deserialization` is gone), and no
  `langchain-community`, which is being sunset. A manifest stops an index built with
  another embedding model from being used.
- **Setup:** requirements are pinned to working ranges, the invalid `libmagic` pip
  package is removed, and the notebook packages have their own file.
- **Secrets:** `.gitignore` and `.env.example` are added, and the notebooks no longer
  hard-code an API key line.
- **Notebooks:** imports updated to the current LangChain packages.
- **New:** FastAPI backend, Makefile with `make setup`, tests and logging.
