# Bias-Aware News Analyzer

A Flask app that fetches news, summarizes articles/topics with an LLM, analyzes bias, provides retrieval‑augmented Q&A, streams answers live, and tracks evaluation metrics with persistence.

## Quick Start

- Python: `3.10+`
- Install:

  ```bash
  pip install -r requirements.txt
  ```

- Environment (`.env`):
  - `NEWS_API_KEY` = NewsAPI key
  - `LLM_PROVIDER` = `gemini` or `ollama`
  - `OLLAMA_MODEL` = model name when using Ollama (e.g. `llama3.1:8b`)
  - `DEFAULT_COUNTRY` = default country for headlines (e.g. `us`)
  - `GOOGLE_API_KEY` = required when `LLM_PROVIDER=gemini`
  - `OLLAMA_BASE_URL` = Ollama server (default `http://localhost:11434`)
  - `CACHE_TTL_SEC` = TTL seconds for NewsAPI cache (default `300`)

- Run:

  ```bash
  python app.py
  ```

- Open `http://127.0.0.1:5000/`

## Features

- Topic fetch via search or category
- Summarize all articles under a topic
- Topic bias analysis with score visualization
- Unbiased topic rewrite
- Per‑article summarize, bias analysis, and unbiased summary
- RAG Q&A (global and article‑scoped)
- Live streaming answers (SSE) for global and article Q&A
- TTL caching for NewsAPI requests (topic and category)
- Background jobs for slow tasks (async summarize all, topic bias)
- Metrics dashboard at `/metrics` persisted in `db.sqlite`

## Routes

- `GET /` — Home UI rendering (`app.py:242–293`)
- `POST /fetch` — Fetch by search topic (`app.py:295–315`)
- `POST /fetch_category` — Fetch by category (`app.py:316–374`)
- `POST /summarize_all` — Topic summary (`app.py:376–396`)
- `POST /summary_bias` — Bias analysis of topic summary (`app.py:397–413`)
- `POST /unbiased_topic_summary` — Unbiased rewrite of topic summary (`app.py:414–427`)
- `POST /ask_global` — Global Q&A with RAG (`app.py:429–489`)
- `GET /ask_global_stream` — Global Q&A streaming (`app.py:650–716`)
- `POST /summarize_article/<i>` — Per‑article summary (`app.py:491–510`)
- `POST /article_bias/<i>` — Per‑article bias analysis (`app.py:512–537`)
- `POST /unbiased_summary/<i>` — Per‑article unbiased summary (`app.py:556–573`)
- `POST /ask_article/<i>` — Article‑scoped Q&A with RAG (`app.py:576–648`)
- `GET /ask_article_stream/<i>` — Article‑scoped Q&A streaming (`app.py:717–793`)
- `POST /summarize_all_async` — Start summarize‑all job (`app.py:918–924`)
- `POST /summary_bias_async` — Start topic bias job (`app.py:925–931`)
- `POST /unbiased_topic_summary_async` — Start unbiased topic rewrite (`app.py:951–956`)
- `POST /summarize_article_async/<i>` — Start per‑article summary job (`app.py:958–964`)
- `POST /article_bias_async/<i>` — Start per‑article bias job (`app.py:966–972`)
- `POST /unbiased_summary_async/<i>` — Start per‑article unbiased summary job (`app.py:974–980`)
- `GET /job/<jid>` — Job status polling (`app.py:982–984`)
- `GET /suggest` — Title suggestions based on query (`app.py:932–950`)
- `GET /metrics` — Metrics dashboard (`app.py:986–994`)

## UI Behavior

- Ask form appears below the topic summary after clicking `Summarize All Articles`.
- Clicking article title/description opens the original link in a new tab (`nofollow`).
- Thumbnails render above titles when available.
- When an article action (summarize/analyze bias) is triggered, that article expands to full width.
- Search input shows placeholder `Russia Ukraine War` with empty value.
- Streaming Q&A intercepts Ask submissions and shows partial output progressively.
- Async job buttons for topic summarization and topic bias analysis show status and reload on completion.

## RAG & LLM

- Embeddings and vector store handled in `services/rag.py` via Chroma.
- Q&A context is constrained (~1200 chars) and blends topic/article summaries with retrieved snippets.
- LLM provider is configured via env (`LLM_PROVIDER`) with Gemini or Ollama.
- Streaming: Ollama uses server‑side streaming; Gemini yields final result via SDK.

## Metrics & Evaluation

- Counts: summaries (topic/article), bias analyses (topic/article), unbiased summaries, Q&A (global/article).
- Aggregates: average answer latency (ms), average answer word count.
- Examples: list of Q&A entries with type, index, question, words, latency.
- View at `http://127.0.0.1:5000/metrics`. Data persists in `db.sqlite`.

## Project Structure

- `app.py` — Flask routes, in‑memory state, caching, metrics, SSE, jobs
- `templates/index.html` — Main UI
- `templates/metrics.html` — Metrics UI
- `services/news_fetcher.py` — NewsAPI integration
- `services/llm.py` — Gemini/Ollama calls
- `services/rag.py` — Chroma CRUD & embeddings
- `db.sqlite` — SQLite database for metrics persistence (auto‑created)

Note: The `services` package uses Python’s implicit namespace packages; `__init__.py` is not required.

## Configuration Notes

- Ollama: ensure the local server is running and the specified model is pulled.
- The app uses `logging` for HTTP/data errors and persists evaluation metrics to SQLite automatically.

## System Architecture

```mermaid
flowchart TD
  A[Browser UI<br/>templates/index.html] -->|HTTP POST/GET| B[Flask App<br/>app.py]
  B -->|Fetch news| C[NewsAPI
  services/news_fetcher.py]
  B -->|LLM calls| D[Gemini or Ollama
  services/llm.py]
  B -->|RAG query/store| E[Chroma Vector DB
  services/rag.py]
  B -->|Persist metrics| F[SQLite
  db.sqlite]
  B -->|Render| A
  B -->|SSE streams| A
  subgraph Async Jobs
    J1[summarize_all_async]
    J2[summary_bias_async]
    J3[unbiased_topic_summary_async]
    J4[summarize_article_async]
    J5[article_bias_async]
    J6[unbiased_summary_async]
  end
  B -->|Start jobs| Async Jobs
```

## Troubleshooting

- Missing NewsAPI key: add `NEWS_API_KEY` to `.env`.
- Ollama connection issues: verify server and model name.
- Vector store errors: ensure `chromadb` and `sentence-transformers` installed; the first run initializes the collection.
- Streaming not working: ensure browser supports `EventSource`; check `/ask_global_stream` and server logs.
- Metrics not persisting: verify write permissions and presence of `db.sqlite` after actions.

## Production

- Use a production WSGI server (e.g. gunicorn + reverse proxy) and proper secret management.
- Consider a proper job queue (RQ/Celery) and a persistent cache for NewsAPI.