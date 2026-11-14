# PUGPT — University of the Punjab GPT

PUGPT is a public Q&A assistant **exclusively for the University of the Punjab, Pakistan** — admissions, fee structure, semesters/academic regulations, merit lists, departments and campus information — backed by retrieval-augmented generation (RAG) over official `pu.edu.pk` documents only. It never answers from general world knowledge, and it refuses to guess.

PUGPT is **pre-production**. Read [deployment.md](deployment.md) before exposing anything publicly.

---

## What PUGPT does and does not do

**Does**
- Answers PU questions from a curated knowledge base (6 official documents, 519 chunks in ChromaDB).
- Cites the exact PDF filename + page number it used.
- Detects its own unsupported numeric claims and refuses rather than hallucinate.
- Falls back (optionally) to **live official PU web search** (Tavily, `pu.edu.pk` only) for current announcements/dates — disabled by default.
- Rate-limits itself: 10 requests / minute / IP and 500 requests / UTC day application-wide.

**Does not**
- Answer non-PU questions.
- Use unofficial/third-party sources as authority.
- Have any login/user accounts (anyone can ask; abuse is bounded by the rate limits, not identity).
- Provide legal/fee guarantees — always verify against official documents.

---

## Architecture

```
 Browser (React + Vite + Tailwind)            frontend/
        |
        v
 FastAPI backend (uvicorn, single worker)      backend/
        |
        |--- FastEmbed (embeddings, in-process)  nomic-embed-text-v1.5-Q
        |--- ChromaDB (persistent)              backend/data/chroma/
        |--- Groq API (generation)              openai/gpt-oss-120b
        \--- Tavily API (optional live PU web)  official pu.edu.pk only
```

Pipeline per question: embed (FastEmbed) -> retrieve top chunks (hard cosine-distance <= 0.5) -> (optional live search if enabled **and** PU-context is missing or the question asks for current/latest info) -> grounded prompt (PDF + labeled web excerpts; PDF wins on regulations, web wins on currency, conflicts are flagged, never silently resolved) -> Groq -> numeric-claim verifier (find_unsupported_claims) -> answer + sources.

---

## Prerequisites

| Component | Tested with | Notes |
|---|---|---|
| Python | 3.14.7 | from [python.org](https://www.python.org); 3.11+ should work, untested |
| Node.js + npm | 24.18.0 / 11.16.0 | LTS recommended |
| Ollama | latest | **NOT required** — only needed for the documented rollback path (`EMBEDDING_PROVIDER=ollama`) |
| Git | any | you must clone/copy this project |

Disk: ~2 GB for Ollama model + ~14 MB ChromaDB (519 records; 5 documents → report/admissions/fee/eligibility/VT PDFs).

---

## 1. Backend installation

```powershell
cd C:\Stoic\pu-gpt\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # exact-pinned, verified versions
```

> The runtime deps are pinned to the exact versions verified on Python 3.14.7. A fully-locked transitive set is not in place yet (a fresh install may resolve newer compatible transitive versions) — see `deployment.md`. Dev/test tooling lives in `requirements-dev.txt` (`-r requirements.txt` + pytest/ruff, whose versions are **not verified in this venv**); the test suites themselves run as plain `python scripts/test_*.py` with no extra tooling.

Create the environment file (never commit it, never share it):

```powershell
copy .env.example .env
# edit .env — see "Configuration" below
```

## 2. Embeddings — FastEmbed (default) or Ollama

PUGPT embeds questions at runtime. The **production provider is FastEmbed**, using the in-process ONNX model `nomic-ai/nomic-embed-text-v1.5-Q` — **no Ollama daemon required** in production. The original Ollama embedding path is retained as a rollback for development.

### 2A. FastEmbed (default)
- Model: `nomic-ai/nomic-embed-text-v1.5-Q` (768-dim, Apache-2.0, ~200 MB resident after
  first embed; loads in ~0.5 s on modern CPUs).
- Nothing to download at build time — at **first use**, FastEmbed fetches the model
  files from Hugging Face (`nomic-ai/nomic-embed-text-v1.5-Q`) into its default cache
  (`%TEMP%/fastembed_cache` on Windows, `~/.cache/fastembed` on Linux; override via
  `FASTEMBED_CACHE_DIR` if you need a persistent image layer).
- First warmup happens at **backend startup** (`FastEmbed model loaded: … (dim=768)`).
- In-process; no network; ~290 MB peak RSS measured (comfortable for Render's 512 MB free tier).

### 2B. Ollama (development / rollback)
Install Ollama from <https://ollama.com> and pull the embedding model:

```powershell
ollama pull nomic-embed-text
```

Ollama must listen on `http://localhost:11434` (or set `OLLAMA_BASE_URL`). PUGPT previously embedded via Ollama for historical reasons — it is kept purely as a rollback path.

### 2C. Provider ↔ collection pairing (must match — otherwise retrieval is garbage)
| Provider | Collection | Status |
|---|---|---|
| `EMBEDDING_PROVIDER=fastembed` (default) | `pugpt_knowledge_fastembed_test` | ACTIVE — re-indexed 2026-11-14 (519 chunks, identical chunking/counts) |
| `EMBEDDING_PROVIDER=ollama` | `pugpt_knowledge` | Rollback — original Ollama vectors (519 records, unchanged) |

**You must select the collection to match the provider.** Rollback procedure:

```bash
# .env or environment:
EMBEDDING_PROVIDER=ollama
CHROMA_COLLECTION=pugpt_knowledge
# Chroma DB is on disk; no re-ingestion required; Ollama must be running.
```

### 2D. Verify (no LLM calls)
```bash
python -c "from app.services.embedding_service import embedding_service; \
  print(embedding_service.provider_name, len(embedding_service.embed_text('test')))"
# expected: fastembed 768
```

## 3. Frontend installation

```powershell
cd C:\Stoic\pu-gpt\frontend
npm install
```

## 4. Configuration (`backend/.env`)

All keys are read from `backend/.env` (values override the built-in defaults; the file is git-ignored). **Never commit or paste real keys anywhere.** Safe, working defaults are in `backend/.env.example`.

### Generation provider
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` → groq if `GROQ_API_KEY` set, else ollama; `groq` forces Groq; `ollama` forces local Ollama | Set `groq` (or `auto` + key) |
| `GROQ_API_KEY` | _(empty)_ | Groq API key (free tier works) — **secret, server-side only** | **Required** for generation |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1/chat/completions` | endpoint | keep default |
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | model | keep (or pick another Groq model) |
| `GROQ_TIMEOUT` | `120` | seconds | keep |

### Tavily (official PU live web search — optional, disabled by default)
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `WEB_SEARCH_ENABLED` | `false` | master switch for the web fallback | flip to `true` **only** when you want live PU web results (consumes quota); leave `false` to serve PDFs only |
| `WEB_SEARCH_PROVIDER` | `tavily` | only Tavily supported | keep |
| `WEB_SEARCH_ON_RECENCY` | `true` | allow web search when the question asks for current/latest info even with PDF context | keep or set `false` for strictest-limits |
| `TAVILY_API_KEY` | _(empty)_ | Tavily key — **secret, server-side only** | only if `WEB_SEARCH_ENABLED=true` |
| `TAVILY_BASE_URL` | `https://api.tavily.com` | endpoint | keep |
| `WEB_SEARCH_TIMEOUT` | `15` | seconds | keep |
| `WEB_SEARCH_MAX_RESULTS` | `5` | 1 credit per search (basic depth) | keep |
| `WEB_SEARCH_ALLOWED_SUFFIXES` | `pu.edu.pk` | comma-separated trusted hosts (official PU only) | keep |

`TAVILY_API_KEY` is consumed only on fallback searches — never on ordinary PDF-answerable questions.

### Embeddings (Ollama)
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint | keep on localhost |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | **ignored under `EMBEDDING_PROVIDER=fastembed`** — used only on rollback | keep (or re-ingest on rollback) |
| `OLLAMA_CHAT_MODEL` | `llama3.2:latest` | only if `LLM_PROVIDER=ollama` | irrelevant with Groq |
| `OLLAMA_TIMEOUT` | `30` | ping/embed timeout | keep |
| `OLLAMA_CHAT_TIMEOUT` | `120` | generation timeout (Ollama only) | irrelevant with Groq |

### HTTP / CORS
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated browser origins | **set to your production origin** (e.g. `https://pu.example.edu.pk`); required or the browser blocks `/api/*` (or serve UI and API from one origin) |

### Retrieval / RAG
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `RAG_TOP_K` | `5` | chunks kept per answer | keep |
| `RAG_DISTANCE_THRESHOLD` | `0.5` | cosine distance hard threshold | keep (do not tune casually) |
| `RETRIEVAL_TOP_K` | `4` | legacy/compat setting | keep |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `800`/`150` | ingestion chunking | keep (must match ingestion) |
| `INGESTION_BATCH_SIZE` | `16` | embeddings per batch | keep |

### Rate limiting
| Variable | Default | Meaning | Public-deploy requirement |
|---|---|---|---|
| `RATE_LIMIT_CHAT` | `10` | per-IP requests per window | keep or tune; never set very high |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | per-IP window | keep |
| `RATE_LIMIT_TRUST_PROXY` | `false` | honor `X-Forwarded-For` (rightmost) **only** behind a conforming proxy | set `true` **only** behind your proxy that guarantees a correct `X-Forwarded-For` |
| `MAX_CHAT_REQUESTS_PER_DAY` | `500` | global requests / UTC day (SQLite, persists across restarts) | tune to your free quota; never `0`/negative (falls back to 500) |
| `RATE_LIMIT_GLOBAL_DB_PATH` | `data/global_rate_limit.sqlite3` (absolute under `backend/`) | where the SQLite counter lives | keep or absolute path; **delete stale file to reset the day** (avoid committing it) |

### ChromaDB
| Variable | Default | Meaning |
|---|---|---|
| `CHROMA_PERSIST_PATH` | `backend/data/chroma` | persistent vector store |
| `CHROMA_COLLECTION` | `pugpt_knowledge_fastembed_test` | collection name — MUST match provider | fastembed->fastembed_test, ollama->pugpt_knowledge |

---

## 5. Knowledge-base ingestion

PUGPT ingests **verified official PU documents** only. Standard workflow:

```powershell
cd C:\Stoic\pu-gpt\backend
.\venv\Scripts\Activate.ps1

# 1) validate the registry (sources.json) and extractable text
python scripts\validate_knowledge_base.py

# 2) human-verify documents; status must be 'verified' in the registry
#    (validator + human decision; nothing is auto-verified)

# 3) ingest (path can be a file or a directory; recursive)
python scripts\ingest_documents.py C:\path\to\pdfs
```

- Ingestion works with **either** provider — FastEmbed needs no external daemon; Ollama requires Ollama running.
- The KB currently holds **519 chunks** (5 documents; Semester Rules OCR = 75 pages).
  - **Semester Rules are OCR-derived and were manually verified** (Step 18A); its registry entry is still marked `verification_status: pending` — flip it to `verified` only after your own review of the audit trail.
- **Backup before re-ingesting:** knowledge_base/reports/ contains migration backups; `chroma_service.update_metadata()` backs up IDs+metadata — see `scripts/update_knowledge_metadata.py` for the `--backup-path` semantics. Re-ingestion is idempotent for unchanged IDs.

## 6. Development startup

Two terminals:

```powershell
# terminal 1 — backend (auto-reload; logs at INFO)
cd C:\Stoic\pu-gpt\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --port 8000 --reload

# terminal 2 — frontend (Vite dev server; proxies /api -> http://localhost:8000)
cd C:\Stoic\pu-gpt\frontend
npm run dev     # http://localhost:5173
```

## 7. Production startup (single worker — READ FIRST)

```powershell
cd C:\Stoic\pu-gpt\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --port 8000
```

- **No `--reload`** (hot-reloading pulls new code mid-request and thrashes ChromaDB).
- **Exactly one worker** (`--workers 1` if you must say it). Why:
  1. ChromaDB's persistent client is not safe for concurrent readers with the same client instance — PUGPT keeps a module-level singleton.
  2. The per-IP limiter (Step 25) is in-memory — each worker would get its own bucket.
  3. The global daily cap (Step 26) is multi-process-safe (SQLite), but the app is still single-worker by design.
  4. If you ever scale out, replace both limiters with a shared store (e.g. Redis) and put a real shared vector store in place. Not yet needed.
- Serve `frontend/dist/` statically via your web server; proxy `/api` to `http://127.0.0.1:8000` (see [deployment.md](deployment.md)).

## 8. Health checking

```powershell
curl http://localhost:8000/api/health
```

Response (never contains keys):

```json
{
  "status": "ok",
  "service": "PUGPT - University of the Punjab GPT",
  "generation": {"provider": "groq", "model": "openai/gpt-oss-120b", "api_key_configured": true},
  "web_search": {"enabled": false, "provider": "tavily", "api_key_configured": false, "on_recency": true},
  "ollama": {"reachable": true, "model": "llama3.2:latest", "model_available": true, "error": null}
}
```

- `/api/health` is **not** rate limited and always returns HTTP 200 (callers distinguish backend-down from Ollama-down).
- There is **no** bare `/health` route — every probe path must be `/api/health` (misrouted probes were observed hammering the app with 404s).

## 9. Rate limiting (what operators will see)

| Layer | Default | Storage | Endpoint scope |
|---|---|---|---|
| Per-IP fixed window | 10 req / 60 s / client IP | in-memory (resets on restart) | `POST /api/chat` only |
| Global daily cap | 500 req / UTC calendar day | SQLite `data/global_rate_limit.sqlite3` (persists) | `POST /api/chat` only |

- Both run **before** RAG/embedding/Groq/Tavily — abuse cannot burn provider quota.
- Per-IP exceed → `HTTP 429` `{"error":"rate_limit_exceeded","retry_after":N}` + `Retry-After: N`.
- Global cap exceed → `HTTP 429` `{"error":"daily_limit_exceeded",…}` + `Retry-After: N≤86400`.
- Counter storage failure → `HTTP 503` `{"error":"rate_limit_unavailable",…}` — **fail-closed**, never silently unprotected.
- `/api/health` is exempt.
- Single worker by design (see §7). Nginx-level limiting is recommended on top — see [deployment.md](deployment.md).

## 10. ChromaDB persistence & backups

- Lives at `backend/data/chroma/` (~14 MB / 519 chunks / 294 files). Back it up (or rebuild it) before any reset/re-ingest.
- Rebuild from scratch: `python scripts\ingest_documents.py C:\path\to\the\verified\pdfs` (validator first, human review, then ingest; uses the current `EMBEDDING_PROVIDER`).
- Pre-migration backups used by Step 18/19 are in `knowledge_base/reports/` (`chroma_metadata_backup_step16.json`, `chroma_metadata_backup_step19.json`).
- Do not hand-edit files under `backend/data/chroma/`.

## 11. Security warnings (READ BEFORE GOING PUBLIC)

1. **Secrets**: `backend/.env` is git-ignored, single-copy. Anyone who gets `GROQ_API_KEY`/`TAVILY_API_KEY` can drain your quotas. Treat them like passwords. Never paste into tickets/chat/logs.
2. **CORS**: set `CORS_ORIGINS` to your exact production origin; never `*` with `allow_credentials=True`.
3. **No auth**: anyone can ask. The only controls are the two rate limiters. Choose `MAX_CHAT_REQUESTS_PER_DAY` conservatively for your free quotas.
4. **Ollama stays private** — localhost-only, never exposed, no firewall exceptions.
5. **Fail-closed is intentional**: if the rate-limit SQLite file becomes unavailable, chat refuses (503) rather than allow unlimited traffic.
6. **Source citations**: PUGPT cites PDF pages but cannot prove a web fact is current — when in doubt, show the citation rather than trusting it.
7. **Verifier refuses rather than guesses** — a refusal ("I could not find a reliable answer…") is correct behavior when evidence is missing; do not "fix" it by weakening the verifier or the 0.5 threshold.
8. **SR (Semester Rules) registry entry is `verification_status: pending`** — flip it only after your review (OCR source retained for audit).
9. **Logs**: the app logs at INFO (no prompts/keys/answers/URLs of users; per-refusal anonymous lines only). Wire stdout to journald/docker logs; do not add request-body logging.
10. **Schema stray**: backend must run as one process; never run `--workers N` without replacing both limiters.

## 12. Project layout

```
backend/            FastAPI app, services, ingestion + test suites
  app/api/routes/   /api/chat, /api/health
  app/services/     rag, chroma, embeddings, generation(groq/ollama), web_search, rate limiters
  scripts/          ingest/validate utilities + regression suites (mocked by default)
frontend/           React+Vite+Tailwind chat UI
knowledge_base/     official PU documents, registry, reports (backups/audits)
```

## 13. Running tests

- All suites are **mocked by default** and safe (no live Groq/Tavily calls; Ollama-dependent tests are skipped via `LLM_PROVIDER=ollama` + unreachable `OLLAMA_BASE_URL` when local models are unavailable).
- Example: `cd backend && .\venv\Scripts\python.exe scripts\test_web_search.py`
- Some suites (e.g., `test_chat_api.py`, `test_step21_reliability.py`, `test_answer_quality.py`) contain sections that require a live local Ollama embedding model and are **skipped/partial without one** — that is intentional.
- No pytest is required — every suite is a plain Python script runnable with `python scripts/test_*.py`. `requirements-dev.txt` adds pytest/ruff (unverified pins; re-check after first fresh dev install).

See [deployment.md](deployment.md) for the planned production architecture.
