

---

## 18. GitHub Actions validation workflow (Step 7B — preparation only)

**Status: created, NOT executed.** A workflow at `.github/workflows/validate-docker.yml` validates the image on a Linux host without publishing anything or needing secrets:

- Trigger: `workflow_dispatch` (manual) and `push` to `main`.
- Builds `Dockerfile` on `ubuntu-latest` (no registry push).
- Starts a container on port 8080 with LLM_PROVIDER=groq and **no GROQ_API_KEY** — so no Groq call can happen; /api/health is key-free by design.
- Asserts `/api/health` → 200 with `embedding.provider=fastembed`, `dimension=768`, model `nomic-ai/nomic-embed-text-v1.5-Q`; Ollama must report unreachable.
- Asserts the packaged ChromaDB reports exactly **519 records** for `pugpt_knowledge_fastembed_test`.
- Runs a **retrieval-only** check (FastEmbed cosine search, 768-dim query) — no generation, no Groq/Tavily calls.
- Stops and removes the container in `if: always()` cleanup.

**Not verified locally:** Docker build is not validated on this host because `files.pythonhosted.org` is bandwidth/connection constrained here; the workflow is expected to work on a normal Linux runner (GitHub-hosted or equivalent), but this cannot be confirmed without a push. **Do NOT push this repository to a public source or registry.** Use a private repository only.

---

## 19. Automatic official-PU web fallback (Step 16)

**Status: implemented in code; OFF in production unless explicitly enabled.**

### Behaviour

1. Every question is answered from the PDF knowledge base first (hard 0.5 cosine-distance threshold, unchanged).
2. After retrieval, a conservative sufficiency heuristic is evaluated: sufficient only when >= 1 chunk is kept AND best ORIGINAL distance <= 0.35 AND combined retained text >= 600 characters. This is a heuristic, not proof of completeness. If it fails and web search is enabled, Tavily runs once (trigger = `pdf_missing` or `sparse_pdf`).
3. A question explicitly asking for current/latest information may still trigger a search even when PDF evidence is strong (trigger = `recency`, gated by `WEB_SEARCH_ON_RECENCY`).
4. In every case at most ONE Tavily request runs per chat; no retries.
5. Tavily is restricted to official `pu.edu.pk` sources, enforced server-side regardless of provider responses.
6. Combined evidence (some PDF + some web) is presented with both source types; the model attributes PDF figures to the PDF filename/session and website figures to pu.edu.pk. The structured citation panel lists them separately.
7. Tavily failure degrades to PDF-only or to the safe refusal.
8. Numeric-claim verification is unchanged; unsupported claims are refused.

### Where the switch lives

- `WEB_SEARCH_ENABLED` (default `false`) — master switch. Set `true` in the Render dashboard to enable the automatic fallback. Never needs to change locally by default.
- `TAVILY_API_KEY` — required for any web call; never committed, never logged.
- `WEB_SEARCH_ON_RECENCY` (default `true`) — current-info questions may search even with strong PDF evidence.
- `MAX_CHAT_REQUESTS_PER_DAY` (default 500) bounds worst-case Tavily credit burn to <= 500/day. Render Free's filesystem is ephemeral, so this counter is best-effort, not a hard cap.

### Quota note

Tavily free tier = 1,000 credits/month (basic depth, 1 credit per search). Automatic fallback on every sparse-evidence question costs 1 credit per question. Monitor usage in the Tavily dashboard; if you approach limits, set `WEB_SEARCH_ENABLED=false` to return to PDF-only answers — no code change.

### Rollback

Setting `WEB_SEARCH_ENABLED=false` restores exact pre-Step-16 retrieval/answer behaviour. The sufficiency heuristic's constants (`PDF_SUFFICIENT_BEST_DISTANCE=0.35`, `PDF_SUFFICIENT_MIN_CHARS=600`, `PDF_RELEVANCE_MIN_COVERAGE=0.34`, `PDF_RELEVANCE_MIN_SUBSTANTIVE_HITS=2`) live in `backend/app/services/rag_service.py` for tuning with real usage.

---

## 20. Limitations of the relevance gate (Steps 18C–18E)

The question-to-evidence relevance signal used for the official-PU web fallback is a **lexical** heuristic, not a semantic judge. It exists to catch substantially off-topic evidence (e.g. admission PDFs versus a question about campus closures), not to prove answer completeness.

Known limitations:
- **Single-term questions** whose lone content word is ubiquitous PU vocabulary (admission, admissions, merit, schedule, programme, programmes, session, dues) are classified insufficient regardless of any match — such evidence is not proof of relevance. A lone `fee` question is judged relevant only when the kept evidence carries a currency-anchored fee figure (e.g. `Rs. 15,000`).
- **Synonyms are not bridged** — "closure" vs "holiday", "application" vs "admission" will not cross-match.
- **Urdu-only questions yield zero English content terms**; they are deliberately treated as "relevance unavailable" (safe: no extra Tavily call, PDF-only answer). Mixed Urdu/English questions work off their English tokens only.
- **Generic PU vocabulary** ("admission", "merit", "programme", "schedule") can match admission-family questions regardless of whether the evidence is on point for the specific sub-topic asked.
- **The 0.34 coverage floor is a heuristic** and does not constitute proof of answer completeness. Distance and length gates (0.35 / 600 chars) still apply on top of it.
- Dates/numeric tokens ("2026", "2026-27") are excluded from substantive counting by design; a question consisting only of date tokens is treated as "relevance unavailable" (no extra Tavily call).

