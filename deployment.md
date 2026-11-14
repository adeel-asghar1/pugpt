

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
