from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory, resolved relative to this file so paths work
# no matter which directory the server is started from.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration. Values can be overridden via a .env file."""

    app_name: str = "PUGPT - University of the Punjab GPT"

    # Comma-separated list of allowed frontend origins (Vite dev server by default).
    cors_origins: str = "http://localhost:5173"

    # Ollama (local model server)
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:latest"
    ollama_embed_model: str = "nomic-embed-text:latest"
    # Timeout (seconds) for any HTTP request to Ollama (pings, embeddings).
    ollama_timeout: float = 30.0
    # Timeout (seconds) for answer generation; CPU generation can outlast the ping timeout.
    ollama_chat_timeout: float = 120.0

    # Groq hosted LLM (OpenAI-compatible chat completions) — used for answer
    # GENERATION only; embeddings stay on local Ollama nomic-embed-text.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_chat_model: str = "openai/gpt-oss-120b"
    # Upper bound on output tokens for one answer. gpt-oss-120b supports up to
    # 65,536 output tokens per completion; 1024 comfortably covers a complete
    # grounded RAG answer (these are typically 300-700 tokens). If a response
    # is truncated at this limit (finish_reason="length"), the answer is
    # treated as unusable — it is never presented as complete.
    groq_max_tokens: int = 1024
    groq_timeout: float = 120.0
    # Generation provider: "auto" (default) -> "groq" when GROQ_API_KEY is set,
    # otherwise "ollama". Set to "groq" or "ollama" to force one explicitly.
    # There is NO automatic fallback between providers.
    llm_provider: str = "auto"

    # ChromaDB persistent storage on disk (configurable via env var)
    chroma_persist_path: Path = BASE_DIR / "data" / "chroma"
    # Active collection. Default is the FastEmbed re-index (created from the
    # same sources with fastembed nomic-ai/nomic-embed-text-v1.5-Q).
    # Rollback: set CHROMA_COLLECTION=pugpt_knowledge to use the original
    # Ollama-measured embeddings (519 records). Do NOT mix provider and
    # collection — they must match.
    chroma_collection: str = "pugpt_knowledge_fastembed_test"

    # Embedding provider for production retrieval. "fastembed" (default) or
    # "ollama" (rollback). Vector spaces differ between them — the collection
    # above was built with the FastEmbed model below; do NOT mix.
    embedding_provider: str = "fastembed"
    fastembed_model_name: str = "nomic-ai/nomic-embed-text-v1.5-Q"

    # Tavily official-PU live web search — FALLBACK ONLY. PUGPT never searches
    # the web for ordinary PDF-answerable questions; search happens only when
    # ChromaDB has no relevant PDF context, or the question explicitly asks for
    # current/latest information. DISABLED until WEB_SEARCH_ENABLED=true.
    # Basic search depth, at most web_search_max_results results, 1 credit per
    # fallback answer on the free tier (1,000 credits/month, no card required).
    # There are NO automatic retries; quota exhaustion degrades gracefully.
    web_search_enabled: bool = False
    web_search_provider: str = "tavily"
    # Allow live search for questions explicitly asking for current/latest
    # information even when PDF context exists (consumes Tavily quota).
    web_search_on_recency: bool = True
    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    # Timeout (seconds) for Tavily search requests.
    web_search_timeout: float = 15.0
    web_search_max_results: int = 5
    # Comma-separated trusted PU host suffixes. Only hosts exactly equal to a
    # suffix or ending with ".<suffix>" are accepted (enforced server-side
    # regardless of what the provider returns).
    web_search_allowed_suffixes: str = "pu.edu.pk"

    # Application-level rate limiting for POST /api/chat (per client IP,
    # in-memory fixed window, single worker). Health and other endpoints are
    # not limited. On restart counters reset; on multiple workers each has
    # its own counters (PUGPT is single-worker by design).
    rate_limit_chat: int = 10              # max requests per window per client IP
    rate_limit_window_seconds: float = 60.0  # fixed window size in seconds
    # Set true ONLY when PUGPT is behind a reverse proxy that guarantees a
    # correct X-Forwarded-For for the real client IP. When enabled, the
    # rightmost X-Forwarded-For entry is trusted; otherwise XFF is ignored.
    rate_limit_trust_proxy: bool = False

    # Global daily chat-request limit (application-wide, per UTC calendar
    # day), stored in SQLite so it survives restarts. Persisted rows contain
    # only the UTC date and an accepted-count — never prompts, IPs, or any
    # personal data. On storage failure the API fails CLOSED (requests are
    # refused) rather than disabling the cap.
    max_chat_requests_per_day: int = 500     # per UTC calendar day, all clients combined
    # Where the SQLite counter lives (single worker; file is created on first use).
    rate_limit_global_db_path: Path = BASE_DIR / "data" / "global_rate_limit.sqlite3"

    # Retrieval
    retrieval_top_k: int = 4

    # RAG configuration.
    # Cosine distance from ChromaDB (1 - cosine similarity); lower = more similar.
    # Chunks whose distance is GREATER than this threshold are treated as
    # weakly relevant and discarded — 0.5 is a conservative starting point for
    # nomic-embed-text and should be tuned with real usage.
    rag_top_k: int = 5
    rag_distance_threshold: float = 0.5

    # Text chunking (approximate characters)
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Ingestion: chunks embedded per Ollama /api/embed request
    ingestion_batch_size: int = 16

    @field_validator("groq_max_tokens", mode="before")
    @classmethod
    def _validate_groq_max_tokens(cls, v):
        # 65536 = documented maximum output for openai/gpt-oss-120b on Groq.
        if isinstance(v, bool):
            raise ValueError("GROQ_MAX_TOKENS must be an integer.")
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError("GROQ_MAX_TOKENS must be an integer.")
            v = int(v)
        if isinstance(v, str):
            try:
                v = int(v.strip())
            except ValueError:
                raise ValueError("GROQ_MAX_TOKENS must be an integer.") from None
        if not isinstance(v, int):
            raise ValueError("GROQ_MAX_TOKENS must be an integer.")
        if v < 1 or v > 65536:
            raise ValueError("GROQ_MAX_TOKENS must be between 1 and 65536.")
        return v

    @field_validator("groq_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _strip_key_outer_whitespace(cls, v):
        # Render's dashboard stores env vars verbatim; pasted keys can carry a
        # trailing newline. Normalize OUTER whitespace only at the boundary —
        # embedded characters are rejected by the service layer, and a
        # leading "Bearer " is deliberately NOT removed (reported, not guessed).
        return v.strip() if isinstance(v, str) else v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
