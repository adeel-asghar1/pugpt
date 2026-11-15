"""GenerationService: single entry point for answer generation.

Dispatches to one of two providers, chosen explicitly by configuration
(LLM_PROVIDER):

- "auto"   (default) -> "groq" when GROQ_API_KEY is configured,
                        otherwise "ollama"
- "groq"             -> Groq hosted OpenAI-compatible API
- "ollama"           -> local Ollama generation (pre-Groq behavior)

Deliberate properties:
- NO automatic fallback between providers. A Groq failure never silently
  falls back to slow local generation; the caller gets a clean error and
  the existing safe user-facing message.
- Embeddings are NOT touched here; they always stay on Ollama
  nomic-embed-text via embedding_service.
- API keys never appear in logs, health output or exception messages.
"""

import logging

from app.config import settings
from app.services.groq_service import GroqService, groq_service
from app.services.ollama_service import OllamaService, ollama_service

logger = logging.getLogger(__name__)

_PROVIDERS = {"groq", "ollama"}


class GenerationConfigError(Exception):
    """Invalid provider configuration (e.g. unknown LLM_PROVIDER value)."""


class GenerationService:
    def __init__(
        self,
        provider: str | None = None,
        groq: GroqService | None = None,
        ollama: OllamaService | None = None,
    ) -> None:
        # Explicit argument wins over settings (used by tests).
        self._provider_override = provider
        self.groq = groq or groq_service
        self.ollama = ollama or ollama_service

    def resolved_provider(self) -> str:
        """Resolve the provider at CALL time (settings are re-read each call)."""
        raw = (self._provider_override or settings.llm_provider or "auto").strip().lower()
        if raw in _PROVIDERS:
            return raw
        if raw == "auto":
            # Read settings live so config-driven behavior is observable
            # (e.g. when LLM_PROVIDER/GROQ_API_KEY change at runtime).
            return "groq" if settings.groq_api_key.strip() else "ollama"
        raise GenerationConfigError(
            f"Unknown LLM_PROVIDER '{raw}'. Expected one of: auto, groq, ollama."
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        provider = self.resolved_provider()
        if provider == "groq":
            return self.groq.generate(prompt, system=system, timeout=timeout)
        return self.ollama.generate(prompt, system=system, timeout=timeout)

    def stream_generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ):
        """Streaming counterpart of generate(). Groq only.

        Streaming is a Groq-only capability; Ollama streaming is deliberately
        not implemented (slow CPU generation has no payoff here). There is
        NO fallback between providers — a misconfigured provider raises
        GenerationConfigError instead of silently switching.
        """
        provider = self.resolved_provider()
        if provider == "groq":
            return self.groq.stream_generate(prompt, system=system, timeout=timeout)
        raise GenerationConfigError(
            "Streaming is supported only by the Groq provider "
            f"(resolved LLM_PROVIDER={provider!r})."
        )


# Shared instance built from central configuration.
generation_service = GenerationService()
