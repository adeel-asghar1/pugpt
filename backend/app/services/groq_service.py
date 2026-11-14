"""GroqService: answer generation via Groq's hosted OpenAI-compatible API.

Only GENERATION goes through Groq — embeddings are intentionally kept on
local Ollama nomic-embed-text.

Design contract with the rest of PUGPT:
- `generate()` raises a small, explicit error taxonomy on any failure
  (GroqUnavailableError family). Callers (RAGService) turn those into the
  existing generic error answer — no stack traces, no provider internals,
  and NEVER the API key, in client responses.
- The API key is never placed in logs, exception messages, health output
  or test artifacts. Errors carry only the HTTP status class or a fixed code.
- If GROQ_API_KEY is missing, `generate()` fails fast with a clear
  configuration error without touching the network.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class GroqError(Exception):
    """Base class for Groq generation failures. Message is safe to log."""


class GroqAuthError(GroqError):
    """HTTP 401/403 — authentication or authorization failed."""


class GroqRateLimitError(GroqError):
    """HTTP 429 — provider rate limit hit; do not retry silently."""


class GroqUpstreamError(GroqError):
    """HTTP 5xx — provider-side failure."""


class GroqTimeoutError(GroqError):
    """The request exceeded the configured timeout."""


class GroqResponseError(GroqError):
    """HTTP 2xx/4xx with an unusable or empty body (never includes the key)."""


def _classify(status_code: int) -> type[GroqError]:
    if status_code in (401, 403):
        return GroqAuthError
    if status_code == 429:
        return GroqRateLimitError
    if status_code >= 500:
        return GroqUpstreamError
    return GroqError


class GroqService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: "httpx.BaseTransport | None" = None,
    ) -> None:
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.base_url = settings.groq_base_url if base_url is None else base_url
        self.model = settings.groq_chat_model if model is None else model
        self.timeout = timeout if timeout is not None else settings.groq_timeout
        self.transport = transport
        # Normalize once; every request goes to exactly one /chat/completions.
        self.completions_url = self._completions_url(self.base_url)

    @staticmethod
    def _completions_url(raw_base_url: str) -> str:
        """Build the exact chat-completions URL from either configured form:

        - https://api.groq.com/openai/v1
        - https://api.groq.com/openai/v1/
        - https://api.groq.com/openai/v1/chat/completions
        - https://api.groq.com/openai/v1/chat/completions/
        - any repeated .../chat/completions[/chat/completions...] suffix

        The result always ends in exactly one '/chat/completions'.
        """
        url = (raw_base_url or "").strip().rstrip("/")
        marker = "/chat/completions"
        while url.lower().endswith(marker):
            url = url[: -len(marker)].rstrip("/")
        return f"{url}{marker}"

    def _client(self, effective_timeout: float) -> httpx.Client:
        return httpx.Client(timeout=effective_timeout, transport=self.transport)

    @property
    def configured(self) -> bool:
        # Live view of configuration: built-in default instances always
        # reflect settings, explicit constructor overrides still work.
        return bool(self.api_key.strip())

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Generate a response. Raises GroqError subclasses on any failure.

        Never returns an empty string; never leaks the API key.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        if not self.api_key.strip():
            # Fail fast, before touching the network. No key -> no request.
            raise GroqError(
                "Groq API key is not configured (GROQ_API_KEY is missing or empty)."
            )

        # Outer whitespace/newlines (e.g. dashboard paste artifacts) are
        # normalized here so the header stays legal. Anything remaining that
        # is not printable ASCII can never produce a legal Authorization
        # header — fail fast with a generic, key-free message instead of
        # letting h11 reject the request client-side (LocalProtocolError).
        key = self.api_key.strip()
        if not all(33 <= ord(c) <= 126 for c in key):
            raise GroqError(
                "Groq API key is misconfigured (contains whitespace or invalid "
                "characters). Fix GROQ_API_KEY in the environment."
            )

        effective_timeout = timeout if timeout is not None else self.timeout
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,  # grounded, deterministic generation for RAG
        }
        try:
            with self._client(effective_timeout) as client:
                response = client.post(self.completions_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError(
                f"Groq did not respond within {effective_timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise GroqUpstreamError("Could not connect to Groq API endpoint.") from exc
        except httpx.HTTPError as exc:
            raise GroqUpstreamError(f"Groq request failed: {type(exc).__name__}") from exc

        if response.status_code != 200:
            exc_type = _classify(response.status_code)
            # Log only the status class/code and a short, non-secret excerpt.
            snippet = (response.text or "").replace("\n", " ")[:160]
            if self.api_key and self.api_key in snippet:
                snippet = snippet.replace(self.api_key, "[REDACTED]")
            logger.warning(
                "Groq generation failed: HTTP %s (provider %s)", response.status_code, snippet
            )
            raise exc_type(f"Groq responded with HTTP {response.status_code}.")

        try:
            body = response.json()
            choices = body["choices"]
            content = choices[0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GroqResponseError("Groq returned an unexpected response format.") from exc

        content = (content or "").strip()
        if not content:
            raise GroqResponseError("Groq returned an empty response.")
        return content


# Shared instance built from central configuration.
groq_service = GroqService()
