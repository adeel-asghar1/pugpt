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

import json
import logging
import threading
import time

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
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.base_url = settings.groq_base_url if base_url is None else base_url
        self.model = settings.groq_chat_model if model is None else model
        self.timeout = timeout if timeout is not None else settings.groq_timeout
        self.max_tokens = settings.groq_max_tokens if max_tokens is None else max_tokens
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) \
                or self.max_tokens < 1 or self.max_tokens > 65536:
            raise ValueError("max_tokens must be an integer between 1 and 65536.")
        self.transport = transport
        # Normalize once; every request goes to exactly one /chat/completions.
        self.completions_url = self._completions_url(self.base_url)
        # One lazily-created HTTPX client per service instance, shared across
        # concurrent requests (sync route handlers run in a threadpool).
        # Created under a lock so races cannot produce duplicate clients.
        self._shared_client: httpx.Client | None = None
        self._client_lock = threading.Lock()

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

    def _get_client(self) -> httpx.Client:
        """Return the shared client, creating it on first use.

        httpx.Client pools keep-alive connections and is thread-safe for
        concurrent sync use; creating one per request forfeits the pooling
        benefit (a TCP+TLS handshake to api.groq.com per generation).
        Per-request timeouts are passed on each call, so sharing the client
        never weakens per-call timeout semantics.
        """
        with self._client_lock:
            if self._shared_client is None or self._shared_client.is_closed:
                self._shared_client = httpx.Client(
                    timeout=self.timeout,
                    transport=self.transport,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
            return self._shared_client

    def close(self) -> None:
        """Close the shared client and drop the reference (process shutdown/test cleanup)."""
        with self._client_lock:
            client, self._shared_client = self._shared_client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("Groq client close failed: %s", type(exc).__name__)

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
            # OpenAI-compatible ceiling on output tokens ("max_tokens" is
            # deprecated for this endpoint). The value is validated in
            # Settings/constructor (1-65536).
            "max_completion_tokens": self.max_tokens,
        }
        try:
            client = self._get_client()
            response = client.post(
                self.completions_url,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
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
            choice = choices[0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GroqResponseError("Groq returned an unexpected response format.") from exc

        # Diagnosability, privacy-conscious: numeric usage + provider finish
        # reason only — never the prompt, the answer or any key.
        usage = body.get("usage") if isinstance(body, dict) else None
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        logger.info(
            "groq usage: prompt_tokens=%s completion_tokens=%s total_tokens=%s finish_reason=%s",
            prompt_tokens, completion_tokens, total_tokens, finish_reason,
        )

        if finish_reason == "length":
            # Truncated at the configured output-token bound — never present
            # a partial answer as complete. Fail closed to the caller's
            # existing error path (generic error answer, sources preserved).
            logger.warning(
                "Groq output truncated at max_completion_tokens=%d; refusing partial answer",
                self.max_tokens,
            )
            raise GroqResponseError(
                "Groq output reached the configured token limit before completion."
            )

        content = (content or "").strip()
        if not content:
            raise GroqResponseError("Groq returned an empty response.")
        return content

    def stream_generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ):
        """Streaming variant of generate(). Exactly one HTTP request.

        Yields dict events:
          - {"delta": "text"} for each content chunk;
          - {"done": True, "finish_reason": ..., "ttft_ms": ...,
             "total_ms": ..., "completion_chars": ...} once at the end.

        Raises the same error taxonomy as generate(). Never returns an empty
        string, never logs/echoes the prompt, answer or key, never retries.
        The upstream HTTP response is closed on every exit path (including
        GeneratorExit from an aborting caller) via the context manager.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if not self.api_key.strip():
            raise GroqError(
                "Groq API key is not configured (GROQ_API_KEY is missing or empty)."
            )
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
            "temperature": 0,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
        }
        started = time.perf_counter()
        ttft_ms: float | None = None
        finish_reason: str | None = None
        completion_chars = 0
        done_marker = False
        try:
            client = self._get_client()
            with client.stream(
                "POST",
                self.completions_url,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            ) as response:
                if response.status_code != 200:
                    # Body is safe to read now; log a redacted excerpt only.
                    snippet = response.read().decode("utf-8", errors="replace")
                    snippet = " ".join(snippet.split())[:160]
                    if self.api_key and self.api_key in snippet:
                        snippet = snippet.replace(self.api_key, "[REDACTED]")
                    logger.warning(
                        "Groq stream failed: HTTP %s (provider %s)",
                        response.status_code, snippet,
                    )
                    raise _classify(response.status_code)(
                        f"Groq responded with HTTP {response.status_code}."
                    )
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
                    while "\n" in buffer:
                        line, _, buffer = buffer.partition("\n")
                        line = line.strip()
                        if not line or line.startswith(":"):
                            # blank separator / SSE keep-alive comment
                            continue
                        if not line.startswith("data:"):
                            # event:/id:/etc. — not used by Groq; ignore
                            continue
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            done_marker = True
                            continue
                        try:
                            event = json.loads(data)
                        except ValueError:
                            raise GroqResponseError(
                                "Groq streaming returned malformed data."
                            ) from None
                        choices = event.get("choices") if isinstance(event, dict) else None
                        if not choices:
                            # e.g. a usage-only tail chunk; nothing to do
                            continue
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - started) * 1000.0
                                logger.info("groq stream: ttft_ms=%.1f", ttft_ms)
                            completion_chars += len(content)
                            yield {"delta": content}
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr
                if buffer.strip():
                    # Unterminated trailing line — the stream broke off mid-event.
                    raise GroqResponseError("Groq streaming ended with an unterminated event.")
            if finish_reason is None and not done_marker:
                raise GroqResponseError("Groq streaming ended unexpectedly.")
            total_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "groq stream: total_ms=%.1f completion_chars=%d finish_reason=%s",
                total_ms, completion_chars, finish_reason,
            )
            yield {
                "done": True,
                "finish_reason": finish_reason,
                "ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
                "total_ms": total_ms,
                "completion_chars": completion_chars,
            }
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError(
                f"Groq did not respond within {effective_timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise GroqUpstreamError("Could not connect to Groq API endpoint.") from exc
        except httpx.HTTPError as exc:
            raise GroqUpstreamError(f"Groq request failed: {type(exc).__name__}") from exc


# Shared instance built from central configuration.
groq_service = GroqService()
