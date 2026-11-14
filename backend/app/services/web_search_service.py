"""WebSearchService: official University of the Punjab live web search via Tavily.

FALLBACK ONLY — RAGService calls this when ChromaDB has no relevant PDF
context, or the question explicitly asks for current/latest information.
PUGPT never searches the web for ordinary PDF-answerable questions.

Design contract:
- `search()` raises a small, explicit error taxonomy (WebSearchError family)
  on any failure. Callers (RAGService) catch broadly and degrade to PDF-only
  answers or the existing safe refusal — never an exception, never a retry.
- If TAVILY_API_KEY is missing, `search()` fails fast before touching the
  network. This module performs NO request of its own until called.
- The API key never appears in logs, exceptions or health output.
- Only results whose URL host is exactly a trusted suffix or a subdomain of
  one are returned — enforced server-side REGARDLESS of what the provider
  returns. Default suffix: pu.edu.pk.
- Publication dates are passed through verbatim from the provider (Tavily's
  estimate, beta). If absent, they stay absent — never invented.
- No retries. Quota exhaustion and rate limits surface as
  WebSearchRateLimitError (HTTP 429/432/433/434) and degrade to safe refusal.
- Logging is privacy-conscious: query length + truncated preview only; per
  result logs official host, relevance score and publication date; failures
  log WARNING with the HTTP status class. Never the key or headers.
"""

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Base class for web-search failures. Message is safe to log."""


class WebSearchAuthError(WebSearchError):
    """HTTP 401/403 — authentication or authorization failed."""


class WebSearchRateLimitError(WebSearchError):
    """HTTP 429/432/433/434 — rate limit or plan/quota exhausted. Do not retry."""


class WebSearchUpstreamError(WebSearchError):
    """HTTP 5xx or transport failure — provider-side problem."""


class WebSearchTimeoutError(WebSearchError):
    """The request exceeded the configured timeout."""


class WebSearchResponseError(WebSearchError):
    """HTTP 200 with an unusable or empty body (never includes the key)."""


def _classify(status_code: int) -> type[WebSearchError]:
    if status_code in (401, 403):
        return WebSearchAuthError
    if status_code in (429, 432, 433, 434):
        return WebSearchRateLimitError
    if status_code >= 500:
        return WebSearchUpstreamError
    return WebSearchError


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    host: str
    content: str
    score: float | None = None
    published_date: str | None = None


@dataclass(frozen=True)
class WebSearchResult:
    results: list[WebSource] = field(default_factory=list)
    response_time: float | None = None
    credits_used: int | None = None


def validated_host(url: str, allowed_suffixes: list[str]) -> str | None:
    """Return the URL's lowercase host iff it is trusted, else None.

    Accepts: exact suffix (pu.edu.pk) or any subdomain (*.pu.edu.pk).
    Rejects: lookalikes (pu.edu.pk.evil.com), sibling labels (notpu.edu.pk),
    non-HTTP(S) schemes, missing/invalid URLs, unparseable input.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return None
    for suffix in allowed_suffixes:
        if host == suffix or host.endswith("." + suffix):
            return host
    return None


class WebSearchService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
        allowed_suffixes: str | None = None,
        transport: "httpx.BaseTransport | None" = None,
    ) -> None:
        self.api_key = settings.tavily_api_key if api_key is None else api_key
        self.base_url = settings.tavily_base_url if base_url is None else base_url
        self.timeout = timeout if timeout is not None else settings.web_search_timeout
        self.max_results = max(
            1, max_results if max_results is not None else settings.web_search_max_results
        )
        raw_suffixes = (
            settings.web_search_allowed_suffixes
            if allowed_suffixes is None
            else allowed_suffixes
        )
        self.allowed_suffixes = [
            s.strip().lower().rstrip(".") for s in str(raw_suffixes).split(",") if s.strip()
        ] or ["pu.edu.pk"]
        self.transport = transport
        # Provider-side domain filter is a hint; server-side validation governs
        # trust. include_domains_mode stays at its default ("restrict").
        self.search_url = self._search_url(self.base_url)

    @staticmethod
    def _search_url(raw_base_url: str) -> str:
        """Build the exact /search URL from either configured form:

        - https://api.tavily.com
        - https://api.tavily.com/
        - https://api.tavily.com/search
        - any repeated .../search[/search...] suffix
        """
        url = (raw_base_url or "").strip().rstrip("/")
        marker = "/search"
        while url.lower().endswith(marker):
            url = url[: -len(marker)].rstrip("/")
        return f"{url}{marker}"

    def _client(self, effective_timeout: float) -> httpx.Client:
        return httpx.Client(timeout=effective_timeout, transport=self.transport)

    @property
    def configured(self) -> bool:
        return bool((self.api_key or "").strip())

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> WebSearchResult:
        """Search official PU sources. Raises WebSearchError subclasses on failure.

        Never returns unofficial results; never retries; never invents dates.
        """
        question = (query or "").strip()
        if not question:
            raise ValueError("Query must not be empty.")

        if not self.configured:
            # Fail fast, before touching the network. No key -> no request.
            raise WebSearchError(
                "Tavily API key is not configured (TAVILY_API_KEY is missing or empty)."
            )

        effective_timeout = timeout if timeout is not None else self.timeout
        limit = max(1, max_results if max_results is not None else self.max_results)
        payload: dict[str, Any] = {
            "query": question,
            "search_depth": "basic",  # 1 credit/search; balanced relevance
            "max_results": limit,
            "include_answer": False,   # PUGPT generates its own grounded answer
            "include_raw_content": False,
            "include_images": False,
            "include_domains": list(self.allowed_suffixes),
            "include_published_date": True,
            "include_usage": True,
            "safe_search": True,
            "country": "pakistan",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with self._client(effective_timeout) as client:
                response = client.post(self.search_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise WebSearchTimeoutError(
                f"Tavily did not respond within {effective_timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise WebSearchUpstreamError("Could not connect to the Tavily API endpoint.") from exc
        except httpx.HTTPError as exc:
            raise WebSearchUpstreamError(
                f"Tavily request failed: {type(exc).__name__}"
            ) from exc

        if response.status_code != 200:
            exc_type = _classify(response.status_code)
            snippet = (response.text or "").replace("\n", " ")[:160]
            if self.api_key and self.api_key in snippet:
                snippet = snippet.replace(self.api_key, "[REDACTED]")
            logger.warning(
                "Tavily search failed: HTTP %s (provider %s)", response.status_code, snippet
            )
            raise exc_type(f"Tavily responded with HTTP {response.status_code}.")

        try:
            body = response.json()
            raw_results = body.get("results", []) if isinstance(body, dict) else []
            if not isinstance(raw_results, list):
                raise TypeError("results is not a list")
        except (ValueError, TypeError, AttributeError) as exc:
            raise WebSearchResponseError("Tavily returned an unexpected response format.") from exc

        results: list[WebSource] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            host = validated_host(url, self.allowed_suffixes)
            if host is None:
                # Unofficial / lookalike / malformed URL — rejected server-side.
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            published = item.get("published_date")
            published_date = str(published).strip() if published else None
            title = str(item.get("title") or "").strip() or host
            score = item.get("score")
            results.append(
                WebSource(
                    title=title,
                    url=url,
                    host=host,
                    content=content,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    published_date=published_date or None,
                )
            )

        # Diagnosability, privacy-conscious: query length + a short truncated
        # preview only (never the full question, key or headers).
        logger.info(
            "tavily search ok: query_len=%d query_preview=%r results_received=%d",
            len(question),
            question[:32],
            len(raw_results),
        )
        for index, source in enumerate(results):
            logger.info(
                "tavily result[%d]: host=%s score=%s published=%s",
                index,
                source.host,
                source.score,
                source.published_date,
            )

        usage = body.get("usage") if isinstance(body, dict) else None
        credits = usage.get("credits") if isinstance(usage, dict) else None
        response_time = body.get("response_time") if isinstance(body, dict) else None
        if credits is not None or response_time is not None:
            logger.info(
                "tavily usage: credits=%s response_time=%s",
                credits,
                response_time,
            )
        return WebSearchResult(
            results=results,
            response_time=float(response_time) if isinstance(response_time, (int, float)) else None,
            credits_used=credits if isinstance(credits, int) else None,
        )


# Shared instance built from central configuration.
web_search_service = WebSearchService()
