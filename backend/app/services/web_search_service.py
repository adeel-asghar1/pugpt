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
import re
import threading
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


_MONTH_TOKENS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)


_PUGPT_STOPWORDS = frozenset(
    "a an the and or but of in on at to for with from by as is are was were be been being am "
    "do does did doing have has had having i me my we our you your he she his her it its they them "
    "their this that these those there here what which who whom whose when where why how all any "
    "both each few more most other some such no nor not only own same so than too very can will "
    "just should now also then into over under again further once about above below between "
    "through during before after against would could should must may might must shall "
    "per via university pu official website ask asked tell give know want need find include "
    "mention describe explain summarize summary please provide latest current currently recent "
    "recently updated today now newly pu punjab".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _norm_token(token: str) -> str:
    """Lowercase, strip a leading apostrophe-s, then strip at most one trailing
    inflection (ies/ing/ed/es/s/ly) with a minimum stem length of 3."""
    if token.endswith("'s"):
        token = token[:-2]
    for suffix in ("ies", "ing", "ed", "s", "ly", "es"):
        stem = token[: -len(suffix)]
        if token.endswith(suffix) and len(stem) >= 3:
            return stem
    return token


def _split_token(raw: str):
    """Yield every hyphenated sub-token from a raw whitespace token."""
    return [p for p in raw.split("-") if p]


def _token_set(text) -> set:
    tokens: set = set()
    for raw in _WORD_RE.findall((text or "").lower()):
        for part in _split_token(raw):
            token = _norm_token(part)
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _question_content_terms(question: str) -> list[str]:
    """Distinct content terms (non-stopword, non-date, non-month, len>=3)."""
    terms: list[str] = []
    for raw in _WORD_RE.findall((question or "").lower()):
        for part in _split_token(raw):
            token = _norm_token(part)
            if len(token) < 3:
                continue
            if token in _PUGPT_STOPWORDS:
                continue
            if token in _MONTH_TOKENS:
                continue  # months never contribute to topical relevance
            if not any(ch.isalpha() for ch in token):
                continue  # date-like / pure-numeric tokens do not count
            if token not in terms:
                terms.append(token)
    return terms


def _web_relevant(source: "WebSource", question: str) -> tuple[bool, int, int]:
    """Conservative lexical relevance gate.

    Returns (relevant, hits, total_terms). Rejects unrelated official-PU pages.
    Pure date/month-only overlap never passes; generic PU vocabulary alone never passes.
    """
    terms = _question_content_terms(question)
    total = len(terms)
    if total == 0:
        # No content signal to judge against: do not attach dubiously-claimed web evidence.
        return False, 0, 0
    evidence_text = f"{source.title or ''} {source.content or ''}"
    ev_tokens = _token_set(evidence_text)
    hits = sum(1 for term in terms if term in ev_tokens)
    if total == 1:
        return hits >= 1, hits, total
    return hits >= 2, hits, total


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
        # One lazily-created HTTPX client per service instance, shared across
        # concurrent requests. Created under a lock so races cannot produce
        # duplicate clients; closed by the app's shutdown lifespan/test cleanup.
        # httpx.Client pools keep-alive connections and is thread-safe for
        # concurrent sync use — a TCP+TLS handshake per search is avoided.
        self._shared_client: httpx.Client | None = None
        self._client_lock = threading.Lock()

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

    def _get_client(self) -> httpx.Client:
        """Return the shared client, creating it on first use.

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
                logger.warning("Tavily client close failed: %s", type(exc).__name__)

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
            client = self._get_client()
            response = client.post(
                self.search_url,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
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
        rejected_relevance = 0
        rejected_domain_content = 0
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            host = validated_host(url, self.allowed_suffixes)
            if host is None:
                rejected_domain_content += 1
                # Unofficial / lookalike / malformed URL — rejected server-side.
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                rejected_domain_content += 1
                # Empty content — rejected.
                continue
            # Query-relevance gate: drop official-but-off-topic pages before they
            # reach the LLM or citations (e.g. webinars or syllabi for a
            # closure question). Conservative; requires >=2 substantive hits
            # (or the single term of a one-term question).
            candidate = WebSource(
                title=str(item.get("title") or "").strip() or host,
                url=url,
                host=host,
                content=content,
                score=(float(item["score"]) if isinstance(item.get("score"), (int, float)) else None),
                published_date=str(item.get("published_date") or "").strip() or None,
            )
            relevant, hits, total_terms = _web_relevant(candidate, question)
            if not relevant:
                rejected_relevance += 1
                continue
            results.append(candidate)

        # Diagnosability, privacy-conscious: query length + a short truncated
        # preview only (never the full question, key or headers).
        logger.info(
            "tavily search ok: query_len=%d query_preview=%r results_received=%d "
            "domain_or_content_rejected=%d relevance_rejected=%d retained=%d",
            len(question),
            question[:32],
            len(raw_results),
            rejected_domain_content,
            rejected_relevance,
            len(results),
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
