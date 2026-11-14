"""Application-level rate limiting for the public POST /api/chat endpoint.

Minimal, dependency-free, in-memory fixed-window limiter, keyed by client IP.

Design properties:
- Default: 10 requests per 60 seconds per client IP (configurable via
  RATE_LIMIT_CHAT / RATE_LIMIT_WINDOW_SECONDS).
- First request from a client opens its window; over-limit requests are
  rejected BEFORE the RAG/generation pipeline runs — this protects Groq and
  Tavily quota and our own compute.
- In-memory process state: counters reset on restart and are NOT shared
  between workers. PUGPT deliberately runs a single worker (ChromaDB
  constraint). If you ever scale horizontally, replace this with a shared
  store (e.g. Redis) or your quota protection will be per-instance.
- Client IP resolution: request.client.host (the TCP peer) by default —
  unspoofable from outside. X-Forwarded-For is honored ONLY when
  RATE_LIMIT_TRUST_PROXY=true AND PUGPT runs behind a reverse proxy that
  guarantees a correct X-Forwarded-For for the real client. When trusted,
  the RIGHTMOST entry is used: that is the address the closest trusted
  proxy observed. Left-most entries may be client-supplied junk. Anything
  unparseable falls back to the peer host.
- Invalid configuration never crashes the app: values are clamped to the
  safe defaults (10 / 60s) with a single WARNING at first use per process.
- No headers, prompts, keys or personal data are logged. Limited clients get
  a generic refusal log line only.
"""

import ipaddress
import logging
import math
import threading
import time

from fastapi import Request

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUESTS = 10
DEFAULT_WINDOW_SECONDS = 60.0

_limits_warned = False


def _clamped_limits() -> tuple[int, float]:
    """Read configured limits; fall back to safe defaults on invalid values."""
    global _limits_warned
    max_requests = getattr(settings, "rate_limit_chat", DEFAULT_MAX_REQUESTS)
    window = getattr(settings, "rate_limit_window_seconds", DEFAULT_WINDOW_SECONDS)
    try:
        max_requests = int(max_requests)
    except (TypeError, ValueError, OverflowError):
        max_requests = 0
    try:
        window = float(window)
    except (TypeError, ValueError):
        window = 0.0
    if max_requests < 1 or not math.isfinite(window) or window < 1.0:
        if not _limits_warned:
            logger.warning(
                "Invalid rate-limit configuration (RATE_LIMIT_CHAT=%r, "
                "RATE_LIMIT_WINDOW_SECONDS=%r); falling back to %d per %.0fs.",
                getattr(settings, "rate_limit_chat", None),
                getattr(settings, "rate_limit_window_seconds", None),
                DEFAULT_MAX_REQUESTS,
                DEFAULT_WINDOW_SECONDS,
            )
            _limits_warned = True
        return DEFAULT_MAX_REQUESTS, DEFAULT_WINDOW_SECONDS
    return max_requests, window


class RateLimiter:
    """Thread-safe fixed-window limiter (one window per key at a time)."""

    def __init__(self, clock=None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[int, float]] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds). retry_after is 0 if allowed."""
        max_requests, window = _clamped_limits()
        with self._lock:
            now = self._clock()
            self._sweep(now, window)
            bucket = self._buckets.get(key)
            if bucket is None or (now - bucket[1]) >= window:
                self._buckets[key] = (1, now)
                return True, 0.0
            count, start = bucket
            if count < max_requests:
                self._buckets[key] = (count + 1, start)
                return True, 0.0
            return False, max(0.0, window - (now - start))

    def _sweep(self, now: float, window: float) -> None:
        """Drop fully-expired buckets so memory stays bounded (every 256+ keys)."""
        if len(self._buckets) < 256:
            return
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < window}

    def reset(self) -> None:
        """Clear all counters. Used by tests."""
        with self._lock:
            self._buckets.clear()


def client_ip(request: Request) -> str:
    """Resolve the client IP for rate limiting.

    Spoofing-resistant by default: the TCP peer is used and any
    X-Forwarded-For the client sent is ignored. Only when trusted-proxy
    mode is explicitly enabled do we read X-Forwarded-For — and then the
    RIGHTMOST entry (the address verified by the closest trusted proxy),
    never client-controlled left entries. Unparseable input falls back
    to the peer.
    """
    peer = request.client.host if request.client and request.client.host else "unknown"
    if not getattr(settings, "rate_limit_trust_proxy", False):
        return peer
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return peer
    candidate = xff.split(",")[-1].strip()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer


# Shared instance used by the /api/chat route.
rate_limiter = RateLimiter()
