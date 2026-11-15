import logging
import math
import time
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.services.rag_service import ERROR_ANSWER, rag_service
from app.services.rate_limiter import client_ip, rate_limiter
from app.services.global_rate_limiter import global_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _rate_limit_or_refuse(request: Request) -> JSONResponse | None:
    """Shared per-IP + daily-cap enforcement, identical for /chat and
    /chat/stream. Evaluated ONCE per request. Returns a refusal response,
    or None when the request may proceed."""
    # Rate limits FIRST — before any embedding/Groq/Tavily work.
    allowed, retry_after = rate_limiter.check(client_ip(request))
    if not allowed:
        wait = max(1, math.ceil(retry_after))
        logger.info("per-IP rate limit exceeded; refusing request (anonymous client)")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many requests. Please wait before sending another message.",
                "error": "rate_limit_exceeded",
                "retry_after": wait,
            },
            headers={"Retry-After": str(wait)},
        )
    # Global daily cap (UTC calendar day, persisted). Checked second so a
    # single misbehaving IP burns only its own allowance; per-IP buckets are
    # consumed before the daily counter — a request refused by the daily cap
    # may have consumed per-IP budget, which is intentional and harmless.
    daily_allowed, daily_retry_after = global_rate_limiter.try_consume()
    if daily_allowed is None:
        # Storage failure — fail CLOSED. Protection is never silently disabled.
        logger.error("global rate limiter unavailable; refusing request (fail-closed)")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "PUGPT is temporarily unavailable. Please try again shortly.",
                "error": "rate_limit_unavailable",
                "retry_after": 300,
            },
            headers={"Retry-After": "300"},
        )
    if not daily_allowed:
        wait = max(1, math.ceil(daily_retry_after))
        logger.info("daily chat limit reached; refusing request (anonymous client)")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "PUGPT has reached its daily question limit. Please try again after midnight UTC.",
                "error": "daily_limit_exceeded",
                "retry_after": wait,
            },
            headers={"Retry-After": str(wait)},
        )
    return None


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse | JSONResponse:
    """Answer a question about the University of the Punjab using RAG."""
    started = time.perf_counter()
    refusal = _rate_limit_or_refuse(request)
    if refusal is not None:
        return refusal
    try:
        result = rag_service.answer(body.message)
    except Exception:  # noqa: BLE001 - never leak internals to the client
        total_ms = (time.perf_counter() - started) * 1000.0
        logger.info("timing chat_total_ms=%.1f status=error", total_ms)
        # rag_service.answer() should not raise; this is the last-resort guard.
        return ChatResponse(answer=ERROR_ANSWER, sources=[], status="error")

    sources = [
        Source(
            filename=source.filename,
            page_number=source.page_number,
            snippet=source.snippet,
            distance=source.distance,
            chunk_index=source.chunk_index,
            source_type="web" if source.source_type == "web" else "pdf",
            url=source.url,
            published_date=source.published_date,
            title=source.title,
        )
        for source in result.sources
    ]
    total_ms = (time.perf_counter() - started) * 1000.0
    logger.info("timing chat_total_ms=%.1f status=%s", total_ms, result.status)
    return ChatResponse(answer=result.answer, sources=sources, status=result.status)


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, request: Request):
    """Stream the answer as Server-Sent Events.

    Same auth-free public contract, the same per-IP + daily rate limits as
    /chat (evaluated exactly once), untrusted input unchanged. Events:
      - data: {"type":"delta","text":"..."}       (zero or more)
      - data: {"type":"verifying"}                 (answer complete, verification pending)
      - data: {"type":"final", ...}                (verified answer + sources + status)
      - data: {"type":"replace", ...}              (verifier/truncation rejection)
      - data: {"type":"error", ...}                (generic failure)

    Client disconnect aborts upstream generation (the underlying httpx stream
    is closed). No secrets, stack traces or internal details are ever sent.
    """
    started = time.perf_counter()
    refusal = _rate_limit_or_refuse(request)
    if refusal is not None:
        return refusal

    def event_stream():
        status = "answered"
        try:
            for event in rag_service.stream_answer(body.message):
                if isinstance(event, dict):
                    status = str(event.get("status", status) or status)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
        except GeneratorExit:
            # Client disconnected — rag_service.stream_answer() and the
            # underlying httpx stream close() run in their finally blocks.
            logger.info("stream client disconnected before completion")
            raise
        except Exception:  # noqa: BLE001 - never leak internals to the client
            # Last-resort guard: the client may already have received deltas;
            # end the stream with a generic terminal error event.
            logger.warning("stream aborted by server error")
            status = "error"
            try:
                last = json.dumps({'type': 'error', 'status': 'error', 'message': ERROR_ANSWER},
                                  ensure_ascii=False)
                yield f"data: {last}\n\n".encode("utf-8")
            except GeneratorExit:
                raise
        finally:
            total_ms = (time.perf_counter() - started) * 1000.0
            logger.info("timing chat_stream_total_ms=%.1f status=%s", total_ms, status)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
