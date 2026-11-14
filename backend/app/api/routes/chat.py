import logging
import math

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.services.rag_service import ERROR_ANSWER, rag_service
from app.services.rate_limiter import client_ip, rate_limiter
from app.services.global_rate_limiter import global_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse | JSONResponse:
    """Answer a question about the University of the Punjab using RAG."""
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
    try:
        result = rag_service.answer(body.message)
    except Exception:  # noqa: BLE001 - never leak internals to the client
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
    return ChatResponse(answer=result.answer, sources=sources, status=result.status)
