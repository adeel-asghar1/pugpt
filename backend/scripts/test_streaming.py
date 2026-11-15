"""Streaming responses (SSE) — offline, fully mocked.

Exercises rag_service.stream_answer() through the real GenerationService +
GroqService path against httpx.MockTransport-fed SSE bodies, the /api/chat
route refactor, and the route's SSE framing. No external network.

Run from the backend/ directory:
    python scripts/test_streaming.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.api.routes import chat as chat_module
from app.config import settings
from app.schemas.chat import ChatRequest
from app.services.generation_service import GenerationService
from app.services.groq_service import GroqService
from app.services.rag_service import (
    ERROR_ANSWER,
    NO_CONTEXT_ANSWER,
    RAGService,
    STATUS_ANSWERED,
    STATUS_NO_CONTEXT,
)
from app.services.web_search_service import WebSearchResult, WebSource

failures = 0


def ok(msg):
    print(f"[PASS] {msg}")


def bad(msg):
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


def check(label, condition, detail=""):
    if condition:
        ok(label)
    else:
        bad(f"{label} :: {detail}")


class patched:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.saved = {k: getattr(settings, k) for k in self.kwargs}
        for k, v in self.kwargs.items():
            setattr(settings, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(settings, k, v)


LONG_PARAGRAPH = (
    "The 2026-27 Admission Regulations state that a candidate seeking admission "
    "to a BS programme must satisfy the eligibility criteria prescribed for that "
    "programme, including the required level of education, the minimum marks, "
    "the subject prerequisites, the prescribed entry test where applicable and "
    "the closing date for applications. A candidate applying on the basis of "
    "intermediate education must possess the required certificate from an "
    "institution recognised by this University or an equivalent qualification "
    "recognised by the University authorities. The University may require the "
    "submission of attested copies of certificates and the application fee by "
    "the closing date and may verify original documents at the time of "
    "admission. Admission is provisional and subject to verification of all "
    "documents. The regulations further provide that the University retains the "
    "right to withhold or cancel admission in case of any discrepancy, "
    "misrepresentation or failure to satisfy these conditions. Merit is "
    "computed in accordance with the formula notified by the University for "
    "each session and each programme category. "
)
STRONG_DOC = LONG_PARAGRAPH * 3


def chunk(text, doc_type="guidelines", filename="Eligibility-2026-27.pdf", page=3, distance=0.1):
    return (
        text,
        {
            "filename": filename,
            "page_number": page,
            "chunk_index": 0,
            "source_path": filename,
            "doc_id": "test",
            "category": doc_type,
            "document_type": doc_type,
            "effective_session": "2026-27",
            "verification_status": "verified",
        },
        distance,
    )


def off_topic_chunk(distance=0.1):
    return chunk(SPARSE_DOC if False else (
        "Regular admission age limit text: applicants must be below the age limit "
        "as printed by the 2026-27 eligibility criteria for the chosen programme."
    ), "guidelines", "AgeLimit.pdf", 1, distance)


SPARSE_DOC = (
    "Regular admission age limit text: applicants must be below the age limit "
    "as printed by the 2026-27 eligibility criteria for the chosen programme."
)


def relevant_closure_source():
    return WebSource(
        title="Campus Closure Notice",
        url="https://www.pu.edu.pk/news/closures",
        host="www.pu.edu.pk",
        content=(
            "Official notice: the University of the Punjab has announced closure "
            "of all campuses from 26 to 30 November. Classes are suspended and "
            "the class schedule is revised accordingly."
        ),
        score=0.9,
        published_date="2026-11-01",
    )


class FakeChroma:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.query_calls = 0
        self.filtered_calls = 0

    def count(self):
        return max(len(self.chunks), 1)

    def query(self, vector, top_k=None):
        self.query_calls += 1
        return {
            "documents": [[c[0] for c in self.chunks]],
            "metadatas": [[c[1] for c in self.chunks]],
            "distances": [[c[2] for c in self.chunks]],
        }

    def query_filtered(self, vector, where=None, top_k=None):
        self.filtered_calls += 1
        doc_type = (where or {}).get("document_type")
        matched = [c for c in self.chunks if c[1].get("document_type") == doc_type]
        return {
            "documents": [[c[0] for c in matched]],
            "metadatas": [[c[1] for c in matched]],
            "distances": [[c[2] for c in matched]],
        }


class FakeEmbeddings:
    def embed_text(self, text):
        return [0.01] * 8


class FakeWeb:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def search(self, question, **kwargs):
        self.calls += 1
        return self.result


def sse_delta(text, finish_reason=None):
    chunk = {"choices": [{"index": 0,
                          "delta": {"role": "assistant", "content": text},
                          "finish_reason": finish_reason}]}
    return f"data: {json.dumps(chunk)}\n\n"


def sse_finish(finish_reason="stop"):
    chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(chunk)}\n\n"


def sse_done():
    return "data: [DONE]\n\n"


def make_groq_sse(deltas, finish_reason="stop"):
    body = ""
    for delta in deltas:
        body += sse_delta(delta)
    body += sse_finish(finish_reason)
    body += sse_done()
    return body


def make_rag(chunks, web_result, sse_body, max_status=200, intent_filtered=None):
    chroma = FakeChroma(chunks)
    web = FakeWeb(web_result)
    groq_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        groq_calls.append(request)
        return httpx.Response(max_status, text=sse_body,
                              headers={"content-type": "text/event-stream"})

    groq = GroqService(api_key="sk-test-STREAM", transport=httpx.MockTransport(handler))
    llm = GenerationService(provider="groq", groq=groq)
    rag = RAGService(embeddings=FakeEmbeddings(), chroma=chroma, web_search=web, llm=llm)
    return rag, chroma, web, groq_calls


def collect(stream_iter):
    return list(stream_iter)


def terminal(events):
    terminals = [e for e in events if e.get("type") in ("final", "replace", "error")]
    return terminals


def types(events):
    return [e.get("type") for e in events]


def event_sequences(events):
    return " -> ".join(types(events))


print("=" * 78)
print("Streaming responses — mocked offline tests")
print("=" * 78)

QUESTION_ELIGIBILITY = "What are the eligibility criteria for undergraduate admissions?"
QUESTION_CLOSURE = (
    "Has the University of the Punjab issued any official notice about "
    "campus-wide closures or changes to the class schedule?"
)

print("\n[1. pdf-sufficient answer — verified final, sources gated to final]")
body = make_groq_sse(["Eligibility ", "criteria published."], finish_reason="stop")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body)
with patched(web_search_enabled=True, web_search_on_recency=False):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("answered path -> final with status answered",
      term and term[0]["type"] == "final" and term[0]["status"] == "answered")
check("verifying phase emitted before final",
      types(events).index("verifying") < types(events).index("final")
      if "verifying" in types(events) else False, event_sequences(events))
check("exactly one Groq HTTP request", len(groq_calls) == 1, f"{len(groq_calls)} calls")
check("retrieval invoked exactly once", chroma.query_calls == 1, f"query_calls={chroma.query_calls}")
check("web fallback never fired (PDF sufficient)", web.calls == 0, f"calls={web.calls}")
final = term[0]
check("sources appear only on the terminal event",
      all("sources" not in e for e in events[:-1]) and bool(final.get("sources")))
check("final answer matches the concatenated deltas",
      final.get("answer") == "Eligibility criteria published.", final.get("answer"))
check("verifier accepted", final.get("verifier") == "accepted")
check("deltas arrive before verification",
      types(events) == sorted(types(events),
                              key=lambda t: {"delta": 0, "verifying": 1, "final": 2}[t])
      if set(types(events)) <= {"delta", "verifying", "final"} else False,
      event_sequences(events))

print("\n[2. verifier rejection — provisional answer replaced]")
body = make_groq_sse(["The age limit is 24 years."], finish_reason="stop")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body)
with patched(web_search_enabled=True):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("verifier-refused -> replace with NO_CONTEXT_ANSWER",
      term and term[0]["type"] == "replace"
      and term[0]["answer"] == NO_CONTEXT_ANSWER, event_sequences(events))
check("refused replacement is not marked answered",
      term and term[0].get("status") == "no_relevant_context", str(term))
check("rejected answer text is not in the terminal payload",
      term and "24 years" not in term[0].get("answer", ""))
check("'24 years' appears only as provisional delta",
      any(e.get("text") == "The age limit is 24 years." for e in events if e.get("type") == "delta")
      and all("24 years" not in json.dumps(e) for e in term))
check("replace carries no sources", term and term[0].get("sources") == [])

print("\n[3. finish_reason=length — truncation never presented as complete]")
body = make_groq_sse(["Partial answer cut off mid-sent"], finish_reason="length")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body)
with patched(web_search_enabled=True):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("truncation -> replace with ERROR_ANSWER, status error",
      term and term[0]["type"] == "replace" and term[0]["answer"] == ERROR_ANSWER
      and term[0]["status"] == "error", event_sequences(events))
check("truncated text not in terminal payload",
      term and "Partial answer" not in term[0].get("answer", ""))
check("verifying is never emitted for truncated streams", "verifying" not in types(events))

print("\n[4. malformed SSE frame — fail closed]")
body_with_bad_line = (
    sse_delta("Good chunk. ") +
    'data: {not-json}\n\n' +
    sse_done()
)
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body_with_bad_line)
with patched(web_search_enabled=True):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("malformed frame -> error event with generic message",
      term and term[0]["type"] == "error" and term[0]["message"] == ERROR_ANSWER,
      event_sequences(events))

print("\n[5. provider HTTP 500 — generic error, no internals]")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, '{"error":"provider xyz broke"}',
    max_status=500)
with patched(web_search_enabled=True):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("provider 500 -> error event, generic message",
      term and term[0]["type"] == "error" and term[0]["message"] == ERROR_ANSWER)
check("provider internals not leaked", term and "xyz" not in term[0].get("message", ""))

print("\n[6. sparse pdf + stream — one tav, one retrieval, mixed sources]")
body = make_groq_sse(["Campus closure dates announced."], finish_reason="stop")
rag, chroma, web, groq_calls = make_rag(
    [off_topic_chunk(0.1) for _ in range(5)],
    WebSearchResult(results=[relevant_closure_source()], response_time=0.2, credits_used=1),
    body)
with patched(web_search_enabled=True, web_search_on_recency=False):
    events = collect(rag.stream_answer(QUESTION_CLOSURE))
term = terminal(events)
check("spaced-sparse evidence triggers exactly one web search", web.calls == 1, f"calls={web.calls}")
check("retrieval invoked exactly once (plain query)", chroma.query_calls == 1, f"calls={chroma.query_calls}")
check("no intent-filtered merge for closure question", chroma.filtered_calls == 0, f"calls={chroma.filtered_calls}")
check("exactly one terminal event", len(term) == 1, event_sequences(events))
check("streamed path answered with pdf+web sources",
      term and term[0]["type"] == "final" and term[0]["status"] == "answered"
      and len(term[0]["sources"]) >= 2, event_sequences(events))

print("\n[7. provider finish_reason=stop but no [DONE] marker — accepted]")
body = make_groq_sse(["Answer body.", ], finish_reason="stop").replace(sse_done(), "")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body)
with patched(web_search_enabled=True):
    events = collect(rag.stream_answer(QUESTION_ELIGIBILITY))
term = terminal(events)
check("stop without [DONE] still terminates cleanly", len(term) == 1
      and term[0]["type"] == "final", event_sequences(events))

print("\n[8. stream abort mid-answer — teardown does not raise]")
body = make_groq_sse(["part one. ", "part two. ", "part three."], finish_reason="stop")
rag, chroma, web, groq_calls = make_rag(
    [chunk(STRONG_DOC) for _ in range(3)], None, body)
with patched(web_search_enabled=True):
    gen = rag.stream_answer(QUESTION_ELIGIBILITY)
    first = next(gen)
    deltas = [first]
    gen.close()
    try:
        next(gen)
        bad("closed generator returned events")
    except StopIteration:
        ok("generator closed mid-stream raises StopIteration")

print("\n[9. shared rate-limit helper — same decision path for both endpoints]")


class FakeRequest:
    def __init__(self, host="127.0.0.1"):
        class C:
            def __init__(self, h):
                self.host = h
        self.client = C(host)
        self.headers = {}


class FakePerIPLimiter:
    def __init__(self, allowed=True, retry_after=0.0):
        self.allowed, self.retry_after = allowed, retry_after
        self.calls = 0

    def check(self, ip):
        self.calls += 1
        return self.allowed, self.retry_after


class FakeGlobalLimiter:
    def __init__(self, allowed=True, retry_after=7.0, unavailable=False):
        self.allowed, self.retry_after, self.unavailable = allowed, retry_after, unavailable
        self.calls = 0

    def try_consume(self):
        self.calls += 1
        if self.unavailable:
            return None, 0.0
        return self.allowed, self.retry_after


_orig_per_ip = chat_module.rate_limiter
_orig_global = chat_module.global_rate_limiter
try:
    req = FakeRequest()
    perip, glb = FakePerIPLimiter(True), FakeGlobalLimiter(True)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    refusal = chat_module._rate_limit_or_refuse(req)
    check("allowed request -> no refusal; each limiter called exactly once",
          refusal is None and perip.calls == 1 and glb.calls == 1,
          f"ip={perip.calls} glb={glb.calls}")

    perip, glb = FakePerIPLimiter(False, 30.0), FakeGlobalLimiter(True)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    refusal = chat_module._rate_limit_or_refuse(req)
    check("per-IP denial -> 429, daily cap untouched",
          refusal is not None and refusal.status_code == 429 and glb.calls == 0,
          f"glb={glb.calls}")

    perip, glb = FakePerIPLimiter(True), FakeGlobalLimiter(False, 8000.0)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    refusal = chat_module._rate_limit_or_refuse(req)
    check("daily-cap denial -> 429 daily_limit_exceeded",
          refusal is not None and refusal.status_code == 429
          and b"daily_limit_exceeded" in refusal.body, refusal.body if refusal else None)

    perip, glb = FakePerIPLimiter(True), FakeGlobalLimiter(unavailable=True)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    refusal = chat_module._rate_limit_or_refuse(req)
    check("limiter storage failure -> 503 fail-closed",
          refusal is not None and refusal.status_code == 503, None)
finally:
    chat_module.rate_limiter, chat_module.global_rate_limiter = _orig_per_ip, _orig_global

print("\n[10. /chat/stream SSE framing + headers (mocked RAG)]")


class FakeStreamingRAG:
    def stream_answer(self, question):
        yield {"type": "delta", "text": "chunk "}
        yield {"type": "verifying"}
        yield {"type": "final", "status": "answered", "answer": "chunk ",
               "sources": [{"filename": "x.pdf", "page_number": 1, "snippet": "s",
                            "distance": 0.1, "chunk_index": 0,
                            "source_type": "pdf", "url": None,
                            "published_date": None, "title": None}],
               "verifier": "accepted"}


_orig_rag = chat_module.rag_service
try:
    chat_module.rag_service = FakeStreamingRAG()
    perip, glb = FakePerIPLimiter(True), FakeGlobalLimiter(True)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    req = FakeRequest()
    body_req = ChatRequest(message="anything")
    response = chat_module.chat_stream(body_req, req)
    check("stream route returns StreamingResponse",
          type(response).__name__ == "StreamingResponse")
    check("SSE headers set (no-cache, no buffering)",
          response.headers.get("cache-control") == "no-cache"
          and response.headers.get("x-accel-buffering") == "no",
          str(dict(response.headers)))
    check("one per-IP call + one daily consumption for the stream request",
          perip.calls == 1 and glb.calls == 1, f"ip={perip.calls} glb={glb.calls}")

    async def collect_async():
        chunks = []
        async for chunk_bytes in response.body_iterator:
            chunks.append(chunk_bytes)
        return b"".join(chunks)

    raw = asyncio.run(collect_async()).decode("utf-8")
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    parsed = []
    framing_ok = True
    for block in blocks:
        if not block.startswith("data: "):
            framing_ok = False
            break
        parsed.append(json.loads(block[len("data: "):]))
    check("every SSE block is 'data: <json>' framed", framing_ok, raw)
    check("streamed events arrive in order: delta, verifying, final",
          [e.get("type") for e in parsed] == ["delta", "verifying", "final"],
          str([e.get("type") for e in parsed]))
finally:
    chat_module.rag_service = _orig_rag
    chat_module.rate_limiter, chat_module.global_rate_limiter = _orig_per_ip, _orig_global

print("\n[11. non-stream /chat path intact after refactor]")


class FakeSyncRAG:
    def answer(self, question):
        from app.services.rag_service import RAGResult
        return RAGResult(STATUS_ANSWERED, "Sync answer.", [], None)


_orig_rag = chat_module.rag_service
try:
    chat_module.rag_service = FakeSyncRAG()
    perip, glb = FakePerIPLimiter(True), FakeGlobalLimiter(True)
    chat_module.rate_limiter, chat_module.global_rate_limiter = perip, glb
    reply = chat_module.chat(ChatRequest(message="anything"), FakeRequest())
    check("/chat returns answered sync response",
          getattr(reply, "status", None) == "answered" and reply.answer == "Sync answer.")
finally:
    chat_module.rag_service = _orig_rag
    chat_module.rate_limiter, chat_module.global_rate_limiter = _orig_per_ip, _orig_global

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
