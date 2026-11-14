"""API tests for POST /api/chat (mocked RAG) and GET /api/health (real).

MOCKED tests patch the route module's shared `rag_service` with a fake —
deterministic, no Ollama calls, no knowledge-base writes.
LIVE test calls the real FastAPI app with the real (currently EMPTY)
knowledge base; because it is empty, the pipeline short-circuits before
any LLM call, so it is fast and touches nothing.

Run from the backend/ directory:
    python scripts/test_chat_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.rag_service import RAGResult, RAGSource

import app.api.routes.chat as chat_module

client = TestClient(app)


def ok(msg):
    print(f"[PASS] {msg}")


def bad(msg):
    print(f"[FAIL] {msg}")
    global failures
    failures += 1


failures = 0

ANSWER = RAGResult(
    status="answered",
    answer="Classes commence on 15 September 2024.",
    sources=[
        RAGSource(
            filename="test_gazette.pdf",
            page_number=2,
            snippet="The Autumn Semester ... will commence on 15 September 2024.",
            distance=0.0467,
            chunk_index=3,
        )
    ],
    error=None,
)
NO_CONTEXT = RAGResult(
    status="no_relevant_context",
    answer="I could not find a reliable answer in the available University of the Punjab documents.",
    sources=[],
    error=None,
)
ERROR_RESULT = RAGResult(
    status="error",
    answer="PUGPT is temporarily unable to answer. Please try again in a moment.",
    sources=[],
    error="internal detail that must not leak",
)


class FakeRAG:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    def answer(self, message):
        if self.exc is not None:
            raise self.exc
        return self.result


print("--- Mocked RAG tests ---\n")

# T1: answered path — answer + sources preserved end to end
with patch.object(chat_module, "rag_service", FakeRAG(result=ANSWER)):
    r = client.post("/api/chat", json={"message": "When does the autumn semester start?"})
body = r.json()
if r.status_code == 200 and body["answer"] == ANSWER.answer:
    ok("T1 valid question -> 200 with RAG answer")
else:
    bad(f"T1: status={r.status_code}, body={body}")
src = (body.get("sources") or [{}])[0]
if src.get("filename") == "test_gazette.pdf" and src.get("page_number") == 2 and src.get("snippet") and isinstance(src.get("distance"), float):
    ok("T1 source filename, page_number, snippet, distance preserved")
else:
    bad(f"T1: source wrong: {src}")
if body.get("status") == "answered":
    ok("T1 status == 'answered'")
else:
    bad(f"T1: status={body.get('status')}")

# T2: no relevant context -> safe unavailable-information response
with patch.object(chat_module, "rag_service", FakeRAG(result=NO_CONTEXT)):
    r = client.post("/api/chat", json={"message": "What is the airspeed of an unladen swallow?"})
body = r.json()
if r.status_code == 200 and body["status"] == "no_relevant_context" and body["answer"] and body["sources"] == []:
    ok("T2 no-context question -> safe response, empty sources, no LLM answer fabricated")
else:
    bad(f"T2: status={r.status_code}, body={body}")

# T3: internal error string must NOT leak
with patch.object(chat_module, "rag_service", FakeRAG(result=ERROR_RESULT)):
    r = client.post("/api/chat", json={"message": "hello"})
body = r.json()
if r.status_code == 200 and body["status"] == "error" and "internal detail" not in r.text:
    ok("T3 error status returned; internal details not exposed")
else:
    bad(f"T3: leak or wrong shape: {body}")

# T4: unhandled RAG exception -> generic answer, no traceback
with patch.object(chat_module, "rag_service", FakeRAG(exc=RuntimeError("boom-secret- internals"))):
    r = client.post("/api/chat", json={"message": "hello"})
if r.status_code == 200 and r.json()["status"] == "error" and "boom-secret" not in r.text and "Traceback" not in r.text:
    ok("T4 unhandled exception -> generic answer, no stack trace")
else:
    bad(f"T4: leak or failure: status={r.status_code}, body={r.text[:200]}")

# T5: whitespace-only message rejected with 422
r = client.post("/api/chat", json={"message": "   "})
if r.status_code == 422:
    ok("T5 whitespace-only message -> HTTP 422")
else:
    bad(f"T5: expected 422, got {r.status_code}")

# T6: health endpoint still works
r = client.get("/api/health")
body = r.json()
if r.status_code == 200 and body.get("status") == "ok" and "ollama" in body:
    ok("T6 /api/health -> 200 with ollama block")
else:
    bad(f"T6: health broken: status={r.status_code}, body={body}")

# T7: response matches the frontend contract (answer: str, sources: list)
with patch.object(chat_module, "rag_service", FakeRAG(result=ANSWER)):
    r = client.post("/api/chat", json={"message": "contract check"})
body = r.json()
if isinstance(body.get("answer"), str) and isinstance(body.get("sources"), list) and all(
    set(src) >= {"filename", "page_number", "snippet", "distance"} for src in body["sources"]
):
    ok("T7 response contract matches frontend expectations (answer str, sources [] with filename/page_number)")
else:
    bad(f"T7: contract broken: {body}")

print("\n--- Live test (real FastAPI + real services; knowledge base is currently EMPTY) ---\n")

# L1 [LIVE]: real service; empty KB short-circuits before any LLM call
r = client.post("/api/chat", json={"message": "When does the autumn semester start?"})
body = r.json()
if r.status_code == 200 and isinstance(body.get("answer"), str) and body["answer"]:
    expected_statuses = {"empty_knowledge_base", "no_relevant_context", "answered"}
    if body["status"] in expected_statuses:
        ok(f"L1 [LIVE] real /api/chat -> 200, status={body['status']} (KB empty short-circuits before LLM)")
    else:
        bad(f"L1: unexpected status {body['status']}")
else:
    bad(f"L1: status={r.status_code}, body={body}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
