"""Mocked Groq provider tests — no network, no real key.

Covers the provider contract PUGPT depends on:
- success (key sent in Authorization header only; system prompt forwarded)
- missing API key (fails fast without a network call)
- HTTP 401 (auth), HTTP 429 (rate limit), HTTP 5xx (upstream), timeout
- malformed / empty provider body
- provider selection: auto -> groq when key set, auto -> ollama otherwise,
  explicit groq / ollama forced, invalid value rejected
- RAG failure path: Groq failure -> 'error' status, generic answer, no key,
  sources still attached
- health route: generation block present, no key revealed

Run from the backend/ directory:
    python scripts/test_groq.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import settings
from app.services.groq_service import (
    GroqAuthError,
    GroqError,
    GroqRateLimitError,
    GroqResponseError,
    GroqService,
    GroqTimeoutError,
    GroqUpstreamError,
)
from app.services.generation_service import GenerationConfigError, GenerationService

TEST_KEY = "sk-test-SECRETKEY123"  # distinctive so leaks are easy to detect
failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


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


def make_groq(response=None, raise_exc=None, api_key=TEST_KEY):
    """Build a GroqService whose HTTP layer is a MockTransport."""
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(500, text="unexpected test setup")

    svc = GroqService(
        api_key=api_key,
        base_url="https://api.groq.test/chat/completions",
        model="test-model",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )
    return svc, captured


print("=" * 78)
print("Groq provider — mocked tests")
print("=" * 78)

# ----------------------------------------------------------------- success
print("\n[success]")
svc, captured = make_groq(
    httpx.Response(
        200, json={"choices": [{"message": {"content": "  grounded answer  "}}]}
    )
)
out = svc.generate("user question", system="system prompt text")
assert out == "grounded answer", out
req = captured[0]
assert req.headers["authorization"] == f"Bearer {TEST_KEY}"
assert req.url == "https://api.groq.test/chat/completions"
body = json.loads(req.content)
assert body["model"] == "test-model"
assert body["messages"][0] == {"role": "system", "content": "system prompt text"}
assert body["messages"][-1]["role"] == "user"
assert TEST_KEY not in json.dumps([m for m in body.values()])
ok("success path returns trimmed content; key sent in Authorization header only")

# ----------------------------------------------------------------- key cfg
print("\n[missing API key]")
svc, captured = make_groq(httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}), api_key="")
try:
    svc.generate("hi")
    bad("missing key did NOT raise")
except GroqError as exc:
    if captured:
        bad("missing key performed a network call (must fail fast)")
    else:
        msg = str(exc)
        if TEST_KEY in msg:
            bad("missing-key error leaked the key")
        elif "GROQ_API_KEY" in msg:
            ok("missing key -> clear configuration error, no network call")
        else:
            bad(f"missing-key error unclear: {msg}")

# ----------------------------------------------------------------- status codes
print("\n[provider error classes]")
cases = [
    (401, GroqAuthError),
    (403, GroqAuthError),
    (429, GroqRateLimitError),
    (500, GroqUpstreamError),
    (503, GroqUpstreamError),
]
for code, exc_type in cases:
    svc, _ = make_groq(httpx.Response(code, json={"error": {"message": "oops"}}))
    try:
        svc.generate("hi")
        bad(f"HTTP {code} did not raise")
    except exc_type:
        ok(f"HTTP {code} -> {exc_type.__name__}")
    except Exception as exc:  # noqa: BLE001
        bad(f"HTTP {code} -> unexpected {type(exc).__name__}")

# ----------------------------------------------------------------- timeout
print("\n[timeout]")
svc, _ = make_groq(raise_exc=httpx.ReadTimeout("boom"))
try:
    svc.generate("hi")
    bad("timeout did not raise")
except GroqTimeoutError:
    ok("ReadTimeout -> GroqTimeoutError")

# ----------------------------------------------------------------- bodies
print("\n[malformed / empty responses]")
for label, response in [
    ("malformed JSON", httpx.Response(200, content=b"not-json")),
    ("empty body", httpx.Response(200, content=b"")),
    ("missing choices", httpx.Response(200, json={})),
    ("empty choices list", httpx.Response(200, json={"choices": []})),
    ("empty content", httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})),
    ("whitespace content", httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})),
]:
    svc, _ = make_groq(response)
    try:
        svc.generate("hi")
        bad(f"{label} did not raise")
    except GroqResponseError:
        ok(f"{label} -> GroqResponseError")

# ------------------------------------------------- provider selection (mock-only)
print("\n[provider selection]")
try:
    with patched(llm_provider="auto", groq_api_key=TEST_KEY):
        provider = GenerationService().resolved_provider()
        assert provider == "groq", provider
    ok("auto + key set -> 'groq'")
except Exception as exc:  # noqa: BLE001
    bad(f"auto + key -> {exc}")

try:
    with patched(llm_provider="auto", groq_api_key=""):
        provider = GenerationService().resolved_provider()
        assert provider == "ollama", provider
    ok("auto + no key -> 'ollama'")
except Exception as exc:  # noqa: BLE001
    bad(f"auto + no key -> {exc}")

try:
    with patched(llm_provider="groq", groq_api_key=""):
        groq = GenerationService(groq=GroqService(api_key=""))
        try:
            groq.generate("hi")
            bad("forced groq without key -> no error raised")
        except GroqError as exc:
            msg = str(exc)
            ok("forced groq + no key -> clear error, no fallback to Ollama")
except Exception as exc:  # noqa: BLE001
    bad(f"forced groq path: {exc}")

try:
    with patched(llm_provider="ollama", groq_api_key=TEST_KEY):
        provider = GenerationService().resolved_provider()
        assert provider == "ollama", provider
    ok("explicit ollama -> 'ollama' (Groq not used despite key)")
except Exception as exc:  # noqa: BLE001
    bad(f"explicit ollama -> {exc}")

try:
    with patched(llm_provider="banana"):
        GenerationService().resolved_provider()
        bad("invalid LLM_PROVIDER accepted")
except GenerationConfigError:
    ok("invalid LLM_PROVIDER -> GenerationConfigError")

# --------------------------------------------------- completions URL forms
print("\n[URL construction — exactly one /chat/completions]")
URL_CASES = [
    ("https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/chat/completions"),
    ("https://api.groq.com/openai/v1/", "https://api.groq.com/openai/v1/chat/completions"),
    ("https://api.groq.com/openai/v1/chat/completions", "https://api.groq.com/openai/v1/chat/completions"),
    ("https://api.groq.com/openai/v1/chat/completions/", "https://api.groq.com/openai/v1/chat/completions"),
    ("https://api.groq.com/openai/v1/chat/completions/chat/completions", "https://api.groq.com/openai/v1/chat/completions"),
    ("  https://api.groq.com/openai/v1/  ", "https://api.groq.com/openai/v1/chat/completions"),
]
for base, expected in URL_CASES:
    svc = GroqService(api_key=TEST_KEY, base_url=base, model="test-model", timeout=5.0)
    if svc.completions_url == expected:
        ok(f"base {base!r} -> {expected}")
    else:
        bad(f"base {base!r} produced {svc.completions_url!r}, expected {expected}")

# And through the actual request path (mock transport, zero network):
print("\n[URL construction — actual request target, mocked]")
for base, expected in URL_CASES:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    svc = GroqService(
        api_key=TEST_KEY,
        base_url=base,
        model="test-model",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )
    svc.generate("ping")
    if captured and captured[0] == expected:
        ok(f"request for {base!r} landed on {expected}")
    else:
        bad(f"request for {base!r} landed on {captured}, expected {expected}")

# ------------------------------ RAG integration: failure maps to safe error
print("\n[RAG integration — Groq failure -> safe error, sources kept, no key]")
from app.services.rag_service import RAGService
from app.services.chroma_service import ChromaService

temp_root = Path(tempfile.mkdtemp(prefix="pugpt_groq_test_"))
try:
    chroma = ChromaService(persist_path=temp_root / "chroma", collection_name="pugpt_groq_test")
    from app.services.embedding_service import embedding_service

    vec = embedding_service.embed_text("attendance rule: 75 percent minimum")
    chroma.add_documents(
        ids=["gtest:0"],
        documents=["Attendance rule: a student must attend at least 75% of classes."],
        embeddings=[vec],
        metadatas=[{"filename": "gtest.pdf", "page_number": 1, "chunk_index": 0, "source_path": "gtest.pdf", "doc_id": "gtest"}],
    )

    class FailingGroq:
        model = "test-model"

        def generate(self, prompt, system=None, timeout=None):
            raise GroqRateLimitError("Groq responded with HTTP 429.")

    rag = RAGService(chroma=chroma, llm=FailingGroq())
    result = rag.answer("What is the attendance requirement?")
    if result.status == "error":
        ok("RAG failure path -> status='error'")
    else:
        bad(f"expected status='error', got {result.status}: {result.answer[:80]}")
    if result.sources:
        ok("sources still attached on safe error path")
    else:
        bad("no sources attached on safe error path")
    if TEST_KEY not in result.answer and TEST_KEY not in (result.error or ""):
        ok("no API key in error payload")
    else:
        bad("API key leaked into error payload")
    # ensure clean up of temp collection happens before dir removal
    try:
        chroma.delete_collection("pugpt_groq_test")
        ok("temp ChromaDB collection cleaned up")
    except Exception as exc:  # noqa: BLE001
        bad(f"temp collection cleanup failed: {exc}")
finally:
    shutil.rmtree(temp_root, ignore_errors=True)

# ------------------------------------------------------------- health payload
print("\n[health route — generation block, no secrets]")
from app.api.routes.health import health_check

try:
    with patched(llm_provider="auto", groq_api_key=TEST_KEY):
        payload = health_check()
    gen = payload.get("generation", {})
    if gen.get("provider") == "groq" and gen.get("model") and gen.get("api_key_configured") is True:
        ok("health reports generation.provider='groq' with model + key-configured flag")
    else:
        bad(f"unexpected generation block: {gen}")
    blob = json.dumps(payload)
    if TEST_KEY in blob:
        bad("health payload contains the API key")
    else:
        ok("health payload does NOT contain the API key")
except Exception as exc:  # noqa: BLE001
    bad(f"health check raised: {exc}")

try:
    with patched(llm_provider="banana"):
        payload = health_check()
    gen = payload.get("generation", {})
    if gen.get("provider") == "invalid":
        ok("invalid LLM_PROVIDER -> health still HTTP 200 with provider='invalid'")
    else:
        bad(f"invalid provider not reported cleanly: {gen}")
except Exception as exc:  # noqa: BLE001
    bad(f"health crashed on invalid LLM_PROVIDER: {exc}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
