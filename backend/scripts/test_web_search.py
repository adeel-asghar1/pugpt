"""Mocked Tavily official-PU web-search tests — no network, no real key.

Covers the Step 23B contract:
- Server-side hostname validation (pu.edu.pk + subdomains only; lookalikes,
  scheme confusion, malformed URLs rejected).
- Tavily client behavior: payload shape (basic depth, 5 results, no answer,
  include_usage, domains restrict), single request (no retries), key sent in
  Authorization header only, unofficial results dropped server-side, error
  taxonomy (401/403 auth, 429/432/433/434 quota, 5xx upstream, timeout,
  malformed body), fail-fast on missing key without any HTTP call.
- RAG integration (fully mocked LLM + fake Chroma/embeddings + fake web):
    * PDF available -> zero web calls
    * PDF missing -> exactly one web search, web answer used, sources labeled
    * recency question -> web search triggered even with PDF context
    * official PU results accepted; unofficial-only -> safe refusal, LLM not called
    * PDF/web conflict -> both presented, conflict preserved, never silent pick
    * web failure (timeout/quota/5xx) -> graceful PDF-only answer or safe refusal
    * WEB_SEARCH_ENABLED=false -> web never called (master switch)
    * verifier passes web-grounded figures; invented figures refused
- Health block exposes booleans only, never the key.

Run from the backend/ directory:
    python scripts/test_web_search.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import Settings, settings
from app.services.rag_service import RAGService, find_unsupported_claims
from app.services.web_search_service import (
    WebSearchAuthError,
    WebSearchError,
    WebSearchRateLimitError,
    WebSearchResponseError,
    WebSearchResult,
    WebSearchService,
    WebSearchTimeoutError,
    WebSearchUpstreamError,
    WebSource,
    validated_host,
)

TEST_KEY = "tvly-test-SECRETKEY123"  # distinctive so leaks are easy to detect
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


# ------------------------------------------------------- test doubles
class FakeEmbedding:
    def embed_text(self, text: str):
        return [0.01] * 8


class FakeChroma:
    def __init__(self, candidates, count=1):
        self.candidates = candidates
        self._count = count

    def count(self):
        return self._count

    def query(self, query_embedding, top_k=None):
        return self.candidates

    def query_filtered(self, query_embedding, where, top_k=10):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class RecordingLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt, system=None, timeout=None):
        self.prompts.append(prompt)
        return self.answer


class FakeWebSearch:
    """Records calls; returns canned results or raises a canned error."""

    def __init__(self, results=None, raise_exc=None):
        self.results = results or []
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    def search(self, query, **kwargs):
        self.calls.append(query)
        if self.raise_exc is not None:
            raise self.raise_exc
        return WebSearchResult(
            results=list(self.results),
            response_time=0.01,
            credits_used=1,
        )


def official_web_source(content: str, url: str = "https://www.pu.edu.pk/admissions/2026-27",
                        title: str = "PU Admissions 2026-27",
                        published: str | None = "2026-08-01") -> WebSource:
    host = validated_host(url, ["pu.edu.pk"])
    return WebSource(title=title, url=url, host=host or "www.pu.edu.pk",
                     content=content, score=0.9, published_date=published)


# Strong PDF evidence fixture: distance 0.2 AND comfortably above the >=600-char
# sufficiency floor, modelling what an answered-from-the-KB question looks like.
PDF_DOC = (
    "Dues 2026-27 schedule. BS regular first-semester total due is Rs. 15,000. "
    "Service Charges are Rs. 8,500. Examination Fee is Rs. 4,800. "
    "Sports & Mosque Fund is Rs. 950 in the Fall semester only. "
    "Registration Fee Rs. 9,000, PU Development Fund Rs. 500 and Library Security "
    "Rs. 5,000 (refundable). Dues must be submitted within two weeks of admission. "
) * 6
PDF_META = {"filename": "Dues-Fee-Schedule-for-the-Year-2026-27.pdf", "page_number": 2,
            "chunk_index": 0, "category": "fees", "document_type": "fee schedule",
            "effective_session": "2026-27", "verification_status": "verified"}
KEPT_PDF = {"documents": [[PDF_DOC]], "metadatas": [[PDF_META]], "distances": [[0.2]]}
EMPTY_PDF = {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def make_rag(candidates, llm, web):
    return RAGService(
        embeddings=FakeEmbedding(),
        chroma=FakeChroma(candidates),
        llm=llm,
        web_search=web,
        distance_threshold=0.5,
    )


print("=" * 78)
print("Tavily official-PU web search — mocked tests")
print("=" * 78)

# ================================================= A. hostname validation
print("\n[A] server-side hostname validation")
cases = [
    ("https://pu.edu.pk/page", True),
    ("http://pu.edu.pk/page", True),
    ("https://cse.pu.edu.pk/program", True),
    ("https://sub.a.pu.edu.pk/deep", True),
    ("https://pu.edu.pk.", True),                     # trailing dot stripped
    ("HTTPS://CSE.PU.EDU.PK/UPPER", True),            # case-insensitive
    ("https://pu.edu.pk.evil.com/x", False),          # lookalike suffix
    ("https://notpu.edu.pk/x", False),                # sibling label
    ("https://epu.edu.pk/x", False),                  # not a subdomain boundary
    ("https://pu.edu.pk.evil.co.uk/x", False),
    ("http://user:pass@pu.edu.pk@evil.com/x", False), # host is evil.com
    ("javascript:alert(1)", False),                   # non-http(s)
    ("ftp://pu.edu.pk/file", False),                  # non-http(s)
    ("pu.edu.pk/page", False),                        # no scheme
    ("https://", False),                              # no host
    ("", False),
    ("not a url at all", False),
]
for url, expected in cases:
    host = validated_host(url, ["pu.edu.pk"])
    if (host is not None) == expected:
        ok(f"{url!r} -> {host!r}")
    else:
        bad(f"{url!r} -> {host!r} (expected {'accept' if expected else 'reject'})")

# ================================================= B. Tavily client
print("\n[B] Tavily client (MockTransport; single request, no retries)")
HANDLED = []


def make_http_service(response=None, raise_exc=None, api_key=TEST_KEY):
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(500, text="unexpected test setup")

    service = WebSearchService(api_key=api_key, transport=httpx.MockTransport(handler))
    return service, captured


svc, captured = make_http_service(
    httpx.Response(200, json={
        "results": [
            {"title": "PU Admissions", "url": "https://www.pu.edu.pk/adm", "content": "current admission schedule published for the 2026-27 session by the University of the Punjab",
             "score": 0.8, "published_date": "2026-08-01"},
            {"title": "Blogs", "url": "https://blog.example.com/pu", "content": "unofficial junk"},
            {"title": "Lookalike", "url": "https://pu.edu.pk.evil.com/", "content": "lookalike junk"},
            {"title": "Empty content", "url": "https://cse.pu.edu.pk/", "content": "   "},
            {"title": "Malformed", "url": "::::", "content": "broken"},
        ],
        "response_time": 0.42,
        "usage": {"credits": 1},
    })
)
result = svc.search("What is the current admission schedule?")
raw_payload = json.loads(captured[0].content.decode())
if len(captured) == 1:
    ok("exactly one HTTP request (no retries)")
else:
    bad(f"expected 1 request, got {len(captured)}")
checks = {
    "search_depth basic": raw_payload.get("search_depth") == "basic",
    "max_results 5": raw_payload.get("max_results") == 5,
    "include_answer false": raw_payload.get("include_answer") is False,
    "include_raw_content false": raw_payload.get("include_raw_content") is False,
    "include_usage true": raw_payload.get("include_usage") is True,
    "include_published_date true": raw_payload.get("include_published_date") is True,
    "include_domains pu.edu.pk": raw_payload.get("include_domains") == ["pu.edu.pk"],
    "country pakistan": raw_payload.get("country") == "pakistan",
    "safe_search true": raw_payload.get("safe_search") is True,
    "key not in body": TEST_KEY not in json.dumps(raw_payload),
    "auth header bearer": captured[0].headers.get("authorization", "").startswith("Bearer "),
    "auth header carries key": captured[0].headers.get("authorization") == f"Bearer {TEST_KEY}",
}
for name, passed in checks.items():
    ok(name) if passed else bad(name)
if [r.url for r in result.results] == ["https://www.pu.edu.pk/adm"]:
    ok("official result kept; unofficial/lookalike/empty/malformed dropped server-side")
else:
    bad(f"server-side filtering wrong: {[r.url for r in result.results]}")
first = result.results[0]
if (first.host == "www.pu.edu.pk" and first.published_date == "2026-08-01"
        and first.score == 0.8 and result.credits_used == 1):
    ok("published_date / score / credits_used parsed")
else:
    bad(f"parsed fields wrong: host={first.host}, date={first.published_date}, "
        f"score={first.score}, credits={result.credits_used}")

error_cases = [
    (httpx.Response(401, text="unauthorized"), WebSearchAuthError, "401 -> WebSearchAuthError"),
    (httpx.Response(403, text="forbidden"), WebSearchAuthError, "403 -> WebSearchAuthError"),
    (httpx.Response(429, text="slow down"), WebSearchRateLimitError, "429 -> WebSearchRateLimitError"),
    (httpx.Response(432, text="plan limit"), WebSearchRateLimitError, "432 -> WebSearchRateLimitError (quota)"),
    (httpx.Response(433, text="paygo limit"), WebSearchRateLimitError, "433 -> WebSearchRateLimitError (quota)"),
    (httpx.Response(434, text="flex capacity"), WebSearchRateLimitError, "434 -> WebSearchRateLimitError (capacity)"),
    (httpx.Response(500, text="boom"), WebSearchUpstreamError, "500 -> WebSearchUpstreamError"),
    (httpx.Response(503, text="boom"), WebSearchUpstreamError, "503 -> WebSearchUpstreamError"),
]
for resp, expected_exc, label in error_cases:
    svc, _ = make_http_service(resp)
    try:
        svc.search("q")
        bad(f"{label}: no exception raised")
    except expected_exc:
        ok(label)
    except Exception as exc:  # noqa: BLE001
        bad(f"{label}: raised {type(exc).__name__} instead")

svc, _ = make_http_service(raise_exc=httpx.TimeoutException("too slow"))
try:
    svc.search("q")
    bad("timeout: no exception raised")
except WebSearchTimeoutError:
    ok("timeout -> WebSearchTimeoutError")
svc, _ = make_http_service(httpx.Response(200, text="<html>not json</html>"))
try:
    svc.search("q")
    bad("malformed body: no exception raised")
except WebSearchResponseError:
    ok("200 with non-JSON body -> WebSearchResponseError")
svc, captured = make_http_service(api_key="")
try:
    svc.search("q")
    bad("missing key: no exception raised")
except WebSearchError:
    if len(captured) == 0:
        ok("missing key -> WebSearchError raised with ZERO HTTP requests")
    else:
        bad(f"missing key: {len(captured)} HTTP requests were made")

# ================================================= C. RAG integration
print("\n[C] RAG integration (mocked LLM + fake web)")
GOOD_PDF_ANSWER = "The 2026-27 first-semester fee is **Rs. 15,000**, per the fee schedule."
GOOD_WEB_ANSWER = "The current admission window is **August 2026**, per the live PU website."
CONFLICT_ANSWER = (
    "**The sources conflict.** The PDF fee schedule shows **Rs. 15,000** for 2026-27, "
    "while the live PU web page reports **Rs. 99,000**. They disagree; verify with the "
    "PU admissions office."
)

with patched(web_search_enabled=True, web_search_on_recency=True):
    # 1. PDF available, ordinary question -> zero web calls
    llm = RecordingLLM(GOOD_PDF_ANSWER)
    web = FakeWebSearch(results=[official_web_source("web junk should not be used")])
    result = make_rag(KEPT_PDF, llm, web).answer("What is the fee for 2026-27?")
    if web.calls == [] and result.status == "answered":
        ok("PDF available + ordinary question -> zero web calls")
    else:
        bad(f"expected 0 web calls & answered, got calls={web.calls}, status={result.status}")
    if all(s.source_type == "pdf" for s in result.sources):
        ok("sources stay PDF-typed")
    else:
        bad(f"non-PDF sources leaked: {[(s.source_type, s.url) for s in result.sources]}")

    # 2. PDF missing -> exactly one web search, web answer used
    llm = RecordingLLM(GOOD_WEB_ANSWER)
    web = FakeWebSearch(results=[official_web_source("current admission window August 2026 for 2026-27")])
    result = make_rag(EMPTY_PDF, llm, web).answer("When is the current admission window?")
    if result.status == "answered" and len(web.calls) == 1:
        ok("PDF missing -> exactly one web search, answer produced")
    else:
        bad(f"status={result.status}, web calls={len(web.calls)}")
    web_src = [s for s in result.sources if s.source_type == "web"]
    if (web_src and web_src[0].url and web_src[0].published_date == "2026-08-01"
            and web_src[0].filename == "www.pu.edu.pk" and web_src[0].title == "PU Admissions 2026-27"):
        ok("web source carries url / host / published_date / title")
    else:
        bad(f"web source wrong: {[(s.url, s.published_date, s.filename, s.title) for s in result.sources]}")

    # 3. Recency question with PDF context -> web triggered (WEB_SEARCH_ON_RECENCY=true)
    llm = RecordingLLM(CONFLICT_ANSWER)
    web = FakeWebSearch(results=[official_web_source("Live web claims the fee is Rs. 99,000 for 2026-27")])
    result = make_rag(KEPT_PDF, llm, web).answer("What is the latest fee for 2026-27?")
    if len(web.calls) == 1:
        ok("recency question -> web search triggered despite PDF context")
    else:
        bad(f"recency question: expected 1 web call, got {len(web.calls)}")
    if "conflict" in result.answer.lower():
        ok("conflict answer preserved verbatim")
    else:
        bad("conflict flag dropped from answer")
    kinds = {s.source_type for s in result.sources}
    if kinds == {"pdf", "web"}:
        ok("conflict case carries both PDF and web sources, never silently resolved")
    else:
        bad(f"conflict sources wrong: {[(s.source_type, s.filename) for s in result.sources]}")

    # 4. Unofficial-only results -> safe refusal, LLM NOT called
    llm = RecordingLLM("should never be shown")
    web = FakeWebSearch(results=[])  # provider had only off-domain junk (filtered out)
    result = make_rag(EMPTY_PDF, llm, web).answer("What is the current admission schedule?")
    if result.status == "no_relevant_context" and llm.prompts == [] and result.sources == []:
        ok("unofficial-only -> existing safe refusal; LLM not called; no sources claimed")
    else:
        bad(f"status={result.status}, prompts={len(llm.prompts)}, sources={len(result.sources)}")

    # 5a. Web timeout -> kept empty => refusal, LLM NOT called
    llm = RecordingLLM("should never be shown")
    web = FakeWebSearch(raise_exc=WebSearchTimeoutError("timeout"))
    result = make_rag(EMPTY_PDF, llm, web).answer("current admission schedule?")
    if result.status == "no_relevant_context" and llm.prompts == []:
        ok("web timeout + no PDF -> existing safe refusal (no LLM call, no exception)")
    else:
        bad(f"status={result.status}, prompts={len(llm.prompts)}")

    # 5b. Web timeout -> PDF kept => PDF-only answer still works
    llm = RecordingLLM(GOOD_PDF_ANSWER)
    web = FakeWebSearch(raise_exc=WebSearchTimeoutError("timeout"))
    result = make_rag(KEPT_PDF, llm, web).answer("What is the fee for 2026-27?")
    kinds = {s.source_type for s in result.sources}
    if result.status == "answered" and kinds == {"pdf"}:
        ok("web failure + PDF context -> graceful PDF-only answer")
    else:
        bad(f"status={result.status}, kinds={kinds}")

    # 5c. Quota exhaustion (429/432 family) -> same graceful path
    llm = RecordingLLM(GOOD_PDF_ANSWER)
    web = FakeWebSearch(raise_exc=WebSearchRateLimitError("HTTP 432"))
    result = make_rag(KEPT_PDF, llm, web).answer("What is the fee for 2026-27?")
    if result.status == "answered":
        ok("quota exhaustion + PDF context -> PDF-only answer, no exception")
    else:
        bad(f"status={result.status}")

    # 5d. HTTP 5xx + no PDF -> refusal
    llm = RecordingLLM("should never be shown")
    web = FakeWebSearch(raise_exc=WebSearchUpstreamError("HTTP 500"))
    result = make_rag(EMPTY_PDF, llm, web).answer("current admission schedule?")
    if result.status == "no_relevant_context" and llm.prompts == []:
        ok("HTTP 5xx + no PDF -> existing safe refusal")
    else:
        bad(f"status={result.status}")

    # 6. Master switch off -> web never called, even for recency questions
    with patched(web_search_enabled=False):
        llm = RecordingLLM(GOOD_PDF_ANSWER)
        web = FakeWebSearch(results=[official_web_source("should never be fetched")])
        result = make_rag(KEPT_PDF, llm, web).answer("what is the latest fee for 2026-27?")
        if web.calls == [] and result.status == "answered":
            ok("WEB_SEARCH_ENABLED=false -> web never called (master switch)")
        else:
            bad(f"web calls with switch off: {web.calls}")

    # 7. Verifier: web figures traceable to web context pass; invented fail
    # NOTE: the verifier's contract is number-FOLLOWED-by-unit (e.g., "75%",
    # "24 years", "2.00 gpa"). Currency written "Rs. 99,000" (unit first) is a
    # known blind spot of the existing verifier — reported, not fixed here.
    unsupported = find_unsupported_claims(
        "Attendance must be 75%.",
        ["Live web claims attendance must be 75%."],
        "attendance?",
    )
    if not unsupported:
        ok("verifier accepts figure present in web context")
    else:
        bad(f"verifier rejected web-grounded figure: {unsupported}")
    unsupported = find_unsupported_claims(
        "Attendance must be 99%.",
        ["Live web claims attendance must be 75%."],
        "attendance?",
    )
    if unsupported:
        ok("verifier refuses invented figure even with web context present")
    else:
        bad("verifier FAILED to refuse invented 99% figure")

# ================================================= D. health block
print("\n[D] /api/health web_search block (no key leakage)")
from fastapi.testclient import TestClient
from app.main import app

with patched(web_search_enabled=True, web_search_on_recency=True, tavily_api_key=TEST_KEY):
    response = TestClient(app).get("/api/health")
    payload = response.json()
    block = payload.get("web_search", {})
    if block.get("enabled") is True and block.get("api_key_configured") is True:
        ok("health reports enabled=True, api_key_configured=True")
    else:
        bad(f"health web_search block wrong: {block}")
    if TEST_KEY not in json.dumps(payload):
        ok("API key never appears in health response")
    else:
        bad("API KEY LEAKED in health response")

# ================================================= E. defaults
print("\n[E] safe defaults")
default_enabled = Settings.model_fields["web_search_enabled"].default
default_recency = Settings.model_fields["web_search_on_recency"].default
default_key = Settings.model_fields["tavily_api_key"].default
if default_enabled is False and default_recency is True and default_key == "":
    ok("defaults: WEB_SEARCH_ENABLED=false, WEB_SEARCH_ON_RECENCY=true, no key")
else:
    bad(f"unsafe defaults: enabled={default_enabled}, on_recency={default_recency}, key={default_key!r}")

# ================================================= F. observability logs
print("\n[F] observability logging (INFO events; privacy-conscious)")
import logging


class ListHandler(logging.Handler):
    """In-memory handler capturing formatted lines for assertions."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - test capture must never crash
            pass


def captured_lines(fn):
    handler = ListHandler()
    root = logging.getLogger()
    previous_root_level = root.level
    previous_disable = logging.root.manager.disable
    root.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    root.addHandler(handler)
    try:
        fn()
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_root_level)
        logging.disable(previous_disable)
    return handler.lines


def assert_contains(lines, needle, label):
    if any(needle in line for line in lines):
        ok(label)
    else:
        bad(f"{label} -> missing {needle!r} in logs: {lines}")


def assert_absent(lines, needle, label):
    if not any(needle in line for line in lines):
        ok(label)
    else:
        bad(f"{label} -> forbidden content {needle!r} found in logs")


# F1: successful mocked Tavily search logs results/host/score/date/usage
with patched(web_search_enabled=True, web_search_on_recency=True):
    svc, captured = make_http_service(
        httpx.Response(200, json={
            "results": [
                {"title": "PU Admissions", "url": "https://www.pu.edu.pk/adm",
                 "content": "current admission notice published for the 2026-27 session by the University of the Punjab", "score": 0.8, "published_date": "2026-08-01"},
                {"title": "CSE", "url": "https://cse.pu.edu.pk/adm",
                 "content": "admission notice for Department of Computer Science current session", "score": 0.5, "published_date": None},
            ],
            "response_time": 0.42,
            "usage": {"credits": 1},
        })
    )
    lines = captured_lines(lambda: svc.search("What is the latest PU admission notice?"))
    assert_contains(lines, "results_received=2", "success: results_received logged")
    assert_contains(lines, "host=www.pu.edu.pk score=0.8 published=2026-08-01", "success: host/score/date logged")
    assert_contains(lines, "host=cse.pu.edu.pk score=0.5 published=None", "success: second host/score/date logged")
    assert_contains(lines, "credits=1", "success: credits used logged")
    assert_contains(lines, "response_time=0.42", "success: response time logged")
    assert_contains(lines, "query_len=", "success: query length logged")

    # F1b: privacy — full question with a marker beyond 32 chars must not appear
    privacy_question = "padding to push past thirty-two chars UNIQMARKER_999"
    assert len(privacy_question) > 32
    lines = captured_lines(lambda: svc.search(privacy_question))
    assert_contains(lines, f"query_len={len(privacy_question)}", "privacy: query length present")
    assert_absent(lines, "UNIQMARKER_999", "privacy: marker beyond preview cut NOT logged")
    assert_absent(lines, privacy_question, "privacy: full question NOT logged")

    # F2: empty results logged honestly, results=0, no per-result lines
    svc, _ = make_http_service(httpx.Response(200, json={"results": [], "usage": {"credits": 1}, "response_time": 0.2}))
    lines = captured_lines(lambda: svc.search("q"))
    assert_contains(lines, "results_received=0", "empty results: logged as 0")
    if not any("tavily result[" in line for line in lines):
        ok("empty results: no per-result lines emitted")
    else:
        bad("empty results: unexpected per-result log lines")

    # F3: HTTP 500 logs WARNING; RAG degrades to safe refusal with no LLM call
    llm = RecordingLLM("should never be shown")
    web = FakeWebSearch(raise_exc=WebSearchUpstreamError("HTTP 500"))
    logs_holder: list[str] = []

    def _call_rag():
        make_rag(EMPTY_PDF, llm, web).answer("latest admission notice?")

    lines = captured_lines(_call_rag)
    assert_contains(lines, "Live web search unavailable", "failed search: warning captured")
    assert_contains(lines, "web search trigger=pdf_missing", "failed search: trigger=pdf_missing logged")
    assert_contains(lines, "retrieved pdf chunks kept=0", "failed search: kept=0 logged")
    if llm.prompts == []:
        ok("failed search: LLM not called (refusal preserved)")
    else:
        bad("failed search: LLM WAS called")

    # F4a: recency path logs trigger + kept + web block count + verifier outcome
    llm = RecordingLLM(CONFLICT_ANSWER)
    web = FakeWebSearch(results=[official_web_source("Live web claims the fee is Rs. 99,000 for 2026-27")])
    lines = captured_lines(
        lambda: make_rag(KEPT_PDF, llm, web).answer("What is the latest fee for 2026-27?")
    )
    assert_contains(lines, "web search trigger=recency", "recency: trigger=recency logged")
    assert_contains(lines, "retrieved pdf chunks kept=1", "recency: kept=1 logged")
    assert_contains(lines, "groq prompt: pdf_blocks=1 web_blocks=1", "recency: web_blocks=1 in groq prompt")
    assert_contains(lines, "verifier outcome=accepted", "recency: verifier outcome=accepted logged")

    # F4b: pdf_missing path with recency-free question
    llm = RecordingLLM(GOOD_WEB_ANSWER)
    web = FakeWebSearch(results=[official_web_source("current admission window August 2026 for 2026-27")])
    lines = captured_lines(
        lambda: make_rag(EMPTY_PDF, llm, web).answer("when is the admission window?")
    )
    assert_contains(lines, "web search trigger=pdf_missing", "pdf_missing: trigger logged")
    assert_contains(lines, "retrieved pdf chunks kept=0", "pdf_missing: kept=0 logged")
    assert_contains(lines, "groq prompt: pdf_blocks=0 web_blocks=1", "pdf_missing: web_blocks=1 in groq prompt")

    # F4c: ordinary PDF question -> web never called, no trigger line
    llm = RecordingLLM(GOOD_PDF_ANSWER)
    web = FakeWebSearch(results=[official_web_source("junk")])
    lines = captured_lines(
        lambda: make_rag(KEPT_PDF, llm, web).answer("What is the fee for 2026-27?")
    )
    if web.calls == [] and not any("trigger=" in line for line in lines):
        ok("ordinary PDF question: web not called, no trigger line")
    else:
        bad(f"ordinary PDF question: web called={web.calls}")
    assert_contains(lines, "retrieved pdf chunks kept=1", "ordinary PDF question: kept=1 logged")

    # F5: verifier refuses -> outcome logged, safe refusal returned, LLM not trusted
    llm = RecordingLLM("Attendance must be 99%.")
    web = FakeWebSearch(results=[official_web_source("Live web claims attendance must be 75%.")])
    result_holder: dict = {"result": None}

    def _call_refused():
        result_holder["result"] = make_rag(EMPTY_PDF, llm, web).answer("current attendance rule?")

    lines = captured_lines(_call_refused)
    assert_contains(lines, "verifier outcome=refused", "verifier refused: outcome logged")
    result_ref = result_holder["result"]
    if result_ref is not None and result_ref.status == "no_relevant_context":
        ok("verifier refused: safe refusal preserved (status=no_relevant_context)")
    else:
        bad(f"verifier refused: status={getattr(result_ref, 'status', None)}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
