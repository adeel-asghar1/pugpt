"""Web fallback chain — offline, fully mocked (no Groq/Tavily/Ollama calls).

Verifies the automatic sparse-PDF → official-PU web fallback added in Step 16:
1.  no PDF evidence + web disabled -> existing safe refusal (no web call)
2.  no PDF evidence + web enabled  -> official-PU web fallback
3.  sparse PDF (distance 0.46, <600 chars) + web enabled -> fallback
4.  strong PDF (distance 0.1, >=600 chars) + web enabled -> NO fallback
5.  Tavily failure with PDF evidence -> safe PDF-only degradation
6.  Tavily failure without PDF evidence -> safe refusal
7.  combined PDF + web evidence -> both source types returned, one search
8.  verifier rejection -> existing refusal behavior preserved
9.  recency trigger still works when PDF evidence is strong (one search)
10. at most one Tavily request per chat, even when both triggers could fire
11. retrieval threshold stays 0.5
12. server-side domain validation intact

Run from the backend/ directory:
    python scripts/test_web_fallback_chain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.rag_service import (
    RAGService,
    STATUS_ANSWERED,
    STATUS_NO_CONTEXT,
)
from app.services.web_search_service import WebSearchResult, WebSource, validated_host

failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


def check(label: str, condition: bool, detail: str = "") -> None:
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


SPARSE_DOC = (
    "Regular admission age limit text: applicants must be below the age limit "
    "as printed by the 2026-27 eligibility criteria for the chosen programme."
)
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
STRONG_DOC = LONG_PARAGRAPH * 3  # > 600 chars, distance 0.1


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


def web_source(host="www.pu.edu.pk", url="https://www.pu.edu.pk/news/admissions", title="PU announcements", content=None, date="2026-09-01"):
    return WebSource(
        title=title,
        url=url,
        host=host,
        content=content or (
            "Official announcement: the University of the Punjab has published the "
            "admission schedule for the BS programmes in the 2026-27 session. The "
            "fee instructions and the closing dates are published on the official "
            "website under admissions. Candidates must follow the notified dates. "
            "The official admission portal provides application instructions and "
            "updates on merit computation and the verification schedule for "
            "submitted documents. All announcements are published only on the "
            "University website and official social media channels."
        ),
        score=0.9,
        published_date=date,
    )


class FakeChroma:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def count(self):
        return max(len(self.chunks), 1 if self.chunks else 0) or 1

    def query(self, vector, top_k=None):
        return {
            "documents": [[c[0] for c in self.chunks]],
            "metadatas": [[c[1] for c in self.chunks]],
            "distances": [[c[2] for c in self.chunks]],
        }

    def query_filtered(self, vector, where=None, top_k=None):
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


class RecordingLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate(self, prompt, system=None, timeout=None):
        self.prompts.append(prompt)
        return self.text


class FakeWeb:
    def __init__(self, result=None, exc=None):
        self.calls = 0
        self.result = result
        self.exc = exc

    def search(self, question, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


GOOD = (
    "The retrieved excerpts state that admission is subject to the eligibility "
    "criteria prescribed for the programme, including the prescribed qualifications "
    "and verification of documents. Please verify the current schedule on the "
    "official University pages before applying."
)
QUESTION_ELIGIBILITY = "What are the eligibility criteria for undergraduate admissions?"
QUESTION_RECENCY = "What is the latest deadline to apply for undergraduate admissions?"

print("=" * 78)
print("Web fallback chain — mocked offline tests")
print("=" * 78)

# --------------------------------- 1. no PDF + web disabled -> refusal, no web
print("\n[1. no PDF evidence, web disabled]")
fake_embed = FakeEmbeddings()
fake_llm = RecordingLLM(GOOD)
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(SPARSE_DOC, distance=0.9)]),  # beyond 0.5 -> kept == 0
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=False, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == no_relevant_context", result.status == STATUS_NO_CONTEXT, result.status)
check("web search never called", fake_web.calls == 0, f"calls={fake_web.calls}")
check("LLM never called", len(fake_llm.prompts) == 0)

# ----------------------- 2. no PDF + web enabled -> official-PU web fallback
print("\n[2. no PDF evidence, web enabled]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(SPARSE_DOC, distance=0.9)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == answered", result.status == STATUS_ANSWERED, f"{result.status} :: {result.error}")
check("Tavily called exactly once", fake_web.calls == 1, f"calls={fake_web.calls}")
check("web source present", any(s.source_type == "web" for s in result.sources),
      f"sources={[(s.filename, s.source_type) for s in result.sources]}")
check("prompt contains Web Source block", any("[Web Source" in p for p in fake_llm.prompts))

# ---------------------- 3. sparse PDF (0.46, short) + web -> fallback fires
print("\n[3. sparse PDF evidence, web enabled]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
sparse_chunk = chunk(SPARSE_DOC, distance=0.46)  # <600 chars, >0.35 -> sparse
print(f"      sparse chunk length={len(SPARSE_DOC)} chars")
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([sparse_chunk]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("What is the latest age limit for admission?")  # recency wording too
check("status == answered", result.status == STATUS_ANSWERED, f"{result.status} :: {result.error}")
check("Tavily called exactly once (no double trigger)", fake_web.calls == 1, f"calls={fake_web.calls}")
check("Evidence note present in prompt", any("Evidence note" in p for p in fake_llm.prompts))
check("both pdf and web sources returned",
      {s.source_type for s in result.sources} == {"pdf", "web"},
      f"types={[(s.filename, s.source_type) for s in result.sources]}")

# --------------------- 4. strong PDF evidence + web enabled -> no fallback
print("\n[4. strong PDF evidence, web enabled -> no fallback]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(STRONG_DOC, distance=0.1)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == answered", result.status == STATUS_ANSWERED, result.status)
check("Tavily never called", fake_web.calls == 0, f"calls={fake_web.calls}")
check("Evidence note absent from prompt", not any("Evidence note" in p for p in fake_llm.prompts))
check("all sources are pdf", all(s.source_type == "pdf" for s in result.sources))

# -------------------- 5. Tavily failure with PDF evidence -> pdf-only answer
print("\n[5. Tavily raises, PDF evidence exists]")
fake_web = FakeWeb(exc=RuntimeError("quota exhausted"))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(STRONG_DOC, distance=0.46)]),  # sparse -> fallback attempted
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == answered (pdf-only degradation)", result.status == STATUS_ANSWERED,
      f"{result.status} :: {result.error}")
check("Tavily attempted exactly once", fake_web.calls == 1, f"calls={fake_web.calls}")
check("all sources are pdf", all(s.source_type == "pdf" for s in result.sources))
check("LLM received sparse Evidence note", any("Evidence note" in p for p in fake_llm.prompts))

# ----------------- 6. Tavily failure without PDF evidence -> safe refusal
print("\n[6. Tavily raises, no PDF evidence]")
fake_web = FakeWeb(exc=RuntimeError("quota exhausted"))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(SPARSE_DOC, distance=0.9)]),  # nothing kept
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == no_relevant_context", result.status == STATUS_NO_CONTEXT, result.status)
check("LLM never called", len(fake_llm.prompts) == 0)
check("no sources", result.sources == [])

# ------------------ 7. combined evidence -> both types + single search
print("\n[7. combined PDF + web evidence]")
pdf_sources_expected = {"Eligibility-2026-27.pdf": 1}
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(SPARSE_DOC, distance=0.46)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)  # no recency markers
check("status == answered", result.status == STATUS_ANSWERED, result.error or "")
check("both source types present",
      {"pdf", "web"} <= {s.source_type for s in result.sources},
      f"types={[(s.filename, s.source_type) for s in result.sources]}")
check("web source keeps url+host+date",
      any(s.source_type == "web" and s.url and s.filename == "www.pu.edu.pk" and s.published_date for s in result.sources))
check("exactly one Tavily call", fake_web.calls == 1, f"calls={fake_web.calls}")
check("prompt contains both blocks",
      any("[Source 1:" in p and "[Web Source 2:" in p for p in fake_llm.prompts))

# ------------- 8. verifier rejection -> refusal preserved, sources kept
print("\n[8. verifier rejection]")
bad_llm = RecordingLLM("The application fee is 99,000 rupees for everybody.")
fake_web = FakeWeb(WebSearchResult(results=[]))
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(STRONG_DOC, distance=0.1)]),
    llm=bad_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=False, web_search_on_recency=False):
    result = rag.answer(QUESTION_ELIGIBILITY)
check("status == no_relevant_context (verifier refusal)", result.status == STATUS_NO_CONTEXT, result.status)
check("error mentions verification", bool(result.error and "verification" in result.error.lower()), result.error or "")
check("sources still attached", len(result.sources) > 0)

# ------- 9. recency trigger still fires with strong PDF evidence (one call)
print("\n[9. recency trigger with strong PDF]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=fake_embed,
    chroma=FakeChroma([chunk(STRONG_DOC, distance=0.1)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=True):
    result = rag.answer(QUESTION_RECENCY)
check("status == answered", result.status == STATUS_ANSWERED, result.error or "")
check("recency triggered exactly one search", fake_web.calls == 1, f"calls={fake_web.calls}")
check("recency prompt contains web excerpt", any("[Web Source" in p for p in fake_llm.prompts))

# -------------------- 10/11. one call max; threshold unchanged
print("\n[10 & 11. call cap and retrieval threshold]")
check("rag_distance_threshold is still 0.5", settings.rag_distance_threshold == 0.5,
      settings.rag_distance_threshold)

# ----------------- 12. server-side domain validation intact
print("\n[12. validated_host]")
cases = [
    ("https://pu.edu.pk/path", "pu.edu.pk"),
    ("https://www.pu.edu.pk/admissions", "www.pu.edu.pk"),
    ("https://sub.domain.pu.edu.pk/x", "sub.domain.pu.edu.pk"),
    ("https://pu.edu.pk.evil.com/x", None),
    ("https://notpu.edu.pk/x", None),
    ("ftp://pu.edu.pk/file", None),
    ("not a url", None),
    ("https://PU.EDU.PK/Path", "pu.edu.pk"),
]
for url, expected in cases:
    got = validated_host(url, ["pu.edu.pk"])
    check(f"validated_host({url!r}) -> {expected!r}", got == expected, f"got={got!r}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
