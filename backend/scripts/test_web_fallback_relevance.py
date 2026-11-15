"""Sparse-PDF relevance detection — offline, fully mocked.

Step 18: substantial-but-off-topic PDF evidence must NOT be treated as strong
enough to suppress the official-PU web fallback. Verifies:

A. Substantial admissions chunks + a November-2026 campus-closure question
   -> fallback fires with trigger=sparse_pdf (not sufficiency).
B. Substantial chunks that directly address the question -> NO fallback.
C. Kept == 0 with web enabled -> trigger pdf_missing (unchanged).
D. Distance > 0.35 or < 600 chars still classifies sparse (existing heuristic).
E. At most one Tavily request per chat.
F. WEB_SEARCH_ENABLED=false -> no Tavily call at all.

The relevance helpers are also exercised directly (deterministic, pure).
Run from the backend/ directory:
    python scripts/test_web_fallback_relevance.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.rag_service import (
    RAGService,
    STATUS_ANSWERED,
    STATUS_NO_CONTEXT,
    _evidence_relevance,
    _question_content_terms,
)
from app.services.web_search_service import WebSearchResult, WebSource

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


BIG_ADMISSIONS = (
    "The 2026-27 Admission Regulations state that a candidate seeking admission "
    "to a BS programme must satisfy the eligibility criteria prescribed for that "
    "programme, including the required level of education and the minimum marks. "
    "The University bases merit computation on the formula notified for each "
    "session and each programme category, and provisional admission is always "
    "subject to verification of all submitted documents at the time of admission. "
    "The closing date for applications, the fee schedule for regular and self "
    "supporting seats, the admission dues, the verification schedule and the "
    "list of required certificates are all governed by these regulations. Any "
    "discrepancy, misrepresentation or failure to satisfy these conditions may "
    "lead to cancellation of admission or withholding of degree, at the sole "
    "discretion of the University authorities under the applicable rules. "
) * 6  # 900+ chars per chunk — substantial text about admissions only

QUESTION_CLOSURES = (
    "Has the University of the Punjab issued any official notice during "
    "November 2026 about campus-wide closures or changes to the class schedule?"
)
QUESTION_ADMISSIONS = "What are the eligibility criteria for undergraduate admissions?"


def admissions_chunk(i, distance=0.1):
    return (
        BIG_ADMISSIONS + f" chunk {i}.",
        {
            "filename": "Admission-Regulations-2026-27.pdf",
            "page_number": i + 1,
            "chunk_index": i,
            "source_path": "Admission-Regulations-2026-27.pdf",
            "doc_id": "admission",
            "category": "eligibility",
            "document_type": "eligibility",
            "effective_session": "2026-27",
            "verification_status": "verified",
        },
        distance,
    )


FEE_DOC = (
    "Dues Schedule Undergraduate for the year 2026-27. The BS regular programme "
    "carries programme-specific tuition payable every semester. Service Charges "
    "are Rs. 8,500. Examination Fee is Rs. 4,800. Sports and Mosque Fund is Rs. 950 "
    "in the Fall semester only. Registration Fee Rs. 9,000 applies at admission. "
    "Library Security Rs. 5,000 is refundable. PU Development Fund is Rs. 500. "
    "The complete A Fee category table lists programme-specific amounts per "
    "programme for the regular and self supporting seats. Fees must be submitted "
    "through the official channels before the deadline shown in the schedule. "
) * 6


def fee_chunk(i, distance=0.12):
    return (
        FEE_DOC + f" chunk {i}.",
        {
            "filename": "Dues-Fee-Schedule-for-the-Year-2026-27.pdf",
            "page_number": i + 1,
            "chunk_index": i,
            "source_path": "Dues-Fee-Schedule-for-the-Year-2026-27.pdf",
            "doc_id": "fee",
            "category": "fees",
            "document_type": "fee schedule",
            "effective_session": "2026-27",
            "verification_status": "verified",
        },
        distance,
    )


def web_source():
    return WebSource(
        title="PU announcements",
        url="https://www.pu.edu.pk/announcements/2026-11",
        host="www.pu.edu.pk",
        content=(
            "Official announcement: in November 2026 the University of the Punjab "
            "notified a temporary adjustment to classes due to an interim semester "
            "review. Candidates should consult the official portal for schedule "
            "changes and campus closures during that month."
        ),
        score=0.9,
        published_date="2026-11-03",
    )


class FakeChroma:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def count(self):
        return max(len(self.chunks), 1)

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
    "The retrieved excerpts do not establish a November 2026 campus-wide closure. "
    "The admission regulations describe the eligibility criteria and the fee dues "
    "process, and the official website notice cited above carries the schedule "
    "adjustment for November 2026. Verify on the official portal before relying."
)

print("=" * 78)
print("Sparse-PDF relevance — mocked offline tests")
print("=" * 78)

# ---------------- direct helper behaviour (deterministic, no I/O) ------------
print("\n[helpers]")
terms = _question_content_terms(QUESTION_CLOSURES)
check("content terms extracted, stopwords removed",
      len(terms) >= 5 and "the" not in terms and "november" in terms and "2026" in terms,
      f"terms={terms}")
kept = [admissions_chunk(i) for i in range(5)]
relevant, hits, total, tag, date_terms = _evidence_relevance(kept, QUESTION_CLOSURES)
check(f"off-topic admission evidence judged not relevant ({hits}/{total})", not relevant)
approx = sum(
    1 for wanted in terms
    if any(wanted in str(item[0]).lower() for item in kept)
)
check("off-topic raw coverage below one third",
      approx / len(terms) < 0.34, f"approx={approx}/{len(terms)}")
relevant_b, hits_b, total_b, _tag_b, _d_b = _evidence_relevance(kept, QUESTION_ADMISSIONS)
check(f"on-topic eligibility evidence judged relevant ({hits_b}/{total_b})", relevant_b)
check("on-topic judged with ok tag", _tag_b == "ok", _tag_b)

chunk = admissions_chunk(0, distance=0.9)
relevant_far, _h, _t, _tag, _d = _evidence_relevance([chunk], QUESTION_ADMISSIONS)
check("relevance of any chunk classifies on evidence content itself", relevant_far)

# ------- A. off-topic substantial PDFs -> sparse_pdf trigger with fallback
print("\n[A. off-topic substantial PDFs trigger sparse_pdf]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_CLOSURES)
check("status == answered", result.status == STATUS_ANSWERED,
      f"{result.status} :: {result.error}")
check("exactly one Tavily call", fake_web.calls == 1, f"calls={fake_web.calls}")
check("web source returned",
      any(s.source_type == "web" for s in result.sources),
      f"sources={[(s.filename, s.source_type) for s in result.sources]}")
check("prompt contains sparse Evidence note",
      any("Evidence note" in p for p in fake_llm.prompts))

# --------------------- B. on-topic evidence -> NO fallback ------------------
print("\n[B. on-topic substantial PDFs -> no fallback]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
fee_chunks = [fee_chunk(i, 0.12) for i in range(5)]
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(fee_chunks),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("status == answered", result.status == STATUS_ANSWERED, result.error or "")
check("web search never called", fake_web.calls == 0, f"calls={fake_web.calls}")

# --------------------- C. kept == 0 + web enabled -> pdf_missing ------------
print("\n[C. no PDF evidence -> pdf_missing]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(0, 0.9)]),  # distance > 0.5 -> nothing kept
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("status == answered", result.status == STATUS_ANSWERED, result.error or "")
check("Tavily called exactly once", fake_web.calls == 1, f"calls={fake_web.calls}")

# -------- D. distance / length checks still classify sparse -----------------
print("\n[D. existing distance/length heuristic remains effective]")
print("      D1: distance 0.46 (relevant topic, too far) -> sparse")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk(0, 0.46)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("What is the fee structure?")
check("distance 0.46 -> fallback fired",
      fake_web.calls == 1, f"calls={fake_web.calls}")

print("      D2: short but relevant PDF (distance 0.1, <600 chars) -> sparse")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
short_chunk = admissions_chunk(0, 0.1)
short_chunk = (short_chunk[0][:200], short_chunk[1], 0.1)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([short_chunk]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("short chunk -> fallback fired", fake_web.calls == 1, f"calls={fake_web.calls}")

# ------------------ E. at most one web search per chat ----------------------
print("\n[E. single-call guarantee under overlap]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=True):
    result = rag.answer("What is the latest campus closure announcement for November 2026?")
check("both sparse and recency cues -> exactly one search", fake_web.calls == 1,
      f"calls={fake_web.calls}")

# --------------------- F. web disabled -> zero web calls --------------------
print("\n[F. WEB_SEARCH_ENABLED=false -> no Tavily call]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=False, web_search_on_recency=False):
    result = rag.answer(QUESTION_CLOSURES)
check("web never called", fake_web.calls == 0, f"calls={fake_web.calls}")
check("still answered from PDFs", result.status == STATUS_ANSWERED, result.status)

# ------------------------------------------------------------------ Step 18C
SUBSTANTIAL_ADMISSIONS_ONLY = (
    "Dues schedule undergraduate for the year 2026-27 listing programme-specific "
    "admission dues payable in each semester. Service Charges Rs. 8,500. "
    "Examination Fee Rs. 4,800. Sports and Mosque Fund Rs. 950 Fall semester. "
    "Registration Fee Rs. 9,000. PU Development Fund Rs. 500. Library Security "
    "Rs. 5,000 refundable. Every paragraph describes the dues and the schedule. "
) * 6
print("\n[18C-1b. exactly one substantive hit (admission) -> must trigger fallback]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
mini = [(
    SUBSTANTIAL_ADMISSIONS_ONLY,
    {"filename": "Dues.pdf", "page_number": 1, "chunk_index": 0, "source_path": "Dues.pdf",
     "doc_id": "dues", "category": "fees", "document_type": "fee schedule",
     "effective_session": "2026-27", "verification_status": "verified"},
    0.1,
)]
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(mini),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("does a hostel closure notice exist in 2026-11")
check("only 'admissions' matched (1 lexical match) -> not sufficient -> fallback",
      fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18C-2. two substantive matches with >=0.34 coverage pass]")
mini_ok = [(
    SUBSTANTIAL_ADMISSIONS_ONLY,
    {"filename": "Dues.pdf", "page_number": 1, "chunk_index": 0, "source_path": "Dues.pdf",
     "doc_id": "dues", "category": "fees", "document_type": "fee schedule",
     "effective_session": "2026-27", "verification_status": "verified"},
    0.1,
)]
fee_llm = RecordingLLM(GOOD)
fee_web = FakeWeb(WebSearchResult(results=[web_source()]))
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(mini_ok),
    llm=fee_llm,
    web_search=fee_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("admission dues fee structure for 2026-27")
check("two substantive matches (admission, dues, fee) -> no fallback",
      fee_web.calls == 0, f"calls={fee_web.calls}")

print("\n[18C-3. matching only years/dates cannot establish relevance]")
date_chunk = [(
    SUBSTANTIAL_ADMISSIONS_ONLY,
    mini_ok[0][1],
    0.1,
)]
fake_web3 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm3 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(date_chunk),
    llm=fake_llm3,
    web_search=fake_web3,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("is there anything in november 2026 for 2027 2026-27 15")
check("no substantive match (only date/numeric tokens matched) -> fallback",
      fake_web3.calls == 1, f"calls={fake_web3.calls}")

print("\n[18C-4. two dates + one generic term that matches once -> not sufficient]")
generic = [(
    SUBSTANTIAL_ADMISSIONS_ONLY,
    mini_ok[0][1],
    0.1,
)]
fake_web4 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm4 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(generic),
    llm=fake_llm4,
    web_search=fake_web4,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("admission hostel 2026-11 2026")
check("dates + one generic match ('admission') + one unmatched generic -> below 2 substantive hits -> fallback",
      fake_web4.calls == 1, f"calls={fake_web4.calls}")

print("\n[18C-5. directly relevant evidence still avoids Tavily]")
fake_web5 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm5 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm5,
    web_search=fake_web5,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("on-topic question -> web never called", fake_web5.calls == 0, f"calls={fake_web5.calls}")

print("\n[18C-6. zero-content-term question: safe behaviour + diagnostic marker]")
log_records = []


class _Capture(logging.Handler):
    def emit(self, record):
        log_records.append(record.getMessage())


_handle = logging.getLogger()
_capture = _Capture()
_handle.addHandler(_capture)
_handle.setLevel(logging.INFO)
try:
    fake_web6 = FakeWeb(WebSearchResult(results=[web_source()]))
    fake_llm6 = RecordingLLM(GOOD)
    rag = RAGService(
        embeddings=FakeEmbeddings(),
        chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
        llm=fake_llm6,
        web_search=fake_web6,
    )
    with patched(web_search_enabled=True, web_search_on_recency=False):
        result = rag.answer("what is the the the the?")
finally:
    _handle.removeHandler(_capture)
check("zero content terms -> no Tavily call", fake_web6.calls == 0, f"calls={fake_web6.calls}")
check("zero content terms -> safe best-effort answer", result.status == STATUS_ANSWERED, result.status)
check("diagnostic marker logged",
      any("relevance_unavailable=no_content_terms" in line for line in log_records),
      f"records={log_records}")

print("\n[18C-7. missing PDF and distance/length guards remain effective]")
fake_web7 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm7 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(0, 0.9)]),  # nothing kept
    llm=fake_llm7,
    web_search=fake_web7,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("nothing kept -> pdf_missing one call", fake_web7.calls == 1, f"calls={fake_web7.calls}")

fake_web7b = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm7b = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk(0, 0.46)]),  # too far
    llm=fake_llm7b,
    web_search=fake_web7b,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("what is the fee?")
check("distance 0.46 -> sparse_pdf one call", fake_web7b.calls == 1, f"calls={fake_web7b.calls}")

fake_web7c = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm7c = RecordingLLM(GOOD)
tiny = [("fee admission.",
         {"filename": "x.pdf", "page_number": 1, "chunk_index": 0, "source_path": "x.pdf",
          "doc_id": "x", "category": "fees", "document_type": "fee schedule",
          "effective_session": "2026-27", "verification_status": "verified"},
         0.1)]
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma(tiny),
    llm=fake_llm7c,
    web_search=fake_web7c,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("admission fee?")
check("short chunk (<600 chars) -> sparse_pdf one call", fake_web7c.calls == 1, f"calls={fake_web7c.calls}")

print("\n[18C-8. at most one Tavily call per chat]")
fake_web8 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm8 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm8,
    web_search=fake_web8,
)
with patched(web_search_enabled=True, web_search_on_recency=True):
    result = rag.answer("What is the latest campus closure announcement for November 2026?")
check("sparse + recency cues overlap -> exactly one search", fake_web8.calls == 1,
      f"calls={fake_web8.calls}")

print("\n[18C-9. disabled web search -> zero calls]")
fake_web9 = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm9 = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm9,
    web_search=fake_web9,
)
with patched(web_search_enabled=False, web_search_on_recency=False):
    result = rag.answer(QUESTION_CLOSURES)
check("web never called", fake_web9.calls == 0, f"calls={fake_web9.calls}")
check("still answered from PDFs", result.status == STATUS_ANSWERED, result.status)

# ==========================================================================
# Step 18D — generic single-term refinement
# ==========================================================================
print("\n[18D-1. single-term 'merit' vs admissions evidence -> sparse_pdf]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("merit?")
check("only generic hit ('merit') -> fallback fires", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18D-2. single-term 'schedule' vs admissions evidence -> sparse_pdf]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("schedule?")
check("generic single-term 'schedule' -> fallback fires", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18D-3. single-term 'admission' vs admissions evidence -> sparse_pdf]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("admission?")
check("generic single-term 'admission' -> fallback fires", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18D-4. single-term 'fee' vs genuine fee evidence -> no Tavily call]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk(i, 0.12) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("genuine fee evidence -> no web call", fake_web.calls == 0, f"calls={fake_web.calls}")

print("\n[18D-5. single-term 'fee' vs incidental mention without amounts -> fallback]")
INCIDENTAL_FEE_DOC = (
    "The admission regulations describe the application procedure for the BS programmes, "
    "the published list of fees to be paid where applicable, and the verification schedule "
    "at the University. Applications close several weeks before the notified date, and all "
    "dues are subject to the approvals of the University authorities. Fees are determined by "
    "the University from time to time under the general clauses of the admission regulations. "
) * 6
incid_meta = {
    "filename": "Admission-Regulations-2026-27.pdf", "page_number": 2, "chunk_index": 0,
    "source_path": "Admission-Regulations-2026-27.pdf", "doc_id": "admission",
    "category": "admission", "document_type": "guidelines",
    "effective_session": "2026-27", "verification_status": "verified",
}
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([(INCIDENTAL_FEE_DOC + " chunk 0", incid_meta, 0.1)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("incidental fee mention (no amounts) -> fallback fires", fake_web.calls == 1,
      f"calls={fake_web.calls}")

print("\n[18D-6. multi-term two-hit behavior unchanged]")
mult_kept = []
for i in range(3):
    _d, _m, _dist = fee_chunk(i, 0.12)
    mult_kept.append((_d, _m, _dist, _dist * 0.75))
relevant_mt, hits_mt, total_mt, tag_mt, d_mt = _evidence_relevance(mult_kept, QUESTION_ADMISSIONS)
check(f"direct helper: multi-term question with >=2 hits passes ({hits_mt}/{total_mt})",
      relevant_mt and tag_mt == "ok",
      f"tag={tag_mt}")

print("\n[18D-7. date-only & zero-content-term behavior unchanged]")
terms = _question_content_terms("2026 2027 2026-27")
substantive = [t for t in terms if True][len(terms):]
relevant_dt, hits_dt, total_dt, tag_dt, d_dt = _evidence_relevance([], "2026 2027 2026-27")
check("date-only question -> safe relevant", relevant_dt and tag_dt == "date_only_terms",
      f"tag={tag_dt}")
relevant_nc, hits_nc, total_nc, tag_nc, d_nc = _evidence_relevance([], "what is the the the the?")
check("zero content terms -> safe relevant", relevant_nc and tag_nc == "no_content_terms",
      f"tag={tag_nc}")

print("\n[18D-8. missing-PDF and sparse-PDF behavior remains unchanged]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(0, 0.9)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer(QUESTION_ADMISSIONS)
check("nothing kept -> exactly one call", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18D-9. at most one Tavily call per request]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=True):
    result = rag.answer("merit?")
check("generic single-term under recency cues -> still exactly one call",
      fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18D-10. web disabled -> zero calls]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([admissions_chunk(i, 0.1) for i in range(5)]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=False, web_search_on_recency=False):
    result = rag.answer("merit?")
check("web never called", fake_web.calls == 0, f"calls={fake_web.calls}")

# ==========================================================================
# Step 18E — fee evidence must locally associate an amount with a fee word
# ==========================================================================
print("\n[18E-1. 'Tuition fee: Rs. 45,000' -> sufficient, no Tavily call]")


def fee_chunk_named(text, filename, distance=0.1):
    return (
        (text + " ") * 40,
        {
            "filename": filename,
            "page_number": 1,
            "chunk_index": 0,
            "source_path": filename,
            "doc_id": "fee",
            "category": "fees",
            "document_type": "fee schedule",
            "effective_session": "2026-27",
            "verification_status": "verified",
        },
        distance,
    )


fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named("Tuition fee: Rs. 45,000\nService Charges: Rs. 8,500", "Fee-Schedule.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("currency associated with 'fee' -> sufficient, no web call", fake_web.calls == 0,
      f"calls={fake_web.calls}")

print("\n[18E-2. 'Admission fee Rs. 10,000' -> sufficient]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named("Admission fee Rs. 10,000 for session 2026-27", "Admissions.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("'Admission fee Rs. 10,000' -> sufficient", fake_web.calls == 0, f"calls={fake_web.calls}")

print("\n[18E-3. 'The fee is Rs. 9,000' -> sufficient]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named("The fee is Rs. 9,000 payable at the start of term", "Fee.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("'The fee is Rs. 9,000' -> sufficient", fake_web.calls == 0, f"calls={fake_web.calls}")

print("\n[18E-4. unrelated fee mention + unrelated amount -> fails]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
unrelated_with_fee_mention = (
    "Merit list published 15-10-2026. Rs. 9,000 was the examination printing cost. "
    "Students must collect original documents and pay the hostel caution. The University "
    "issues these instructions every session. "
) * 16
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named(unrelated_with_fee_mention, "Merit.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("fee mention nowhere local to the amount -> fallback fires", fake_web.calls == 1,
      f"calls={fake_web.calls}")

print("\n[18E-5. 'The fee structure is available.' (no amount) -> fails]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named("The fee structure is available. Service Charges apply.", "Admissions.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("fee word without amount -> fallback fires", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18E-6. Rs. amount with no fee context -> fails]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fake_llm = RecordingLLM(GOOD)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk_named("Rs. 9,000 refund issued to students on verification.", "Refunds.pdf")]),
    llm=fake_llm,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("fee?")
check("Rs. without fee context -> fallback fires", fake_web.calls == 1, f"calls={fake_web.calls}")

print("\n[18E-7. multi-term fee question still passes on fee evidence]")
fake_web = FakeWeb(WebSearchResult(results=[web_source()]))
fee_llm_multi = RecordingLLM(
    "The admission dues are Rs. 55,000 and Rs. 8,500 service charges respectively."
)
rag = RAGService(
    embeddings=FakeEmbeddings(),
    chroma=FakeChroma([fee_chunk(i, 0.12) for i in range(5)]),
    llm=fee_llm_multi,
    web_search=fake_web,
)
with patched(web_search_enabled=True, web_search_on_recency=False):
    result = rag.answer("admission fees structure?")
check("multi-term question, fee evidence -> sufficient", fake_web.calls == 0,
      f"calls={fake_web.calls}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)

