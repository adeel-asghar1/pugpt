"""Release answer-quality regression tests — mocked generation only.

Covers the three answer-quality fixes from the release checklist:
  1. Admission age: 24-year vs 26-year figures are INTAKE-SPECIFIC
     (regular 1st-semester = 24y; BS 5th-semester/ADP = 26y; No age limit for
     Self-Supporting/Replica/Afternoon/Evening) — must never be cross-applied.
  2. Fees: "A" (program-specific 2026-27) vs "B" Service Charges vs one-time
     admission dues; no universal first-semester fee without the programme.
  3. GPA vs CGPA: semester GPA vs cumulative CGPA, first-semester vs
     later-semester thresholds, CGPA from individual courses.

Plus the Professional Presentation rules: direct-answer-first, Markdown
formatting guidance, bullet lists for conditions, tables only for comparisons,
and no model-generated Sources section (citations render from retrieved metadata).

Deterministic: no live model calls of any kind. RAG integration tests use a
mocked LLM and the REAL knowledge base read-only.

Run from the backend/ directory:
    python scripts/test_answer_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts import PUGPT_SYSTEM_PROMPT
from app.services.rag_service import RAGService, find_unsupported_claims

failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


print("=" * 78)
print("Release answer-quality — regression tests (mocked)")
print("=" * 78)

# ------------------------------------------------------- 1. admission age
print("\n[admission age — intake-specific figures]")
for marker in [
    "regular 1st-semester admission",
    "not more than 24 years",
    "not more than 26 years",
    "BS (5th Semester)",
    "Self Supporting, Replica, Afternoon and Evening programmes have no age limit",
    "Never apply one intake's age limit to another",
]:
    if marker.lower() in PUGPT_SYSTEM_PROMPT.lower():
        ok(f"prompt carries age-intake rule: {marker!r}")
    else:
        bad(f"age-intake rule missing: {marker!r}")

# ------------------------------------------------------------- 2. fees
print("\n[fees — A/B/admission dues distinctions]")
for marker in [
    '"A" Fee',
    '"B" Service Charges',
    "Rs. 8,500",
    "Rs. 4,800",
    "Rs. 950",
    "Registration Fee Rs. 9,000",
    "Library Security Rs. 5,000",
    "Rs. 500",
    "never present one universal fee",
]:
    if marker.lower() in PUGPT_SYSTEM_PROMPT.lower():
        ok(f"fee category rule present: {marker!r}")
    else:
        bad(f"fee category rule missing: {marker!r}")

# ------------------------------------------------------- 3. GPA vs CGPA
print("\n[GPA vs CGPA]")
for marker in [
    "first-semester",
    "1st Semester",
    "> 2.00",
    "1.70",
    "individual courses",
    "not from semester GPAs",
    "last-semester requirement",
]:
    if marker.lower() in PUGPT_SYSTEM_PROMPT.lower():
        ok(f"GPA/CGPA distinction present: {marker!r}")
    else:
        bad(f"GPA/CGPA distinction missing: {marker!r}")

# ------------------------------------------------- presentation rules
print("\n[presentation — professional markdown answers]")
for marker in [
    "direct answer",
    "Markdown",
    "bold",
    "bullet",
    'do not generate a "sources" heading',
    "application displays citations automatically",
    "Never force",
]:
    if marker.lower() in PUGPT_SYSTEM_PROMPT.lower():
        ok(f"presentation rule present: {marker!r}")
    else:
        bad(f"presentation rule missing: {marker!r}")

# ---------------------------- verifier — source-faithful figures pass
print("\n[verifier — figures traceable to context pass]")
unsupported = find_unsupported_claims(
    "The minimum CGPA is 2.00 for the last semester.",
    ["A student has to obtain CGPA 2.00 in the last semester for the award of degree."],
    "degree question",
)
if not unsupported:
    ok("verifier passes faithful figures")
else:
    bad(f"verifier wrongly refused: {unsupported}")
unsupported2 = find_unsupported_claims(
    "The age limit is 99 years.",
    ["Applicants age must not be more than 24 years"],
    "age question",
)
if unsupported2:
    ok("verifier refuses invented figures")
else:
    bad("verifier FAILED to refuse invented 99 years figure")

# ------------------------------------------------- mocked RAG integration
print("\n[mocked RAG — answer passes through unchanged; sources attached]")
try:
    from app.services.embedding_service import embedding_service
    from app.services.rag_service import RAGService as _R

    GOOD_MARKDOWN_ANSWER = (
        "**The minimum attendance requirement is 75%.**\n\n"
        "- A student must attend **at least 75%** of classes in a course to sit the exams.\n"
        "- Students below 75% must repeat the course.\n\n"
        "Sources: Semester-Rules-Undergraduate-OCR-VERIFIED.pdf — page 6"
    )

    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, system=None, timeout=None):
            self.calls.append(prompt)
            return GOOD_MARKDOWN_ANSWER

    class FakeWebSearch:
        """Hermetic: the live-knowledge-base check above must stay offline."""

        def search(self, query, **kwargs):
            from app.services.web_search_service import WebSearchResult
            return WebSearchResult(results=[])

    rag = _R(llm=RecordingLLM(), web_search=FakeWebSearch())
    result = rag.answer("What is the minimum attendance requirement?")
    if result.answer == GOOD_MARKDOWN_ANSWER:
        ok("markdown content passes through byte-identically to the client")
    else:
        bad("markdown content was altered in transit")
    if result.sources and all(s.filename.endswith(".pdf") for s in result.sources):
        ok(f"sources attach correctly -> {[(s.filename, s.page_number) for s in result.sources]}")
    else:
        bad(f"bad sources: {[(s.filename, s.page_number) for s in result.sources]}")
except Exception as exc:  # noqa: BLE001
    bad(f"integration failed: {exc}")

# ------------------------------------- live retrieval (embeddings only)
print("\n[live retrieval — fee question surfaces the fee schedule (no LLM)]")
try:
    from app.services.embedding_service import embedding_service
    from app.services.chroma_service import chroma_service

    vec = embedding_service.embed_text("What is the first-semester fee for a regular BS student?")
    res = chroma_service.query(vec, top_k=5)
    files = [m.get("filename") for m in (res["metadatas"] or [[]])[0]]
    dists = res["distances"][0]
    if any("Dues-Fee-Schedule" in (f or "") for f in files):
        ok(f"fee schedule IS in top-5 (closest: {[round(d, 4) for d in dists]}) -> fee query eligible for grounded answer")
    else:
        bad(f"fee schedule NOT retrieved: {files}")
except Exception as exc:  # noqa: BLE001
    bad(f"retrieval check failed: {exc}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
