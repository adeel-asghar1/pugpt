"""Focused verifier regression tests for currency-formatted claims — Step 23C.

Scope: ONLY `find_unsupported_claims` with unit-first currency claims
(Rs. 15,000 / Ps / PKR styles). Number-followed-by-unit claims
("75%", "24 years", "2.00 cgpa"), percentages, ages, academic thresholds and
session references must behave exactly as before.

Support rule for currency claims (new): the amount must be printed as money in
the context — Rs/PKR/rupees-marked ("Rs. 9,000") or comma-grouped thousands
("15,750", the PU fee-table format). Bare ungrouped digit runs (applicant IDs,
page numbers, marks, session years) are NOT currency evidence.

Run from the backend/ directory:
    python scripts/test_currency_verifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag_service import RAGService, find_unsupported_claims
from app.services.web_search_service import WebSearchResult

failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


def supported(answer: str, context: str, label: str) -> None:
    unsupported = find_unsupported_claims(answer, [context], "fee question")
    if not unsupported:
        ok(label)
    else:
        bad(f"{label} -> wrongly refused: {unsupported}")


def refused(answer: str, context: str, label: str) -> None:
    unsupported = find_unsupported_claims(answer, [context], "fee question")
    if unsupported:
        ok(label)
    else:
        bad(f"{label} -> NOT refused (claim accepted from unrelated/absurd context)")


print("=" * 78)
print("Currency-formatted number verification — Step 23C")
print("=" * 78)

# Realistic fee-document-style contexts (comma grouping and Rs.-marking are
# exactly how Dues-Fee-Schedule-for-the-Year-2026-27.pdf prints figures).
MARKED_CONTEXT = (
    "Note2: Registration Fee @ Rs.9,000 will be charged at the time of "
    "admission from students, not already registerd."
)
GROUPED_CONTEXT = (
    "2026-2027\nADP Sem. 1-6\nA Fee\n15,750\nBS regular\n21,000\n"
    "Service Charges Rs. 8,500 and Examination Fee Rs. 4,800 apply per semester; "
    "Sports & Mosque Fund Rs. 950 in the Fall semester only."
)
UNRELATED_CONTEXT = (
    "applicant id 15000; page 15000 of the form; 15000 marks weighted average; "
    "session 2026-27 covers 2026 only; form number 15750."
)
OTHER_CONTEXT = "Service Charges Rs. 8,500; Examination Fee Rs. 4,800."
# Fee-schedule-shaped context mirroring Dues-Fee-Schedule-for-the-Year-2026-27.pdf
# (page 2): FCIT BS CS row, B Service Charges, Examination Fee, and the Sports &
# Mosque Fund column printing a BARE, ungrouped "950" (the case that failed live).
FEE_SCHEDULE_CONTEXT = (
    "Dues Schedule Undergraduate (Regular/Morning) For the year 2026-27\n"
    "2026-2027\n"
    "FCIT (Software Engineering, Computer Science, Information Technology, "
    "Data Science, Aartificial Intelligence)\n"
    "55,000\n"
    "Service Charges\n8,500\n"
    "Examination Fee\n4,800\n"
    "Sports & Mosque Fund\n950\n"
    "29,950\n"
    "Note2: Registration Fee @ Rs.9,000 will be charged at the time of admission "
    "from students\n"
    "Sports & Mosque Fund\n950\n"
)
GOOD_BS_CS_ANSWER = (
    "BS CS (FCIT) fee structure for 2026-27: A Fee **Rs. 55,000**; "
    "Service Charges **Rs. 8,500**; Examination Fee **Rs. 4,800**; "
    "Sports & Mosque Fund **Rs. 950** in the Fall semester only; "
    "Registration Fee **Rs. 9,000** if not already registered."
)

# --------------------------------------- A. supported by context
print("\n[A] supported currency claims")
for answer in [
    "The registration fee is Rs. 9,000.",
    "The registration fee is Rs.9,000.",
    "The registration fee is Rs 9000.",
    "The registration fee is PKR 9,000.",
    "The registration fee is PKR 9000.",
    "The registration fee is rupees 9,000 only.",
    "The registration fee is rs. 9,000.",
]:
    supported(answer, MARKED_CONTEXT, f"marked evidence supports {answer!r}")
for answer in [
    "The A Fee for this category is Rs. 21,000.",
    "The A Fee for this category is Rs 21000.",
    "The A Fee for this category is PKR 21,000.",
]:
    supported(answer, GROUPED_CONTEXT, f"comma-grouped evidence supports {answer!r}")
supported(
    "The examiniation fee is Rs. 4,800.",
    GROUPED_CONTEXT,
    "mixed marked+grouped context supports Rs. 4,800",
)
supported(
    "Live web (pu.edu.pk): https://www.pu.edu.pk/x — published 2026-08-01: admission fee Rs. 12,500.",
    "Live web (pu.edu.pk): https://www.pu.edu.pk/x — published 2026-08-01 — fee Rs. 12,500 for 2026-27",
    "official web context supports the quoted web figure",
)

# --------------------------------------- B. unsupported currency amount
print("\n[B] unsupported currency amounts")
refused("The registration fee is Rs. 12,345.", MARKED_CONTEXT, "absent amount refused")
refused("The registration fee is Rs. 99,000.", GROUPED_CONTEXT, "unrelated amount refused")
refused("The fee is Rs. 10,575.", GROUPED_CONTEXT, "hallucinated amount refused")

# --------------------------------------- C. different amount, similar digits
print("\n[C] similar digits do not support")
refused("The fee is Rs. 1,575.", GROUPED_CONTEXT, "1,575 vs 15,750 refused")
refused("The fee is Rs. 157,500.", GROUPED_CONTEXT, "157,500 vs 15,750 refused")
refused("The fee is Rs. 15,000.", GROUPED_CONTEXT, "15,000 vs 15,750 refused")
refused("The fee is Rs. 850.", OTHER_CONTEXT, "850 vs 8,500 refused")
refused("The fee is Rs. 5,004.", OTHER_CONTEXT, "5,004 vs 4,800/9,000 refused")

# --------------------------------------- D. comma / no-comma equivalence
print("\n[D] comma and no-comma formatting equivalence")
supported("The fee is Rs. 15,750.", GROUPED_CONTEXT, "context '15,750' supports 'Rs. 15750'")
supported("The fee is Rs. 15750.", GROUPED_CONTEXT, "context '15,750' supports 'Rs. 15750' (no commas in answer)")
supported("The fee is PKR 15750.", GROUPED_CONTEXT, "context '15,750' supports 'PKR 15750'")
supported("The fee is Rs. 9000.", MARKED_CONTEXT, "context 'Rs.9,000' supports 'Rs. 9000'")
supported("The fee is PKR 9,000.", MARKED_CONTEXT, "context 'Rs.9,000' supports 'PKR 9,000'")

# --------------------------------------- E. unrelated context gives no support
print("\n[E] digits in unrelated context are NOT currency evidence")
refused("The fee is Rs. 15,000.", UNRELATED_CONTEXT,
        "bare unmarked 15000 (id/page/marks) does not support Rs. 15,000")
refused("The form charge is Rs. 15,750.", UNRELATED_CONTEXT,
        "form number 15750 is not money evidence")
refused("The fee is Rs. 2026.", UNRELATED_CONTEXT,
        "session year 2026 is not money evidence")
refused("The amount is Rs. 9,000.", "page 9000 of the notice board lists applications",
        "page 9000 is not money evidence")
# question-side digits count as support (question is context): guard behavior
supported("The fee is Rs. 15,000.", "The PDF states the fee is Rs. 15,000 for 2026-27",
          "explicit Rs. figure in context supports")

# ------------------------- F. preserved: percentages and thresholds
print("\n[F] percentages / ages / academic thresholds unchanged")
supported("Minimum attendance is 75%.", "Students must attend 75% of classes",
          "supported percentage passes")
refused("Minimum attendance is 99%.", "Students must attend 75% of classes",
        "invented percentage refused")
supported("The limit is 24 years.", "Applicants age must not be more than 24 years",
          "supported age passes")
refused("The limit is 26 years.", "Applicants age must not be more than 24 years",
        "invented age refused")
supported("Probation begins below 1.70 cgpa.", "probation at 1.70 cgpa is required",
          "supported academic threshold passes")
refused("Probation begins below 1.60 cgpa.", "probation at 1.70 cgpa is required",
        "invented academic threshold refused")
# Number followed by unit: preserved (plain token semantics, by design)
supported("The dues are 15,750 rupees.", "A Fee 15,750",
          "number-then-unit claim keeps pre-existing token behavior")

# ---------------- G. Step 23H — narrow table-label currency evidence
print("\n[G] Step 23H: monetary table label support ('Sports & Mosque Fund 950')")
# A. Grounded BS CS answer with Rs. 950 must be accepted
print("\n    [A] grounded BS CS fee answer is accepted")
supported(GOOD_BS_CS_ANSWER, FEE_SCHEDULE_CONTEXT,
          "grounded BS CS fee answer (Rs. 950 included) accepted")
supported("The Sports & Mosque Fund is Rs. 950 in the Fall semester.",
          FEE_SCHEDULE_CONTEXT,
          "Rs. 950 supported by 'Sports & Mosque Fund 950' co-occurrence")
supported("The A Fee for BS CS (FCIT) is Rs. 55,000.",
          FEE_SCHEDULE_CONTEXT,
          "Rs. 55,000 (same table row as FCIT CS) still supported")
supported("Service Charges Rs. 8,500 and Examination Fee Rs. 4,800.",
          FEE_SCHEDULE_CONTEXT,
          "Rs. 8,500 and Rs. 4,800 still supported")
supported("Registration Fee Rs. 9,000.",
          FEE_SCHEDULE_CONTEXT,
          "marked Rs.9,000 still supported")
supported("The department fee is Rs. 29,950.",
          FEE_SCHEDULE_CONTEXT,
          "Rs. 29,950 (comma-grouped) still supported")

# B. Rs. 950 must still be refused without monetary table context
print("\n    [B] bare 950 near non-monetary labels does NOT establish support")
for ctx, label in [
    ("applicant id 950 of the batch", "applicant ID 950"),
    ("page 950 of the prospectus", "page 950"),
    ("total of 950 marks", "950 marks"),
    ("roll number 950 registered", "roll number 950"),
    ("registration number 950", "registration number 950"),
]:
    refused("The Sports & Mosque Fund is Rs. 950.", ctx,
            f"Rs. 950 refused; {label} is not money evidence")

# C. Neighboring amounts are not admitted just because 950 is supported
print("\n    [C] neighboring amounts remain unsupported")
refused("The fee is Rs. 9,500.", FEE_SCHEDULE_CONTEXT, "Rs. 9,500 refused")
refused("The fee is Rs. 90.", FEE_SCHEDULE_CONTEXT, "Rs. 90 refused")
refused("The fee is Rs. 9,050.", FEE_SCHEDULE_CONTEXT, "Rs. 9,050 refused")

# D. Unicode whitespace variants in the claim
print("\n    [D] Unicode whitespace variants in Rs./PKR claims")
for unit_space, label in [("\u202f", "U+202F narrow no-break"),
                          ("\u00a0", "U+00A0 non-breaking")]:
    supported(f"The Sports & Mosque Fund is Rs.{unit_space}950.",
              FEE_SCHEDULE_CONTEXT,
              f"Rs.{unit_space!r}950 ({label}) accepted")
supported("The Sports & Mosque Fund is PKR\u202f950.",
          FEE_SCHEDULE_CONTEXT,
          "PKR U+202F 950 accepted")

# E. Mocked RAG integration: full grounded BS CS answer must be ANSWERED
print("\n    [E] mocked RAG integration: grounded BS CS answer -> STATUS_ANSWERED")


class _FakeEmbedding:
    def embed_text(self, text: str):
        return [0.01] * 8


class _FakeChroma:
    def __init__(self):
        self.doc = FEE_SCHEDULE_CONTEXT
        self.meta = {
            "filename": "Dues-Fee-Schedule-for-the-Year-2026-27.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "category": "fees",
            "document_type": "fee schedule",
            "effective_session": "2026-27",
            "verification_status": "verified",
        }

    def count(self):
        return 1

    def query(self, query_embedding, top_k=None):
        return {"documents": [[self.doc]], "metadatas": [[self.meta]], "distances": [[0.18]]}

    def query_filtered(self, query_embedding, where, top_k=10):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class _FakeWebSearch:
    """Hermetic: records queries; never performs network calls."""

    def __init__(self, results=None, raise_exc=None):
        self.results = results or []
        self.raise_exc = raise_exc
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(query)
        if self.raise_exc is not None:
            raise self.raise_exc
        return WebSearchResult(results=list(self.results))


class _RecordingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, system=None, timeout=None):
        self.prompts.append(prompt)
        return self.answer


_rag = RAGService(embeddings=_FakeEmbedding(), chroma=_FakeChroma(),
                  llm=_RecordingLLM(GOOD_BS_CS_ANSWER),
                  web_search=_FakeWebSearch())
_result = _rag.answer("give fee structure of bs cs")
if _result.status == "answered":
    ok("grounded BS CS fee answer returns STATUS_ANSWERED (not no_relevant_context)")
else:
    bad(f"BS CS fee answer failed: status={_result.status}, error={_result.error}")
if "Rs. 950" in _result.answer:
    ok("answer still contains Rs. 950 verbatim")
else:
    bad("answer lost the Rs. 950 figure")
if _result.sources and all(s.source_type == "pdf" for s in _result.sources):
    ok("sources remain PDF-typed")
else:
    bad("sources wrong")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
