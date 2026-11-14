"""Step 21 regression tests — targeted answer-reliability improvements.

Covers the four Step 20 end-to-end findings:
  1. Merit retrieval gap: merit document never entered the candidate pool for
     meta-questions; fixed by the intent-filtered pool rescue in RAGService.
     (Integration check against the REAL knowledge base, read-only.)
  2. Fee overgeneralization: prompt rule 18 — program-specific figures stay
     program-specific; no universal fee unless the document states it.
  3. GPA vs CGPA: prompt rule 19 — distinction carried exactly as the PU
     documents define it; nothing invented.
  4. Unsupported session distinctions: prompt rule 20 — no session scope may
     be asserted unless a retrieved source explicitly establishes it.

Deterministic checks (prompt text, verifier) need no services besides the
embedder. RAG integration tests use the REAL Chroma knowledge base (read-
only; no writes anywhere).

Run from the backend/ directory:
    python scripts/test_step21_reliability.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
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
print("Step 21 — reliability regression tests")
print("=" * 78)

# ---------------------------------------------------------------- T2/T3/T4
print("\n[prompt-content checks]")
if "quota-specific" in PUGPT_SYSTEM_PROMPT and "ask which program" in PUGPT_SYSTEM_PROMPT:
    ok("T2 prompt rule 18: program-specific fee handling present")
else:
    bad("T2 prompt rule 18 missing — fee rule not found in PUGPT_SYSTEM_PROMPT")

if "semester's credit hours" in PUGPT_SYSTEM_PROMPT and "individual courses" in PUGPT_SYSTEM_PROMPT:
    ok("T3 prompt rule 19: GPA/CGPA distinction present (PU-documented definitions)")
else:
    bad("T3 prompt rule 19 missing — GPA/CGPA distinction not found in prompt")

if "explicitly states that limitation" in PUGPT_SYSTEM_PROMPT and "session limitation" in PUGPT_SYSTEM_PROMPT:
    ok("T4 prompt rule 20: session-scope honesty rule present")
else:
    bad("T4 prompt rule 20 missing — session-distinction rule not found in prompt")

# ---------------------------------------------------------------- T5
print("\n[verifier sanity — safeguards not weakened]")
# 13,300 IS in this synthetic context, so the verifier passes it — but a
# genuinely out-of-context figure must still be rejected:
hungry2 = find_unsupported_claims(
    "The fee is Rs. 9,999,999 rupees for everyone.",
    ["BS Agriculture: 7,590  FCIT, Physics: 6,000  Commerce: 14,250"],
    "fee question",
)
if hungry2:
    ok(f"T5 verifier still rejects out-of-context comma-formatted figures -> {hungry2}")
else:
    bad("T5 verifier FAILED to reject out-of-context figure")

hungry3 = find_unsupported_claims(
    "The maximum age is 999 years.",
    ["admission: age must not exceed 24 years"],
    "age question",
)
if hungry3:
    ok(f"T5b verifier still rejects out-of-context year/count figures -> {hungry3}")
else:
    bad("T5b verifier FAILED to reject out-of-context figure")

# ---------------------------------------------------------------- T6
print("\n[threshold unchanged]")
if settings.rag_distance_threshold == 0.5:
    ok("T6 RAG_DISTANCE_THRESHOLD is still 0.5")
else:
    bad(f"T6 threshold changed: {settings.rag_distance_threshold}")

# ------------------------------------------------- T1 integration (read-only)
print("\n[integration — REAL Chroma 'pugpt_knowledge', read-only]")
try:
    from app.services.embedding_service import embedding_service

    rag = RAGService()

    def kept_files(question: str) -> list[str]:
        vec = embedding_service.embed_text(question)
        kept = rag._retrieve(vec, question)
        return [str(m.get("filename", "")) for _d, m, _dist, _adj in kept]

    q_merit = "What does the 2025 merit document contain?"
    files = kept_files(q_merit)
    if any("Merit-of-Last-Year-2025" in f for f in files):
        ok(f"T1a merit document IS retrieved for 'What does the 2025 merit document contain?' -> {files}")
    else:
        bad(f"T1a merit document NOT retrieved: {files}")

    q_merit2 = "What are the closing merits for BS programs for 2025?"
    files2 = kept_files(q_merit2)
    if any("Merit-of-Last-Year-2025" in f for f in files2):
        ok(f"T1b merit document IS retrieved for closing-merit question -> {files2}")
    else:
        bad(f"T1b merit document NOT retrieved: {files2}")

    q_oos = "What is the capital of France?"
    files3 = kept_files(q_oos)
    if not files3:
        ok("T1c out-of-scope question still returns NO sources (no forced injection)")
    else:
        bad(f"T1c out-of-scope question retrieved sources: {files3}")

    q_weather = "What is the weather forecast for Lahore tomorrow?"
    from app.services.rag_service import _detect_intent
    if _detect_intent(q_weather.lower()) is None:
        ok("T1d non-intent small-talk question triggers NO intent pool rescue (merit cannot be forced in)")
    else:
        bad(f"T1d weather question unexpectedly detected an intent: {_detect_intent(q_weather.lower())}")
except Exception as exc:  # noqa: BLE001
    bad(f"integration tests could not run: {exc}")

# ------------------------------------------------------------- T4 end-to-end
print("\n[integration — full RAG path, mocked LLM records what it sees]")
try:
    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, system=None, timeout=None):
            self.calls.append(prompt)
            return "Mock answer: see context."

    rec = RecordingLLM()
    rag = RAGService(llm=rec)
    result = rag.answer("What does the 2025 merit document contain?")
    merged = rec.calls[0] if rec.calls else ""
    if "Merit-of-Last-Year-2025" in merged:
        ok("R1 merged prompt for merit question INCLUDES the merit document")
    else:
        bad("R1 merged prompt does NOT include the merit document")
    print(f"      (status={result.status}; sources: {[s.filename for s in result.sources]})")
except Exception as exc:  # noqa: BLE001
    bad(f"R1 end-to-end recording failed: {exc}")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
