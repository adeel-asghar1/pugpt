"""Smoke test for the embedding service.

Verifies: single-text embedding, non-empty/numeric/768-dim vector, two
different sentences both embedding successfully, batch embedding, and
rejection of empty input. Does NOT touch ChromaDB.

Run from the backend/ directory:
    python scripts/test_embeddings.py
"""

import sys
from pathlib import Path

# Make 'app' importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedding_service import embedding_service

EXPECTED_DIMENSIONS = 768
SENTENCE_PU = (
    "The University of the Punjab is the largest public sector university "
    "in Lahore, Pakistan, known for its departments of computer science and law."
)
SENTENCE_OTHER = "Mangoes are a popular summer fruit across South Asia."


def check_vector(name: str, vector: list[float]) -> bool:
    ok = True
    if not vector:
        print(f"[FAIL] {name}: vector is empty")
        ok = False
    else:
        if not all(isinstance(v, float) for v in vector):
            print(f"[FAIL] {name}: not every value is numeric")
            ok = False
        if len(vector) != EXPECTED_DIMENSIONS:
            print(f"[FAIL] {name}: expected {EXPECTED_DIMENSIONS} dimensions, got {len(vector)}")
            ok = False
    if ok:
        print(f"[PASS] {name}: non-empty, {len(vector)}-dim, all numeric")
    return ok


def main() -> int:
    failures = 0

    # 1. Embed a University of the Punjab sentence (single string)
    print(f"[1] Embedding sample sentence: {SENTENCE_PU!r}")
    try:
        vector = embedding_service.embed_text(SENTENCE_PU)
        if not check_vector("sentence (PU)", vector):
            failures += 1
    except Exception as exc:
        print(f"[FAIL] sentence (PU): {exc}")
        failures += 1

    # 2. Embed a second, different sentence
    print(f"[2] Embedding a different sentence: {SENTENCE_OTHER!r}")
    try:
        vector2 = embedding_service.embed_text(SENTENCE_OTHER)
        if not check_vector("sentence (other)", vector2):
            failures += 1
    except Exception as exc:
        print(f"[FAIL] sentence (other): {exc}")
        failures += 1

    # 3. Batch embedding: both sentences in one call
    print("[3] Batch-embedding both sentences in a single request")
    try:
        batch = embedding_service.embed_texts([SENTENCE_PU, SENTENCE_OTHER])
        if len(batch) != 2:
            print(f"[FAIL] batch: expected 2 vectors, got {len(batch)}")
            failures += 1
        else:
            for i, vec in enumerate(batch):
                if not check_vector(f"batch[{i}]", vec):
                    failures += 1
    except Exception as exc:
        print(f"[FAIL] batch: {exc}")
        failures += 1

    # 4. Empty/whitespace input must be rejected with ValueError
    print("[4] Submitting empty input (must be rejected)")
    try:
        embedding_service.embed_text("   ")
        print("[FAIL] empty input was NOT rejected")
        failures += 1
    except ValueError as exc:
        print(f"[PASS] empty input rejected: {exc}")
    except Exception as exc:
        print(f"[FAIL] empty input raised unexpected error: {type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures:
        print(f"RESULT: FAILED ({failures} check(s) failed)")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
