"""Temporary smoke test for the ChromaDB service.

Verifies: persistent client init, collection create/retrieve, insert with a
manually supplied test embedding, query, persistence across client reopen,
and cleanup. Uses a THROWAWAY collection ('pugpt_test_collection') that is
deleted at the end — the real 'pugpt_knowledge' collection is never touched.

Run from the backend/ directory:
    python scripts/test_chromadb.py
"""

import sys
from pathlib import Path

# Make 'app' importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.chroma_service import ChromaService

TEST_COLLECTION = "pugpt_test_collection"
SAMPLE_DOC = "PUGPT ChromaDB smoke test document."
# Manually supplied fake embedding (8 dimensions). Deliberately NOT real
# university data, and NOT from the real knowledge collection.
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main() -> None:
    path = settings.chroma_persist_path

    # Best effort: remove leftovers from a previous failed run.
    try:
        ChromaService(path, TEST_COLLECTION).delete_collection(TEST_COLLECTION)
        print("[0] Test collection cleared (any leftover from a failed run removed)")
    except Exception:
        pass  # Did not exist — nothing to clean up.

    # 1-2. Persistent client + collection
    service = ChromaService(path, TEST_COLLECTION)
    assert service.is_ready, f"ChromaDB init failed: {service.last_error}"
    print(f"[1] Persistent client initialized at: {path}")
    print(f"[2] Collection ready: {TEST_COLLECTION}")

    # 3. Insert with a manually supplied embedding
    service.add_documents(
        ids=["smoke-test-doc-1"],
        documents=[SAMPLE_DOC],
        embeddings=[SAMPLE_EMBEDDING],
        metadatas=[{"source": "smoke-test"}],
    )
    print(f"[3] Inserted sample document (count = {service.count()})")

    # 4. Query and confirm the expected document comes back
    results = service.query(SAMPLE_EMBEDDING, top_k=1)
    retrieved = results["documents"][0]
    assert retrieved and retrieved[0] == SAMPLE_DOC, f"Unexpected result: {retrieved}"
    print(f"[4] Query retrieved: {retrieved[0]!r} (distance = {results['distances'][0][0]:.4f})")

    # 5. Reopen the client and confirm the data is still there
    reopened = ChromaService(path, TEST_COLLECTION)
    assert reopened.is_ready, f"Reopen failed: {reopened.last_error}"
    persisted = reopened.query(SAMPLE_EMBEDDING, top_k=1)["documents"][0]
    assert persisted and persisted[0] == SAMPLE_DOC, "Data did NOT persist"
    print("[5] Data persisted after closing and reopening the client")

    # 6. Clean up the throwaway collection (via the service's client/settings)
    service.delete_collection(TEST_COLLECTION)
    print(f"[6] Deleted test collection '{TEST_COLLECTION}'")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
