"""Sources consistency regression — mocked generation, real retrieval read-only.

Rule 23 (Step 14A): the model must NOT emit its own Sources section; the UI
renders exactly one Sources block, built from structured retrieval metadata.

Verifies:
1. PUGPT_SYSTEM_PROMPT forbids a model-generated Sources section and does not
   carry the old "End every answer with ... 'Sources' section" instruction.
2. The backend passes the (Sources-free) model answer through byte-identically
   and attaches deduped structured metadata: multiple chunks from the same PDF
   page collapse into ONE source entry.
3. The frontend's structured Sources component is intact (single render layer,
   beneath the ReactMarkdown body).

Run from the backend/ directory:
    python scripts/test_sources_consistency.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from app.prompts import PUGPT_SYSTEM_PROMPT
from app.services.rag_service import RAGService

failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


print("=" * 78)
print("Sources consistency — mocked LLM, offline")
print("=" * 78)

# ------------------------------------------------- 1. prompt contract
print("\n[prompt contract]")
if 'do not generate a "sources" heading' in PUGPT_SYSTEM_PROMPT.lower():
    ok("prompt forbids model-generated Sources heading")
else:
    bad("prompt lost the new Sources-forbidding rule")

if "application displays citations automatically" in PUGPT_SYSTEM_PROMPT.lower():
    ok("prompt states the application renders citations")
else:
    bad("prompt lost the application-renders-citations clause")

if 'end every answer with a clearly separated "sources" section' not in PUGPT_SYSTEM_PROMPT.lower():
    ok("old model-generated Sources instruction is gone")
else:
    bad("OLD model-generated Sources instruction still present")

# ----------------------------- 2. dedup + structured metadata via mocked RAG
print("\n[mocked RAG — dedup same-page chunks; answer untouched; metadata intact]")
from app.services.chroma_service import ChromaService
from app.services.embedding_service import embedding_service

shared_text = (
    "consistency-test-chunk The PU consistency-test document covers the exact same "
    "wording on every chunk so that vectors coincide; fee categories A B and admission dues "
    "are listed with programme labels."
)
GOOD_ANSWER = (
    "**Consistency test answer — no model-written Sources section.**\n\n"
    "- Fee categories **A** and **B** differ as printed in the schedule.\n"
    "- The programme name must accompany every quoted figure."
)

temp_root = Path(tempfile.mkdtemp(prefix="pugpt_sources_test_"))
try:
    chroma = ChromaService(persist_path=temp_root / "chroma", collection_name="pugpt_sources_test")
    docs, ids, metas = [], [], []
    # three chunks on page 1, one chunk on page 2 — same filename
    for i in range(3):
        docs.append(shared_text + f" chunk-{i}")
        ids.append(f"ctest:{i}")
        metas.append({"filename": "ConsistencyTest.pdf", "page_number": 1, "chunk_index": i,
                      "source_path": "ConsistencyTest.pdf", "doc_id": "ctest"})
    docs.append(shared_text + " chunk-3")
    ids.append("ctest:3")
    metas.append({"filename": "ConsistencyTest.pdf", "page_number": 2, "chunk_index": 3,
                  "source_path": "ConsistencyTest.pdf", "doc_id": "ctest"})

    embeddings = [embedding_service.embed_text(d) for d in docs]
    chroma.add_documents(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    class SourcesFreeLLM:
        def generate(self, prompt, system=None, timeout=None):
            return GOOD_ANSWER

    rag = RAGService(chroma=chroma, llm=SourcesFreeLLM())
    result = rag.answer(shared_text)

    if result.answer == GOOD_ANSWER:
        ok("answer passes through byte-identically (backend injects nothing)")
    else:
        bad("backend altered the answer in transit")

    if "\nSources" not in result.answer and not result.answer.rstrip().endswith("Sources:"):
        ok("answer contains NO model-written Sources section")
    else:
        bad("answer contains a Sources section")

    pages = sorted({s.page_number for s in result.sources})
    if pages == [1, 2]:
        ok(f"same-page chunks deduped -> sources list shows pages {pages} once each")
    else:
        bad(f"dedup wrong -> source pages {pages}")

    if all(s.filename == "ConsistencyTest.pdf" for s in result.sources):
        ok("all structured sources carry filename metadata")
    else:
        bad(f"unexpected filenames: {[s.filename for s in result.sources]}")

    if all(s.distance is not None for s in result.sources):
        ok("structured sources retain distance metadata")
    else:
        bad("structured sources lost distance metadata")

    try:
        chroma.delete_collection("pugpt_sources_test")
        ok("temp ChromaDB collection cleaned up")
    except Exception as exc:  # noqa: BLE001
        bad(f"temp collection cleanup failed: {exc}")
finally:
    shutil.rmtree(temp_root, ignore_errors=True)

# --------------------------- 3. frontend structured Sources component intact
print("\n[frontend — single structured Sources render layer]")
message_jsx = (PROJECT_ROOT / "frontend" / "src" / "components" / "Message.jsx").read_text(encoding="utf-8")
sourcelist_jsx = (PROJECT_ROOT / "frontend" / "src" / "components" / "SourceList.jsx").read_text(encoding="utf-8")

if "ReactMarkdown" in message_jsx:
    ok("Message.jsx still renders the model answer via ReactMarkdown")
else:
    bad("Message.jsx lost the ReactMarkdown body")
if "SourceList" in message_jsx:
    ok("Message.jsx still renders the structured SourceList layer")
else:
    bad("Message.jsx lost the SourceList layer")
if "sources-label" in sourcelist_jsx and "Sources" in sourcelist_jsx:
    ok("SourceList.jsx intact — the only Sources block rendered by the app")
else:
    bad("SourceList.jsx missing or altered")

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
