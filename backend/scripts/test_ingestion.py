"""End-to-end smoke test for the ingestion pipeline.

Uses a DEDICATED test collection ('pugpt_ingest_test') — the real
'pugpt_knowledge' collection is never touched. Creates temporary
programmatically generated PDFs, runs the full pipeline against the real
Ollama embedding model and a real persistent ChromaDB, and cleans up
everything at the end.

Run from the backend/ directory:
    python scripts/test_ingestion.py

Exit codes: 0 = all checks passed, 1 = failures, 3 = Ollama unavailable (skipped).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from app.config import settings
from app.services.chroma_service import ChromaService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import embedding_service
from app.services.ingestion_service import IngestionService, ingestion_service

TEST_COLLECTION = "pugpt_ingest_test"
SMALL_PROCESSOR = DocumentProcessor(chunk_size=150, chunk_overlap=30)

PDF_A_V1 = (
    "ALPHAUNI Marker Document. Alpha University of Zantoria was founded in "
    "1882 by the Zantorian colonial education council and remains the oldest "
    "and largest public sector university in the Zantorian province. It is "
    "located in the city of Rawalpindi near the northern foothills. Alpha "
    "University offers faculties of engineering, law, medicine, arts, "
    "science and business administration. Alpha University admitted its "
    "first students in 1883 and today enrolls more than forty thousand "
    "students across its many departments and affiliated colleges."
)
PDF_A_V2 = "BETAUNI Marker Document. Beta Institute Zantoria was founded in 1995. It has three faculties."
PDF_A_V3 = (
    "DELTAUNI Marker Document. Delta University of Zantoria traces its "
    "origins to 1882 when the Zantorian council chartered an advanced "
    "college for aspirants in Rangpur. Over decades Alpha traditions merged "
    "into Delta University. Today Delta University hosts twelve faculties "
    "and a central library with more than half a million volumes."
)
PDF_B = (
    "GAMMAMED Marker Document. Gamma Medical School Zantoria was established "
    "in 1974 and is affiliated with the provincial health service. It trains "
    "medical and dental students in two teaching hospitals and a network of "
    "district clinics across the province."
)


def write_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=11)
    doc.save(path)
    doc.close()


def make_moving_embedding_probe() -> bool:
    try:
        embedding_service.embed_text("probe")
        return True
    except Exception as exc:
        print(f"\nSKIPPED: Ollama embedding service unavailable: {exc}")
        print("Start Ollama (ollama serve) and re-run this test.")
        return False


def main() -> int:
    if not make_moving_embedding_probe():
        return 3

    failures = 0

    def ok(msg: str) -> None:
        print(f"[PASS] {msg}")

    def bad(msg: str) -> None:
        nonlocal failures
        print(f"[FAIL] {msg}")
        failures += 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="pugpt_ingest_test_"))
    pdf_a = tmp_dir / "A.pdf"
    pdf_b = tmp_dir / "B.pdf"
    test_chroma = ChromaService(settings.chroma_persist_path, TEST_COLLECTION)
    # Start from a clean slate in case of a previous aborted run.
    try:
        test_chroma.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    test_chroma = ChromaService(settings.chroma_persist_path, TEST_COLLECTION)
    service = IngestionService(processor=SMALL_PROCESSOR, chroma=test_chroma)

    try:
        # ---- 1-4. Basic ingestion of PDF A ----
        write_pdf(pdf_a, PDF_A_V1)
        summary = service.ingest_pdf(pdf_a)
        if summary["status"] == "success":
            ok(f"PDF A ingested: {summary['pages_extracted']} page(s), "
               f"{summary['chunks_created']} chunk(s)")
        else:
            bad(f"PDF A ingestion failed: {summary['error']}")
            summary = {"chunks_created": 0, "doc_id": "n/a", "pages_extracted": 0}

        stored = test_chroma.count()
        if stored == summary["chunks_created"] and stored > 0:
            ok(f"ChromaDB holds exactly the expected chunk count ({stored})")
        else:
            bad(f"expected {summary['chunks_created']} chunks, found {stored}")

        doc_id = summary["doc_id"]
        records = test_chroma.get_where({"doc_id": doc_id})
        metas = records.get("metadatas") or []
        if metas and all(
            m.get("filename") == "A.pdf"
            and m.get("page_number", 0) >= 1
            and "chunk_index" in m
            and m.get("source_path")
            and m.get("doc_id") == doc_id
            for m in metas
        ):
            ok("metadata preserved (filename, page_number, chunk_index, source_path, doc_id)")
        else:
            bad(f"metadata incomplete: {metas[:2]}")

        # ---- 4. Semantic query retrieval ----
        query = embedding_service.embed_text("When was Alpha University of Zantoria founded?")
        result = test_chroma.query(query, top_k=1)
        top_docs = (result.get("documents") or [[]])[0]
        top_metas = (result.get("metadatas") or [[]])[0]
        if top_docs and "1882" in top_docs[0] and top_metas and top_metas[0].get("filename") == "A.pdf":
            ok("semantic query retrieved the stored document ('1882', A.pdf)")
        else:
            bad(f"semantic query missed the target document: {top_docs[:1]}")

        # ---- 5. Re-ingesting the same file: no duplicates ----
        before = test_chroma.count()
        summary = service.ingest_pdf(pdf_a)
        after = test_chroma.count()
        if summary["status"] == "success" and after == before:
            ok(f"re-ingestion created no duplicates (count stayed {after})")
        else:
            bad(f"re-ingestion changed the count: {before} -> {after}")

        # ---- 6. Update + re-ingest: old chunks replaced ----
        write_pdf(pdf_a, PDF_A_V2)  # shorter, completely different text
        summary = service.ingest_pdf(pdf_a)
        docs = test_chroma.get_where({"doc_id": doc_id}).get("documents") or []
        if summary["status"] == "success" and not any("ALPHAUNI" in d for d in docs):
            ok("updated PDF re-ingested: old ALPHAUNI chunks replaced")
        else:
            bad(f"old chunks still present after re-ingest: {len(docs)} record(s)")

        # ---- 7. Shorter PDF: obsolete chunks removed ----
        metas = test_chroma.get_where({"doc_id": doc_id}).get("metadatas") or []
        if len(metas) == summary["chunks_created"] == 1:
            ok("obsolete chunks removed — single-page PDF now has exactly 1 chunk")
        else:
            bad(f"expected 1 chunk after shrink, found {len(metas)} "
                f"(summary said {summary['chunks_created']})")

        # ---- 8. Ingesting a second PDF keeps the first ----
        write_pdf(pdf_b, PDF_B)
        summary_b = service.ingest_pdf(pdf_b)
        a_docs = test_chroma.get_where({"doc_id": doc_id}).get("documents") or []
        b_docs = test_chroma.get_where({"doc_id": service.compute_doc_id(pdf_b)}).get("documents") or []
        if summary_b["status"] == "success" and a_docs and b_docs:
            ok(f"second PDF ingested ({len(b_docs)} chunk(s)); first PDF kept ({len(a_docs)} chunk(s))")
        else:
            bad("second PDF replaced/deleted the first PDF's records")

        # ---- 9. Embedding failure preserves previously stored data ----
        write_pdf(pdf_a, PDF_A_V3)  # known-good current A = one 'BETAUNI' chunk

        class BrokenEmbedder:
            def embed_texts(self, texts):
                raise RuntimeError("Ollama unreachable (simulated for test)")

            def embed_text(self, text):
                raise RuntimeError("Ollama unreachable (simulated for test)")

        broken = IngestionService(processor=SMALL_PROCESSOR, embeddings=BrokenEmbedder(),
                                  chroma=test_chroma)
        failed = broken.ingest_pdf(pdf_a)
        docs_after = test_chroma.get_where({"doc_id": doc_id}).get("documents") or []
        if (failed["status"] == "failed"
                and any("BETAUNI" in d for d in docs_after)
                and not any("DELTAUNI" in d for d in docs_after)):
            ok("embedding failure left previous data intact (BETAUNI kept, nothing new stored)")
        else:
            bad(f"previous data NOT preserved on failure: status={failed['status']}, "
                f"docs={[d[:40] for d in docs_after]}")

        # Restore a sane state: ingest DELTAUNI properly.
        restored = service.ingest_pdf(pdf_a)
        docs_final = test_chroma.get_where({"doc_id": doc_id}).get("documents") or []
        if restored["status"] == "success" and any("DELTAUNI" in d for d in docs_final):
            ok("document re-ingested successfully after the simulated failure")
        else:
            bad("re-ingestion after simulated failure did not succeed")

    finally:
        # ---- 10. Cleanup: test collection + temp files ----
        try:
            test_chroma.delete_collection(TEST_COLLECTION)
            print(f"[cleanup] deleted test collection '{TEST_COLLECTION}'")
        except Exception as exc:
            print(f"[cleanup] warning: could not delete test collection: {exc}")
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[cleanup] removed temporary directory {tmp_dir}")

    print()
    if failures:
        print(f"RESULT: FAILED ({failures} check(s) failed)")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
