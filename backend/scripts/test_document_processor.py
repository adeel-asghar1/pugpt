"""Smoke test for the document processor and chunker.

Creates a temporary multi-page PDF (NOT a real university document),
extracts pages, chunks text, checks overlap, verifies error handling,
and cleans up all temporary files.

Run from the backend/ directory:
    python scripts/test_document_processor.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from app.services.document_processor import (
    DocumentProcessor,
    DocumentProcessorError,
    InvalidPDFError,
    PDFNotFoundError,
)

LONG_SENTENCE = (
    "The University of the Punjab was established in 1882 and remains the "
    "largest and oldest public sector university in Lahore Pakistan with many "
    "faculties departments and thousands of students enrolled each year. "
) * 20  # ~3800 chars: comfortably exceeds the 2000-char long-text assertions
          # and the 200-char chunk size used in this test

SHORT_TEXT = "PUGPT smoke-test short page."


def longest_prefix_in_suffix(chunk_a: str, chunk_b: str) -> int:
    """Longest k where chunk_b starts with chunk_a's last k characters."""
    upper = min(len(chunk_a), len(chunk_b))
    for k in range(upper, 0, -1):
        if chunk_b.startswith(chunk_a[-k:]):
            return k
    return 0


def main() -> int:
    failures = 0

    def ok(message: str) -> None:
        print(f"[PASS] {message}")

    def bad(message: str) -> None:
        nonlocal failures
        print(f"[FAIL] {message}")
        failures += 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="pugpt_docproc_test_"))
    pdf_path = tmp_dir / "test_document.pdf"
    try:
        # ---- Build the temporary multi-page PDF ----
        doc = pymupdf.open()
        page1 = doc.new_page()
        # A textbox wraps the text across lines so the full paragraph is
        # actually stored in the page (single-line insert_text truncates).
        fitted = page1.insert_textbox(
            pymupdf.Rect(50, 50, 545, 800), LONG_SENTENCE, fontsize=11
        )
        if fitted < 0:
            print(f"[setup] warning: textbox returned {fitted} (text may be clipped)")
        doc.new_page()  # page 2 intentionally left empty (no text)
        page3 = doc.new_page()
        page3.insert_textbox(pymupdf.Rect(50, 50, 545, 800), SHORT_TEXT, fontsize=11)
        doc.set_metadata({"title": "PUGPT smoke test"})
        doc.save(pdf_path)
        doc.close()
        print(f"[setup] temporary PDF created at {pdf_path}")

        processor = DocumentProcessor(chunk_size=200, chunk_overlap=60)

        # ---- 1-4. Extraction ----
        try:
            pages = processor.extract_pages(pdf_path)
        except DocumentProcessorError as exc:
            bad(f"extraction raised unexpectedly: {exc}")
            pages = []

        if len(pages) == 2:
            ok("empty page ignored — extracted 2 of 3 pages")
        else:
            bad(f"expected 2 pages (empty one skipped), got {len(pages)}")

        filenames = {p.filename for p in pages}
        if filenames == {"test_document.pdf"}:
            ok("document filename preserved in every page")
        else:
            bad(f"filename not preserved: {filenames}")

        page_numbers = [p.page_number for p in pages]
        if page_numbers == [1, 3]:
            ok("page numbers preserved correctly (1-based, empty page skipped)")
        else:
            bad(f"page numbers wrong: {page_numbers}")

        if any(len(p.text) > 2000 for p in pages):
            ok("long text extracted intact")
        else:
            bad("long page text missing or truncated")

        # ---- 5-7. Chunking of long text ----
        chunks = processor.chunk_pages(pages)
        long_page_chunks = [c for c in chunks if c.page_number == 1]
        if len(long_page_chunks) > 1:
            ok(f"long text divided into {len(long_page_chunks)} chunks")
        else:
            bad(f"long text NOT split (got {len(long_page_chunks)} chunk(s))")

        if all(c.text.strip() for c in chunks):
            ok("no empty chunks produced")
        else:
            bad("at least one empty chunk was produced")

        max_len = max((len(c.text) for c in chunks), default=0)
        if max_len <= 200:
            ok(f"all chunks respect chunk_size (max length = {max_len})")
        else:
            bad(f"chunk exceeds chunk_size: {max_len} > 200")

        # Overlap: each chunk after the first should start before the
        # previous one ends (longest shared suffix/prefix >= 1 char).
        overlaps = [
            longest_prefix_in_suffix(a.text, b.text)
            for a, b in zip(long_page_chunks, long_page_chunks[1:])
        ]
        if overlaps and min(overlaps) >= 1:
            ok(f"chunk overlap works (min shared chars across pairs: {min(overlaps)})")
        else:
            bad(f"no overlap between consecutive chunks ({overlaps})")

        # Metadata preserved on every chunk
        if all(c.filename == "test_document.pdf" and c.page_number in (1, 3)
               for c in chunks):
            ok("chunk metadata (filename, page_number) preserved")
        else:
            bad("chunk metadata missing or wrong")

        indices = [c.chunk_index for c in chunks]
        if indices == list(range(len(chunks))):
            ok("chunk indices sequential from 0")
        else:
            bad(f"chunk indices not sequential: {indices}")

        # ---- 8. Short text remains intact ----
        page3 = next(p for p in pages if p.page_number == 3)
        short_chunks = processor.chunk_pages([page3])
        if len(short_chunks) == 1 and short_chunks[0].text == SHORT_TEXT:
            ok("short text remains intact (single chunk, unmodified)")
        else:
            bad(f"short text altered: {[c.text for c in short_chunks]}")

        # ---- 9. Invalid / missing input handled gracefully ----
        not_a_pdf = tmp_dir / "not_a_pdf.pdf"
        not_a_pdf.write_text("this is plainly not a PDF", encoding="utf-8")
        try:
            processor.extract_pages(not_a_pdf)
            bad("invalid PDF was NOT rejected")
        except InvalidPDFError as exc:
            ok(f"invalid PDF rejected with InvalidPDFError: {exc}")
        except DocumentProcessorError as exc:
            ok(f"invalid PDF rejected with DocumentProcessorError: {exc}")

        missing = tmp_dir / "does_not_exist.pdf"
        try:
            processor.extract_pages(missing)
            bad("missing PDF was NOT rejected")
        except PDFNotFoundError as exc:
            ok(f"missing PDF rejected with PDFNotFoundError: {exc}")

        empty_pdf = tmp_dir / "empty_text.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(empty_pdf)
        doc.close()
        try:
            scanned_pages = processor.extract_pages(empty_pdf)
            if scanned_pages == []:
                ok("PDF with no text layer handled gracefully (0 pages, warning logged)")
            else:
                bad(f"expected no extractable pages, got {scanned_pages}")
        except DocumentProcessorError as exc:
            bad(f"text-free PDF raised unexpectedly: {exc}")

    finally:
        # ---- 10. Clean up all temporary files ----
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
