"""Ingest PDFs into ChromaDB.

Usage (from the backend/ directory):
    python scripts/ingest_documents.py path/to/file.pdf
    python scripts/ingest_documents.py path/to/pdf/directory   # recursive

Non-PDF files are skipped. Each document is processed independently;
a failure in one does not stop the rest. Exit code 0 = all ingested,
1 = at least one failed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ingestion_service import ingestion_service


def collect_pdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == ".pdf" else []
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: path does not exist: {target}")
        return 2

    pdfs = collect_pdfs(target)
    if not pdfs:
        print(f"No PDF files found at: {target}")
        return 2

    print(f"Found {len(pdfs)} PDF(s). Ingesting into the knowledge base...\n")

    succeeded = 0
    failed = 0
    for pdf in pdfs:
        summary = ingestion_service.ingest_pdf(pdf)
        status = summary["status"].upper()
        if summary["status"] == "success":
            succeeded += 1
            action = "replaced" if summary["replaced_existing"] else "added"
            print(f"[{status}] {summary['filename']}: "
                  f"{summary['pages_extracted']} page(s) -> "
                  f"{summary['chunks_stored']} chunk(s) {action}")
        else:
            failed += 1
            print(f"[{status}] {summary['filename']}: {summary['error']}")

    print(f"\nDone. Succeeded: {succeeded}, failed: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
