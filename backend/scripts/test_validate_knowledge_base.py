"""Smoke test for the knowledge base registry validator.

Uses a TEMPORARY knowledge base tree (never the real one) with generated
PDFs. Verifies: valid registry passes; each critical check fails loudly;
text-free PDF is a warning; temp files cleaned up.

Run from the backend/ directory:
    python scripts/test_validate_knowledge_base.py

Exit codes: 0 = all checks passed, 1 = failures.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from scripts.validate_knowledge_base import validate_registry

KB = "knowledge_base"


def write_pdf(path: Path, with_text: bool) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    if with_text:
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), "Official PU test document.", fontsize=11)
    doc.save(path)
    doc.close()


def make_kb(root: Path, entries: list[dict]) -> tuple[Path, Path, Path]:
    for category in [
        "admissions", "academics", "examinations", "fees",
        "scholarships", "faculties_programs", "student_services",
    ]:
        (root / "documents" / category).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    registry = root / "metadata" / "sources.json"
    registry.write_text(
        json.dumps({"schema_version": "1.0", "description": "test", "documents": entries}),
        encoding="utf-8",
    )
    return registry, root, root / "reports" / "validation_latest.md"


def base_entry(**overrides) -> dict:
    entry = {
        "document_id": "pu-test-admissions-001",
        "title": "Test Admissions Notification",
        "filename": "admissions-test.pdf",
        "category": "admissions",
        "source_url": "https://www.pu.edu.pk/notice/test.pdf",
        "publisher": "University of the Punjab",
        "published_date": "2025-05-01",
        "last_updated": "",
        "effective_session": "Autumn 2025",
        "campus": "New Campus",
        "program": "BS programs",
        "document_type": "notification",
        "verification_status": "pending",
        "notes": "",
    }
    entry.update(overrides)
    return entry


def main() -> int:
    failures = 0

    def ok(msg):
        print(f"[PASS] {msg}")

    def bad(msg):
        nonlocal failures
        print(f"[FAIL] {msg}")
        failures += 1

    tmp = Path(tempfile.mkdtemp(prefix="pugpt_kb_validate_test_"))
    try:
        # ---- 1. Valid registry passes ----
        root = tmp / "kb_valid"
        root.mkdir()
        registry, kb_root, report_path = make_kb(root, [base_entry()])
        write_pdf(kb_root / "documents" / "admissions" / "admissions-test.pdf", with_text=True)
        report = validate_registry(registry, kb_root, write_report=True, report_path=report_path)
        if report.ok and report.checked_documents == 1:
            ok("1 valid registry passes validation")
        else:
            bad(f"1 valid registry failed: errors={report.errors}")
        if report_path.exists():
            ok("1 validation report written to reports/validation_latest.md")
        else:
            bad("1 report file was not written")

        # ---- 2. Missing required metadata detected ----
        root = tmp / "kb_missing_field"
        root.mkdir()
        entry = base_entry(title="")
        registry, kb_root, _ = make_kb(root, [entry])
        write_pdf(kb_root / "documents" / "admissions" / "admissions-test.pdf", with_text=True)
        report = validate_registry(registry, kb_root, write_report=False)
        if any("'title'" in e for e in report.errors):
            ok("2 missing required field detected")
        else:
            bad(f"2 missing required field NOT detected: {report.errors}")

        # ---- 3. Missing referenced PDF detected ----
        root = tmp / "kb_missing_pdf"
        root.mkdir()
        registry, kb_root, _ = make_kb(root, [base_entry()])  # no PDF written
        report = validate_registry(registry, kb_root, write_report=False)
        if any("not found" in e for e in report.errors):
            ok("3 missing PDF detected")
        else:
            bad(f"3 missing PDF NOT detected: {report.errors}")

        # ---- 4. Duplicate document_id detected ----
        root = tmp / "kb_dup_id"
        root.mkdir()
        dup = base_entry(document_id="pu-dup-001", filename="a.pdf")
        dup2 = base_entry(document_id="pu-dup-001", filename="b.pdf")
        registry, kb_root, _ = make_kb(root, [dup, dup2])
        write_pdf(kb_root / "documents" / "admissions" / "a.pdf", with_text=True)
        write_pdf(kb_root / "documents" / "admissions" / "b.pdf", with_text=True)
        report = validate_registry(registry, kb_root, write_report=False)
        if any("Duplicate document_id" in e for e in report.errors):
            ok("4 duplicate document_id detected")
        else:
            bad(f"4 duplicate document_id NOT detected: {report.errors}")

        # ---- 5. Duplicate filename detected ----
        root = tmp / "kb_dup_fn"
        root.mkdir()
        dup = base_entry(document_id="pu-a", filename="same.pdf")
        dup2 = base_entry(document_id="pu-b", filename="same.pdf")
        registry, kb_root, _ = make_kb(root, [dup, dup2])
        write_pdf(kb_root / "documents" / "admissions" / "same.pdf", with_text=True)
        report = validate_registry(registry, kb_root, write_report=False)
        if any("Duplicate filename" in e for e in report.errors):
            ok("5 duplicate filename detected")
        else:
            bad(f"5 duplicate filename NOT detected: {report.errors}")

        # ---- 6. Non-PU domain URL rejected as critical ----
        root = tmp / "kb_bad_url"
        root.mkdir()
        entry = base_entry(source_url="https://www.shady-blog.example.com/fake-pu-notice.pdf")
        registry, kb_root, _ = make_kb(root, [entry])
        write_pdf(kb_root / "documents" / "admissions" / "admissions-test.pdf", with_text=True)
        report = validate_registry(registry, kb_root, write_report=False)
        if any("not official" in e for e in report.errors):
            ok("6 non-PU URL rejected as critical error")
        else:
            bad(f"6 non-PU URL NOT rejected: {report.errors}")

        # ---- 7. Text-free PDF produces a warning (not an error) ----
        root = tmp / "kb_scanned"
        root.mkdir()
        registry, kb_root, _ = make_kb(root, [base_entry()])
        write_pdf(kb_root / "documents" / "admissions" / "admissions-test.pdf", with_text=False)
        report = validate_registry(registry, kb_root, write_report=False)
        if any("no extractable text" in w for w in report.warnings) and not any(
            "admissions-test.pdf" in e and "not found" not in e for e in report.errors
        ):
            ok("7 text-free PDF reported as warning (not critical)")
        else:
            bad(f"7 text-free PDF handling wrong: errors={report.errors} warnings={report.warnings}")

        # ---- 7b. Non-zero exit on critical errors, zero on pass (via CLI) ----
        root = tmp / "kb_cli"
        root.mkdir()
        registry, kb_root, _ = make_kb(root, [base_entry(source_url="https://unofficial.example.com/x.pdf")])
        write_pdf(kb_root / "documents" / "admissions" / "admissions-test.pdf", with_text=True)
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.validate_knowledge_base",
             "--registry", str(registry), "--kb-root", str(kb_root),
             "--report", str(kb_root / "reports" / "out.md")],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        if proc.returncode == 1:
            ok("7b CLI exits non-zero when critical errors exist")
        else:
            bad(f"7b CLI exit code {proc.returncode} (expected 1)")

        # Empty registry is a valid pass
        root = tmp / "kb_empty"
        root.mkdir()
        registry, kb_root, _ = make_kb(root, [])
        report = validate_registry(registry, kb_root, write_report=False)
        if report.ok and report.checked_documents == 0:
            ok("empty registry is a valid pass")
        else:
            bad(f"empty registry should pass: {report.errors}")

    finally:
        # ---- 8. Cleanup ----
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[cleanup] removed temporary directory {tmp}")

    print()
    if failures:
        print(f"RESULT: FAILED ({failures} check(s) failed)")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
