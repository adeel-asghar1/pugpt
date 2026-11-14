"""Validate the PUGPT knowledge base registry and its PDFs.

Checks (per entry in knowledge_base/metadata/sources.json):
  CRITICAL (exit code 1):
    - registry JSON parseable, top-level "documents" is a list
    - required fields present and non-empty (see REQUIRED_FIELDS)
    - category is one of the seven knowledge-base folders
    - filename is a bare filename (no path traversal)
    - verification_status is pending | verified | rejected
    - source_url parses and its host is an official PU domain (pu.edu.pk / *.pu.edu.pk)
    - referenced PDF exists at documents/<category>/<filename>
    - duplicate document_id or duplicate filename
  WARNING (reported, exit code unchanged):
    - published_date / last_updated / effective_session missing
    - source_url uses http instead of https
    - PDF contains no extractable text (scanned; needs OCR)
    - document_type outside the suggested list
  INFO:
    - verification_status is 'pending' (not yet verified)

Does NOT ingest anything and does NOT auto-verify documents.

Run from the backend/ directory:
    python scripts/validate_knowledge_base.py

Optional flags: --registry PATH --kb-root PATH --report PATH
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_ROOT = PROJECT_ROOT / "knowledge_base"
DEFAULT_REGISTRY = DEFAULT_KB_ROOT / "metadata" / "sources.json"
DEFAULT_REPORT = DEFAULT_KB_ROOT / "reports" / "validation_latest.md"

ALLOWED_CATEGORIES = {
    "admissions",
    "academics",
    "examinations",
    "fees",
    "scholarships",
    "faculties_programs",
    "student_services",
}

# Official University of the Punjab domains: host must equal or be a subdomain of pu.edu.pk.
OFFICIAL_PU_DOMAIN = "pu.edu.pk"

REQUIRED_FIELDS = [
    "document_id",
    "title",
    "filename",
    "category",
    "source_url",
    "publisher",
    "document_type",
    "verification_status",
]

DATE_FIELDS = ["published_date", "last_updated", "effective_session"]
VALID_STATUSES = {"pending", "verified", "rejected"}
SUGGESTED_TYPES = {
    "policy",
    "prospectus",
    "notification",
    "fee schedule",
    "bye laws",
    "guidelines",
    "circular",
    "form",
    "report",
    "other",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    checked_documents: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_official_pu_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL could not be parsed"
    if parsed.scheme not in ("http", "https"):
        return False, f"URL scheme '{parsed.scheme or '<missing>'}' is not http/https"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no host"
    if host == OFFICIAL_PU_DOMAIN or host.endswith("." + OFFICIAL_PU_DOMAIN):
        return True, ""
    return False, f"host '{host}' is not an official University of the Punjab domain (*.{OFFICIAL_PU_DOMAIN})"


def _pdf_has_extractable_text(pdf_path: Path) -> bool:
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            return any(page.get_text("text").strip() for page in doc)
    except Exception:
        # Unreadable/corrupt PDF: treat as extractable-attempted; report elsewhere.
        return False


def validate_registry(
    registry_path: Path,
    kb_root: Path,
    write_report: bool = True,
    report_path: Path | None = None,
) -> ValidationReport:
    report = ValidationReport()
    registry_path = Path(registry_path)
    kb_root = Path(kb_root)

    # ---- 1-2. Load and structurally validate the registry ----
    if not registry_path.exists():
        report.errors.append(f"Registry not found: {registry_path}")
        return report
    try:
        raw = registry_path.read_text(encoding="utf-8")
        registry = json.loads(raw)
    except json.JSONDecodeError as exc:
        report.errors.append(f"Registry is not valid JSON: {exc}")
        return report

    documents = registry.get("documents")
    if not isinstance(documents, list):
        report.errors.append("Registry must contain a top-level 'documents' list.")
        return report

    if not documents:
        report.info.append("Registry is empty - nothing to validate yet.")

    # ---- 5 (pre-pass). Duplicate IDs and filenames ----
    seen_ids: dict[str, int] = {}
    seen_filenames: dict[str, int] = {}
    for index, entry in enumerate(documents):
        if isinstance(entry, dict):
            doc_id = str(entry.get("document_id", "")).strip()
            filename = str(entry.get("filename", "")).strip()
            if doc_id:
                seen_ids.setdefault(doc_id, index)
            if filename:
                seen_filenames.setdefault(filename, index)
    for doc_id, first in seen_ids.items():
        count = sum(1 for e in documents if isinstance(e, dict) and str(e.get("document_id", "")).strip() == doc_id)
        if count > 1:
            report.errors.append(f"Duplicate document_id '{doc_id}' used {count} times (first at index {first}).")
    for filename, first in seen_filenames.items():
        count = sum(1 for e in documents if isinstance(e, dict) and str(e.get("filename", "")).strip() == filename)
        if count > 1:
            report.errors.append(f"Duplicate filename '{filename}' used {count} times (first at index {first}).")

    report.checked_documents = len(documents)

    # ---- 3, 4, 6, 7. Per-document checks ----
    for index, entry in enumerate(documents):
        label = f"documents[{index}]"
        if not isinstance(entry, dict):
            report.errors.append(f"{label}: entry must be an object, got {type(entry).__name__}.")
            continue

        doc_id = str(entry.get("document_id", "")).strip()
        name = doc_id or str(entry.get("filename", "")).strip() or label

        for field_name in REQUIRED_FIELDS:
            value = entry.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                report.errors.append(f"{name}: required field '{field_name}' is missing or empty.")

        category = str(entry.get("category", "")).strip()
        if category and category not in ALLOWED_CATEGORIES:
            report.errors.append(f"{name}: category '{category}' is not one of {sorted(ALLOWED_CATEGORIES)}.")

        filename = str(entry.get("filename", "")).strip()
        if filename and (Path(filename).name != filename):
            report.errors.append(f"{name}: filename '{filename}' must be a bare filename, not a path.")

        status = str(entry.get("verification_status", "")).strip().lower()
        if status and status not in VALID_STATUSES:
            report.errors.append(f"{name}: verification_status '{status}' must be one of {sorted(VALID_STATUSES)}.")
        elif status == "pending":
            report.info.append(f"{name}: verification_status is 'pending' — review before trusting.")
        elif status == "rejected":
            report.info.append(f"{name}: verification_status is 'rejected' — do NOT ingest this document.")

        url = str(entry.get("source_url", "")).strip()
        if url:
            official, reason = _is_official_pu_url(url)
            if not official:
                report.errors.append(f"{name}: source_url not official: {reason}")
            elif url.lower().startswith("http://"):
                report.warnings.append(f"{name}: source_url uses http, not https.")

        dtype = str(entry.get("document_type", "")).strip().lower()
        if dtype and dtype not in SUGGESTED_TYPES:
            report.warnings.append(f"{name}: document_type '{dtype}' is outside the suggested list {sorted(SUGGESTED_TYPES)}.")

        for date_field in DATE_FIELDS:
            value = entry.get(date_field)
            if value is None or (isinstance(value, str) and not value.strip()):
                report.warnings.append(f"{name}: '{date_field}' is missing — record it if the document states it.")

        # ---- 3 & 7. Referenced PDF must exist and contain text ----
        if category in ALLOWED_CATEGORIES and filename and Path(filename).name == filename:
            pdf_path = kb_root / "documents" / category / filename
            if not pdf_path.exists():
                report.errors.append(f"{name}: referenced PDF not found at {pdf_path}")
            elif pdf_path.suffix.lower() != ".pdf":
                report.errors.append(f"{name}: referenced file is not a .pdf: {pdf_path.name}")
            elif not _pdf_has_extractable_text(pdf_path):
                report.warnings.append(f"{name}: PDF contains no extractable text — likely scanned; OCR required.")

    if write_report:
        report_path = Path(report_path) if report_path else kb_root / "reports" / "validation_latest.md"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_markdown_report(report), encoding="utf-8")
        except OSError as exc:
            report.warnings.append(f"Could not write report file {report_path}: {exc}")
    return report


def render_markdown_report(report: ValidationReport) -> str:
    lines = [
        "# PUGPT Knowledge Base Validation Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Documents checked: {report.checked_documents}",
        f"Result: {'PASS' if report.ok else 'FAIL'}",
        "",
        f"## Errors ({len(report.errors)})",
        *([f"- {e}" for e in report.errors] or ["- none"]),
        "",
        f"## Warnings ({len(report.warnings)})",
        *([f"- {w}" for w in report.warnings] or ["- none"]),
        "",
        f"## Info ({len(report.info)})",
        *([f"- {i}" for i in report.info] or ["- none"]),
        "",
    ]
    return "\n".join(lines)


def print_report(report: ValidationReport) -> None:
    print("=" * 60)
    print("PUGPT Knowledge Base Validation")
    print("=" * 60)
    print(f"Documents checked: {report.checked_documents}\n")

    def section(title, items):
        print(f"[{title}] ({len(items)})")
        if items:
            for item in items:
                print(f"  - {item}")
        else:
            print("  (none)")
        print()

    section("ERRORS", report.errors)
    section("WARNINGS", report.warnings)
    section("INFO", report.info)
    print("RESULT:", "PASS" if report.ok else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the PUGPT knowledge base registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.report
    if report_path is None:
        report_path = args.kb_root / "reports" / "validation_latest.md"

    report = validate_registry(args.registry, args.kb_root, write_report=True, report_path=report_path)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
