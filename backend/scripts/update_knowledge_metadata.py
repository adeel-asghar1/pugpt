"""One-time metadata migration for pugpt_knowledge (Step 16).

- Reads the source registry (knowledge_base/metadata/sources.json).
- Fetches ALL records and writes a full backup of (ids, metadatas) BEFORE changing anything
  (target: --backup-path, default knowledge_base/reports/chroma_metadata_backup_step16.json;
  refuses to overwrite an existing backup file).
- Merges registry document-level fields into each record's metadata where the
  filename matches a registry entry:
    document_id, category, document_type, effective_session,
    verification_status, campus, program
- Leaves records unchanged when the filename is not found in the registry
  (reported, not touched).
- Verifies afterwards: count unchanged, ID set unchanged.

Usage:
    python scripts/update_knowledge_metadata.py [--dry-run] [--backup-path PATH]

Options:
    --dry-run       show what would change; write nothing
    --backup-path   where to write the pre-enrichment backup. Defaults to
                    knowledge_base/reports/chroma_metadata_backup_step16.json
                    (unchanged default behavior). The script REFUSES to run
                    (without --dry-run) if the requested backup path already
                    exists — existing backups are never overwritten.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chroma_service import chroma_service
from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "knowledge_base" / "metadata" / "sources.json"
DEFAULT_BACKUP_PATH = PROJECT_ROOT / "knowledge_base" / "reports" / "chroma_metadata_backup_step16.json"
BACKUP_PATH = DEFAULT_BACKUP_PATH  # backwards-compatible alias for existing imports

REGISTRY_FIELDS = [
    "document_id",
    "category",
    "document_type",
    "effective_session",
    "verification_status",
    "campus",
    "program",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would change; write nothing")
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=DEFAULT_BACKUP_PATH,
        help=(
            "path for the pre-enrichment backup. Default: "
            "knowledge_base/reports/chroma_metadata_backup_step16.json "
            "(existing default behavior). The script refuses to proceed if the "
            "requested backup path already exists."
        ),
    )
    args = parser.parse_args()
    backup_path: Path = args.backup_path

    # Refuse to overwrite anything before touching the database.
    if backup_path.exists():
        print(f"Error: backup path already exists: {backup_path}")
        print("Refusing to overwrite an existing backup file.")
        if not args.dry_run:
            return 1
        print("[dry-run] continuing without writing the backup.")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_filename = {entry["filename"]: entry for entry in registry["documents"]}
    print(f"Registry: {len(by_filename)} entries")

    records = chroma_service.get_all(include=["metadatas"])
    ids = records["ids"]
    metadatas = records["metadatas"] or []
    before_count = len(ids)
    print(f"Collection records: {before_count}")

    # ---- backup before touching anything ----
    backup = [{"id": id_, "metadata": meta} for id_, meta in zip(ids, metadatas)]
    if not args.dry_run:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(json.dumps(backup, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Backup written: {backup_path} ({backup_path.stat().st_size:,} bytes)")
    else:
        print(f"[dry-run] backup NOT written (target: {backup_path})")

    # ---- merge ----
    new_metadatas = []
    updated = skipped = 0
    skipped_files: set[str] = set()
    for id_, meta in zip(ids, metadatas):
        meta = dict(meta)
        filename = meta.get("filename")
        entry = by_filename.get(filename)
        if entry is None:
            skipped += 1
            skipped_files.add(str(filename))
            new_metadatas.append(meta)
            continue
        for field in REGISTRY_FIELDS:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                meta[field] = value  # string values only — Chroma-compatible
        new_metadatas.append(meta)
        updated += 1

    print(f"records to enrich: {updated}")
    if skipped:
        print(f"records left unchanged (no registry match): {skipped} -> {sorted(skipped_files)}")

    changed = sum(1 for a, b in zip(metadatas, new_metadatas) if a != b)
    print(f"records whose metadata would change: {changed}")

    if args.dry_run:
        print("[dry-run] collection NOT modified")
        return 0

    chroma_service.update_metadata(ids, new_metadatas)

    # ---- verify ----
    after = chroma_service.get_all(include=["metadatas"])
    after_ids = set(after["ids"])
    print(f"count after: {len(after_ids)}  (unchanged: {len(after_ids) == before_count})")
    print(f"id set unchanged: {after_ids == set(ids)}")
    sample_meta = next(
        m for m in after["metadatas"] if m.get("filename") == "Merit-of-Last-Year-2025.pdf"
    )
    print("sample Merit-2025 record metadata:")
    print(json.dumps(sample_meta, indent=2, ensure_ascii=False))
    return 0 if (len(after_ids) == before_count and after_ids == set(ids)) else 1


if __name__ == "__main__":
    sys.exit(main())
