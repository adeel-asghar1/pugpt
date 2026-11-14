"""IngestionService: PDF -> pages -> chunks -> embeddings -> ChromaDB.

Reuses the existing DocumentProcessor, EmbeddingService and ChromaService.
The pipeline order is deliberate: extraction -> chunking -> EMBEDDING FIRST,
and only then the replace of the document's old chunks. If extraction or
embedding fails, previously stored data for that document is untouched.
If the final write fails, a best-effort restore of the backup is attempted.

Duplicate prevention / replacement:
- Each source file gets a stable doc_id = sha256(resolved absolute path)[:16].
- Every chunk is stored under id "<doc_id>:<chunk_index>" with metadata
  {filename, page_number, chunk_index, source_path, doc_id}.
- Before storing, records matching doc_id are deleted (all of them), so
  re-ingesting the same unchanged file is a pure upsert, and re-ingesting an
  updated (shorter or different) file removes obsolete chunks automatically.
- Deleting happens ONLY after the new embeddings are ready, and a backup of
  the old records is kept so a failed write can restore them.
"""

import hashlib
import logging
from pathlib import Path

from app.config import settings
from app.services.chroma_service import ChromaService, chroma_service
from app.services.document_processor import DocumentProcessor, document_processor
from app.services.embedding_service import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        processor: DocumentProcessor | None = None,
        embeddings: EmbeddingService | None = None,
        chroma: ChromaService | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.processor = processor or document_processor
        self.embeddings = embeddings or embedding_service
        self.chroma = chroma or chroma_service
        self.batch_size = batch_size or settings.ingestion_batch_size
        if self.batch_size <= 0:
            raise ValueError("ingestion batch_size must be a positive integer.")

    @staticmethod
    def compute_doc_id(source_path: str | Path) -> str:
        """Stable identifier for a document: path-based, content-independent."""
        resolved = str(Path(source_path).resolve())
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in manageable batches, preserving order."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_embeddings.extend(self.embeddings.embed_texts(batch))
        return all_embeddings

    def _replace_document_chunks(
        self,
        doc_id: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> bool:
        """Delete the doc's old chunks and store the new ones.

        Backs up existing records first; on failure, attempts restore.
        Returns True if pre-existing chunks were replaced.
        """
        existing = self.chroma.get_where({"doc_id": doc_id})

        # ChromaDB returns lists (ids/documents/metadatas) but numpy arrays
        # for embeddings — avoid truthiness tests on arrays.
        def _as_list(value):
            return list(value) if value is not None else []

        backup_ids = _as_list(existing.get("ids"))
        backup_docs = _as_list(existing.get("documents"))
        backup_metadatas = _as_list(existing.get("metadatas"))
        raw_embeddings = existing.get("embeddings")
        backup_embeddings = (
            [[float(v) for v in vector] for vector in raw_embeddings]
            if raw_embeddings is not None and len(raw_embeddings) > 0
            else []
        )

        try:
            if backup_ids:
                self.chroma.delete_where({"doc_id": doc_id})
            self.chroma.add_documents(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
        except Exception:
            if backup_ids:
                try:
                    self.chroma.add_documents(
                        ids=backup_ids,
                        documents=backup_docs,
                        embeddings=backup_embeddings,
                        metadatas=backup_metadatas,
                    )
                    logger.info("Restored previous chunks for document %s", doc_id)
                except Exception as restore_exc:  # best effort
                    logger.error(
                        "CRITICAL: could not restore previous chunks for %s: %s",
                        doc_id,
                        restore_exc,
                    )
            raise
        return bool(backup_ids)

    def ingest_pdf(self, pdf_path: str | Path) -> dict:
        """Ingest one PDF. Never raises; 'status' in the summary says how it went."""
        path = Path(pdf_path)
        summary = {
            "filename": path.name,
            "source_path": str(path.resolve()) if path.exists() else str(path),
            "doc_id": None,
            "pages_extracted": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "replaced_existing": False,
            "status": "failed",
            "error": None,
        }
        try:
            doc_id = self.compute_doc_id(path)
            summary["doc_id"] = doc_id

            pages = self.processor.extract_pages(path)
            summary["pages_extracted"] = len(pages)
            chunks = self.processor.chunk_pages(pages)
            summary["chunks_created"] = len(chunks)
            if not chunks:
                raise ValueError(
                    f"No extractable text in '{path.name}' "
                    "(scanned image PDFs are not supported)."
                )

            texts = [chunk.text for chunk in chunks]
            embeddings = self._embed_in_batches(texts)  # nothing deleted yet

            ids = [f"{doc_id}:{chunk.chunk_index}" for chunk in chunks]
            metadatas = [
                {
                    **chunk.to_metadata(),  # filename, page_number, chunk_index
                    "source_path": summary["source_path"],
                    "doc_id": doc_id,
                }
                for chunk in chunks
            ]
            replaced = self._replace_document_chunks(doc_id, ids, texts, embeddings, metadatas)

            summary.update(
                chunks_stored=len(ids),
                replaced_existing=replaced,
                status="success",
            )
        except Exception as exc:  # noqa: BLE001 - per-document failures must not kill directory runs
            summary["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Ingestion of '%s' failed: %s", path.name, summary["error"])
        return summary


# Shared instance built from central configuration and shared services.
ingestion_service = IngestionService()
