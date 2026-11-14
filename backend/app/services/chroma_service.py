"""ChromaService: all ChromaDB (vector database) operations live here.

API routes never talk to ChromaDB directly — they use this service.
Initialization failures are captured in `last_error` instead of crashing
the app; methods raise a clear RuntimeError when the database is unavailable.
"""

from pathlib import Path

import chromadb

from app.config import settings


class ChromaService:
    def __init__(self, persist_path: Path, collection_name: str) -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.last_error: str | None = None
        self._client = None
        self._collection = None

        try:
            # Fail early with a clear message on path/permission problems.
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - surface any init failure clearly
            self.last_error = (
                f"Could not initialize ChromaDB at '{self.persist_path}': {exc}"
            )

    @property
    def is_ready(self) -> bool:
        return self._collection is not None

    def _require_collection(self):
        if self._collection is None:
            raise RuntimeError(
                self.last_error or "ChromaDB collection is not available."
            )
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update documents. Embeddings are always supplied by the
        caller (nomic-embed-text via the embedding service, never Chroma's
        default embedder)."""
        collection = self._require_collection()
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict:
        """Return the top_k most similar documents for an embedding."""
        collection = self._require_collection()
        n_results = top_k or settings.retrieval_top_k
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def query_filtered(
        self,
        query_embedding: list[float],
        where: dict,
        top_k: int = 10,
    ) -> dict:
        """Same as `query`, restricted to records matching a metadata filter.

        Used only to ADD candidates (never to bypass the relevance
        threshold); callers still apply the original-distance threshold.
        """
        if not where:
            raise ValueError("query_filtered requires a non-empty where filter.")
        collection = self._require_collection()
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        collection = self._require_collection()
        return collection.count()

    def get_where(
        self,
        where: dict,
        include: list[str] | None = None,
    ) -> dict:
        """Fetch records matching a metadata filter (used for backups
        during re-ingestion). Defaults to ids+documents+embeddings+metadatas."""
        collection = self._require_collection()
        return collection.get(
            where=where,
            include=include or ["documents", "embeddings", "metadatas"],
        )

    def get_all(self, include: list[str] | None = None) -> dict:
        """Fetch every record in the collection (used for audits and
        metadata migrations)."""
        collection = self._require_collection()
        return collection.get(include=include or ["documents", "metadatas"])

    def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
        """Replace metadata for existing records in place. IDs and
        embeddings are untouched."""
        collection = self._require_collection()
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length.")
        collection.update(ids=ids, metadatas=metadatas)

    def delete_where(self, where: dict) -> int:
        """Delete records matching a metadata filter. Returns number deleted."""
        existing = self.get_where(where, include=["documents"])
        ids = existing.get("ids") or []
        if ids:
            self._require_collection().delete(where=where)
        return len(ids)

    def delete_collection(self, name: str | None = None) -> None:
        """Delete a collection. Used for tests and maintenance.

        Goes through this service's client so settings always match.
        Defaults to this service's own collection.
        """
        if self._client is None:
            raise RuntimeError(self.last_error or "ChromaDB client is not available.")
        target = name or self.collection_name
        self._client.delete_collection(target)


# Shared instance built from central configuration.
chroma_service = ChromaService(
    persist_path=settings.chroma_persist_path,
    collection_name=settings.chroma_collection,
)
