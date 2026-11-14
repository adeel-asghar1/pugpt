"""EmbeddingService: text -> embedding vectors.

Provider-dispatched through configuration (EMBEDDING_PROVIDER):

- "ollama"    (default fallback for local/dev and emergency rollback)
              -> Ollama /api/embed via nomic-embed-text:latest
- "fastembed" (production default)
              -> FastEmbed nomic-ai/nomic-embed-text-v1.5-Q, fully in-process
                 — no Ollama daemon or external network call required after
                 model files are cached (downloaded on first use).

PUGPT MUST NOT silently fall back between providers: if the selected provider
is unavailable the caller gets a clear error (health surfaces it) rather than
the wrong embedding space (vectors differ between providers — see README).

Both providers produce 768-dimensional vectors. ChromaDB uses cosine distance.
Document and query embeddings MUST come from the SAME provider and model —
switching providers requires re-ingestion or a matching re-indexed collection
(see README's "Collection selection and rollback").
"""

import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_FASTEMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5-Q"
_FASTEMBED_DIM = 768
_OLLAMA_DIM = 768


class EmbeddingServiceError(Exception):
    """Base class for embedding failures. Safe to surface in health output."""


class EmbeddingService:
    """Ollama-backed embedding service (kept for dev/rollback)."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _normalize_inputs(texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Provide a non-empty text string or a non-empty list of strings.")
        cleaned = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {index} is not a string.")
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Text at index {index} is empty or whitespace-only.")
            cleaned.append(stripped)
        return cleaned

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def embedding_model_name(self) -> str:
        return self.model

    @property
    def dimension(self) -> int:
        return _OLLAMA_DIM

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            with self._client() as client:
                response = client.post(
                    "/api/embed",
                    json={"model": self.model, "input": texts},
                )
        except httpx.ConnectError as exc:
            raise EmbeddingServiceError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Try: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError(
                f"Ollama at {self.base_url} did not respond within {self.timeout}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingServiceError(
                f"Ollama /api/embed responded with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            embeddings = response.json()["embeddings"]
        except (ValueError, KeyError) as exc:
            raise EmbeddingServiceError(
                "Ollama /api/embed returned an unexpected response (missing 'embeddings')."
            ) from exc

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingServiceError(
                f"Expected {len(texts)} embedding vector(s), "
                f"got {len(embeddings) if isinstance(embeddings, list) else 'none'}."
            )
        return [list(float(v) for v in row) for row in embeddings]

    def embed_texts(self, texts: str | list[str]) -> list[list[float]]:
        return self._post_embed(self._normalize_inputs(texts))

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts(text)[0]

    def status(self) -> dict:
        # Ollama is pinged by its own /api/tags health path; we avoid a
        # redundant live HTTP call here — report configuration + dimension only.
        return {
            "reachable": None,  # unknown without a live call
            "provider": "ollama",
            "model": self.model,
            "dimension": _OLLAMA_DIM,
            "error": None,
        }


class FastEmbedService:
    """FastEmbed nomic-embed-text-v1.5-Q service (production default).

    The ONNX model is loaded lazily on first use and reused thereafter.
    No network call occurs after the model files are cached locally.
    """

    def __init__(self, model_name: str, dim: int = _FASTEMBED_DIM) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def embedding_model_name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        return self.dim

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # local import — costs nothing if unused
            except ImportError as exc:
                raise EmbeddingServiceError(
                    "fastembed is not installed. Install it with: pip install fastembed==0.9.1"
                ) from exc
            try:
                self._model = TextEmbedding(self.model_name)
            except Exception as exc:  # noqa: BLE001 - remap to safe message
                raise EmbeddingServiceError(
                    f"FastEmbed failed to load '{self.model_name}': {type(exc).__name__}: {exc}"
                ) from exc
        return self._model

    @staticmethod
    def _normalize_inputs(texts: str | list[str]) -> list[str]:
        return EmbeddingService._normalize_inputs(texts)

    def embed_texts(self, texts: str | list[str]) -> list[list[float]]:
        cleaned = self._normalize_inputs(texts)
        try:
            model = self._get_model()
            embedded = list(model.embed(cleaned))
        except EmbeddingServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingServiceError(
                f"FastEmbed embedding failed: {type(exc).__name__}: {exc}"
            ) from exc
        results: list[list[float]] = []
        for vector in embedded:
            if hasattr(vector, "tolist"):
                row = vector.tolist()
            else:
                row = list(vector)
            floats = [float(v) for v in row]
            if len(floats) != self.dim:
                raise EmbeddingServiceError(
                    f"FastEmbed returned dim={len(floats)}, expected {self.dim} for {self.model_name}."
                )
            results.append(floats)
        return results

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts(text)[0]

    def status(self) -> dict:
        return {
            "reachable": self._model is not None,   # lazy; becomes True after first use
            "model": self.model_name,
            "provider": "fastembed",
            "dimension": self.dim,
            "error": None,
        }


def _build_default():
    provider = (settings.embedding_provider or "fastembed").strip().lower()
    if provider == "fastembed":
        return FastEmbedService(settings.fastembed_model_name or _FASTEMBED_MODEL_NAME)
    if provider == "ollama":
        return EmbeddingService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
        )
    raise EmbeddingServiceError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Expected one of: fastembed, ollama."
    )


# Shared services built from configuration. Tests may inject their own.
fastembed_service = FastEmbedService(settings.fastembed_model_name or _FASTEMBED_MODEL_NAME)
ollama_embedding_service = EmbeddingService(
    base_url=settings.ollama_base_url,
    model=settings.ollama_embed_model,
    timeout=settings.ollama_timeout,
)
embedding_service = _build_default()
