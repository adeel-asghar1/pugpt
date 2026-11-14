"""DocumentProcessor: PDF text extraction and text chunking.

Pure text pipeline — no embeddings, no ChromaDB, no ingestion.

- extract_pages(pdf_path) -> list[ExtractedPage]
    One entry per non-empty page, preserving 1-based page numbers
    and the original filename.
- chunk_pages(pages) -> list[TextChunk]
    Splits page text into ~chunk_size-character chunks with
    chunk_overlap-character overlap, preferring word boundaries.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.config import settings

_WHITESPACE = re.compile(r"\s")

logger = logging.getLogger(__name__)


class DocumentProcessorError(Exception):
    """Base error for document processing failures."""


class PDFNotFoundError(DocumentProcessorError):
    """The given path does not exist or is not a file."""


class InvalidPDFError(DocumentProcessorError):
    """The file could not be opened or read as a PDF."""


@dataclass(frozen=True)
class ExtractedPage:
    filename: str
    page_number: int  # 1-based, as shown in PDF viewers
    text: str


@dataclass(frozen=True)
class TextChunk:
    filename: str
    page_number: int
    chunk_index: int  # 0-based, within the whole document
    text: str

    def to_metadata(self) -> dict:
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    # ---------------- PDF extraction ----------------

    def extract_pages(self, pdf_path: str | Path) -> list[ExtractedPage]:
        path = Path(pdf_path)
        if not path.exists():
            raise PDFNotFoundError(f"PDF not found: {path}")
        if not path.is_file():
            raise PDFNotFoundError(f"Path is not a file: {path}")

        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise InvalidPDFError(
                f"Could not open '{path.name}' as a PDF: {exc}"
            ) from exc

        pages: list[ExtractedPage] = []
        try:
            if document.needs_pass:
                raise InvalidPDFError(
                    f"'{path.name}' is password-protected; encrypted PDFs are not supported."
                )
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue  # ignore empty pages (e.g. images, blank pages)
                pages.append(
                    ExtractedPage(filename=path.name, page_number=index, text=text)
                )
        except InvalidPDFError:
            raise
        except Exception as exc:
            raise InvalidPDFError(
                f"Failed to extract text from '{path.name}': {exc}"
            ) from exc
        finally:
            document.close()

        if not pages:
            logger.warning(
                "No extractable text in '%s'. The PDF may be a scanned image; "
                "OCR is not supported.",
                path.name,
            )
        return pages

    # ---------------- Chunking ----------------

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0
        for page in pages:
            for piece in self._split_text(page.text):
                chunks.append(
                    TextChunk(
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=piece,
                    )
                )
                chunk_index += 1
        return chunks

    def extract_and_chunk(self, pdf_path: str | Path) -> list[TextChunk]:
        return self.chunk_pages(self.extract_pages(pdf_path))

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        length = len(text)
        if length == 0:
            return []
        if length <= self.chunk_size:
            return [text]  # short text stays intact

        pieces: list[str] = []
        start = 0
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                # Prefer ending at the last whitespace boundary in the window.
                boundaries = list(_WHITESPACE.finditer(text, start, end))
                if boundaries and boundaries[-1].start() > start:
                    end = boundaries[-1].start()
            piece = text[start:end].strip()
            if piece:  # never emit empty chunks
                pieces.append(piece)
            if end >= length:
                break
            # Overlap: start the next chunk before this one ends.
            next_start = max(0, end - self.chunk_overlap)
            if next_start > 0:
                # Avoid starting mid-word: begin at the next space boundary
                # within the overlap region (or drop the overlap).
                match = re.search(r"\s", text[next_start:end])
                next_start = next_start + match.start() + 1 if match else end
            # Safety: never go backwards (termination guarantee).
            start = next_start if next_start > start else end
        return pieces


# Shared instance built from central configuration.
document_processor = DocumentProcessor()
