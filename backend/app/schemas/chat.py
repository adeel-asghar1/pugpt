from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class Source(BaseModel):
    """A reference to the context the answer was drawn from.

    PDFs keep filename + page_number. Live official-PU web sources carry
    source_type="web", the clickable URL, its host as `filename`, the
    provider-reported published date (never invented) as `published_date`,
    and the page title. Results are always labeled — never mixed silently.
    """

    filename: str
    page_number: int | None = None
    snippet: str | None = None
    distance: float | None = None
    chunk_index: int | None = None
    source_type: Literal["pdf", "web"] = "pdf"
    url: str | None = None
    published_date: str | None = None
    title: str | None = None


ChatStatus = Literal[
    "answered",              # grounded answer generated from retrieved context
    "no_relevant_context",   # nothing passed the relevance threshold; LLM not called
    "empty_knowledge_base",  # no documents ingested yet
    "error",                 # infrastructure/service failure; details are server-side only
]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    status: ChatStatus = "answered"
