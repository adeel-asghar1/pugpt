from fastapi import APIRouter

from app.config import settings
from app.services.generation_service import generation_service
from app.services.embedding_service import embedding_service
from app.services.ollama_service import ollama_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Backend + Ollama status, generation-provider config and web-search config.

    Always responds with HTTP 200 so callers can distinguish 'backend
    is running but Ollama is not' from a crashed backend. Never performs
    a live Groq or Tavily request and never exposes any API key — only
    provider names, configured models and whether keys are present.

    The embedding block reports the CONFIGURED provider and (when known)
    whether it is loaded/usable — it never embeds text here.
    """
    ollama = ollama_service.status()
    try:
        provider = generation_service.resolved_provider()
        generation_model = (
            settings.groq_chat_model if provider == "groq" else settings.ollama_chat_model
        )
        generation_info = {
            "provider": provider,
            "model": generation_model,
            "api_key_configured": bool(settings.groq_api_key.strip()),
        }
    except Exception:  # invalid LLM_PROVIDER — report config issue without secrets
        generation_info = {
            "provider": "invalid",
            "model": None,
            "api_key_configured": bool(settings.groq_api_key.strip()),
        }

    embedding_provider = (settings.embedding_provider or "fastembed").strip().lower()
    try:
        dimension = embedding_service.dimension
        # FastEmbed reports reachable=True after first successful embed; until
        # then the app may still be healthy — report model + dimension and
        # let callers judge via the load-on-first-use behaviour.
        embedding_info = {
            "provider": embedding_provider,
            "model": getattr(embedding_service, "embedding_model_name", None),
            "dimension": dimension,
            "reachable": embedding_service.status().get("reachable") if hasattr(embedding_service, "status") else None,
        }
    except Exception:  # pragma: no cover - defensive
        embedding_info = {
            "provider": embedding_provider,
            "model": None,
            "dimension": None,
            "reachable": False,
        }

    web_search_info = {
        # Booleans only — never the key, never a live search, never a URL.
        "enabled": settings.web_search_enabled,
        "provider": settings.web_search_provider,
        "api_key_configured": bool(settings.tavily_api_key.strip()),
        "on_recency": settings.web_search_on_recency,
    }
    return {
        "status": "ok",
        "service": settings.app_name,
        "generation": generation_info,
        "embedding": embedding_info,
        "web_search": web_search_info,
        "ollama": {
            "reachable": ollama["reachable"],
            "model": ollama["model"],
            "model_available": ollama["model_available"],
            "error": ollama["error"],
        },
    }
