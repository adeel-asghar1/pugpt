from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.config import settings
from app.services.embedding_service import embedding_service

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Pre-load the FastEmbed ONNX model at import time so the first request does
# not pay the ~200 MB load latency and health reflects reality immediately.
# Failures surface here at startup, not silently per-request.
if settings.embedding_provider.strip().lower() == "fastembed":
    try:
        vec = embedding_service.embed_text("warmup")
        logger.info("FastEmbed model loaded: %s (dim=%d)", embedding_service.embedding_model_name, len(vec))
    except Exception as exc:  # noqa: BLE001 - fail loudly with a clear message
        logger.error(
            "FastEmbed provider FAILED to load at startup (EMBEDDING_PROVIDER=fastembed): %s: %s",
            type(exc).__name__,
            exc,
        )
        raise

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
