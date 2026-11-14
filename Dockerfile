# PUGPT — production image for Render (or any container host).
# Single worker, no reloads, no dev tooling.
# Build from the REPOSITORY ROOT (render.yaml uses dockerfilePath: ./Dockerfile).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    EMBEDDING_PROVIDER=fastembed \
    FASTEMBED_MODEL_NAME=nomic-ai/nomic-embed-text-v1.5-Q \
    CHROMA_COLLECTION=pugpt_knowledge_fastembed_test \
    CHROMA_PERSIST_PATH=/app/chroma \
    FASTEMBED_CACHE_DIR=/root/.cache/fastembed \
    LLM_PROVIDER=groq

WORKDIR /app

# runtime deps (slim has no libstdc++6 certificate — required by onnxruntime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# install pinned Linux runtime dependencies (requirements-docker.txt).
# backend/requirements.txt is Windows-pinned (pymupdf==2.1 has no manylinux wheel);
# deployment/requirements-docker.txt is the Linux equivalent used by this image.
COPY deployment/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# application code + packaged vector DB (519 chunks, FastEmbed indexing)
COPY backend/app ./backend/app
COPY deployment/chroma_fastembed/chroma /app/chroma

# pre-download the FastEmbed model so runtime startup never needs the network
RUN python -c "from fastembed import TextEmbedding; m = TextEmbedding('nomic-ai/nomic-embed-text-v1.5-Q'); print('fastembed model cached OK', m._model_name if hasattr(m, '_model_name') else 'ok')"

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/api/health', timeout=5)" || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
