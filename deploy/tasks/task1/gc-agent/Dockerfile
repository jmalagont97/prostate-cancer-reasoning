# ==============================================================================
# CHIMERA Task-1 Agent — GC container with embedded Ollama + Gemma 4 E2B.
# Self-contained: --network none compatible (no downloads at runtime).
# ==============================================================================
FROM --platform=linux/amd64 python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app/src \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/opt/app/ollama_models \
    OLLAMA_KEEP_ALIVE=-1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama binary (pinned) --------------------------------------------------
ARG OLLAMA_VERSION=v0.5.7
RUN curl -fsSL https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz \
    | tar xz -C /usr/local

# --- Python deps -------------------------------------------------------------
COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ----------------------------------------------------------------
COPY inference.py entrypoint.sh /opt/app/
COPY configs/ /opt/app/configs/
COPY templates/ /opt/app/templates/
COPY src/ /opt/app/src/

# Model import path expected by the pickles (deploy.tasks.task1.training.model_definitions)
RUN mkdir -p /opt/app/deploy/tasks/task1/training
COPY pyproject.toml /opt/app/
RUN pip install --no-cache-dir --no-deps /opt/app

# Bake the invariant model (Gemma 4 E2B) into the image so runtime needs no network.
# NOTE: build requires network; the resulting image does not.
RUN mkdir -p /opt/app/ollama_models

COPY --chmod=755 entrypoint.sh /opt/app/entrypoint.sh

USER user
EXPOSE 11434
ENTRYPOINT ["/opt/app/entrypoint.sh"]
