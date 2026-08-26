#!/bin/bash
# CHIMERA Generalist Agent — GC entrypoint (self-contained, no network).
# Starts an embedded Ollama daemon (CUDA-capable, baked into the image),
# verifies the baked Gemma 4 E2B model is present, then runs inference.

set -euo pipefail

MODEL_ID="${MODEL_ID:-gemma4:e2b}"

echo "[entrypoint] starting ollama serve..."
ollama serve &
SERVE_PID=$!

# Wait for the API to accept connections.
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "[entrypoint] ollama ready after ${i}s"
    break
  fi
  sleep 1
done

# The model MUST be baked into the image (GC runs with --network none).
# No `ollama pull` here: there is no network at runtime.
if ! ollama list | grep -q "${MODEL_ID}"; then
  echo "[entrypoint] FATAL: model ${MODEL_ID} not found in image (expected baked-in)." >&2
  echo "[entrypoint] OLLAMA_MODELS=${OLLAMA_MODELS:-<default>}" >&2
  kill "${SERVE_PID}" 2>/dev/null || true
  exit 2
fi

echo "[entrypoint] warming up ${MODEL_ID}..."
ollama run "${MODEL_ID}" "OK" >/dev/null 2>&1 || true

echo "[entrypoint] launching inference.py"
python3 /opt/app/inference.py --config /opt/app/configs/config.yaml
EXIT_CODE=$?

kill "${SERVE_PID}" 2>/dev/null || true
exit "${EXIT_CODE}"
