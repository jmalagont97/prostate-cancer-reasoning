#!/bin/bash
# Start embedded Ollama, ensure Gemma 4 E2B is available, then run inference.
set -euo pipefail

MODEL_ID="gemma3n:e2b"

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

# Pull the model only if not baked in (dev convenience); on GC (--network none)
# the model must already exist in OLLAMA_MODELS inside the image.
if ! ollama list | grep -q "${MODEL_ID}"; then
  echo "[entrypoint] model ${MODEL_ID} not found locally — attempting pull"
  ollama pull "${MODEL_ID}"
fi

echo "[entrypoint] warming up ${MODEL_ID}..."
ollama run "${MODEL_ID}" "OK" >/dev/null 2>&1 || true

echo "[entrypoint] launching inference.py"
python3 /opt/app/inference.py
EXIT_CODE=$?

kill "${SERVE_PID}" 2>/dev/null || true
exit "${EXIT_CODE}"
