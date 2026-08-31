#!/bin/bash
# CHIMERA Generalist Agent — GC entrypoint (self-contained, no network).
# Starts an embedded Ollama daemon (CUDA-capable, baked into the image),
# verifies the baked Gemma 4 E2B model is present, then runs inference.

set -euo pipefail

MODEL_ID="${MODEL_ID:-gemma4:e2b}"
READY_FILE="${CHIMERA_READY_FILE:-/tmp/chimera-gemma.ready}"

rm -f "${READY_FILE}"

cleanup() {
  kill "${OLLAMA_PID:-}" "${API_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start the HTTP adapter immediately. Grand Challenge probes /health before
# the model is necessarily loaded; /invoke is guarded by READY_FILE below.
echo "[entrypoint] launching Grand Challenge HTTP API on port ${PORT:-4743}"
python3 -m chimera_agent.server &
API_PID=$!

echo "[entrypoint] starting ollama serve..."
ollama serve &
OLLAMA_PID=$!

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
  exit 2
fi

echo "[entrypoint] warming up ${MODEL_ID}..."
if ! ollama run "${MODEL_ID}" "OK" >/dev/null 2>&1; then
  echo "[entrypoint] FATAL: model warm-up failed for ${MODEL_ID}." >&2
  exit 3
fi

touch "${READY_FILE}"
echo "[entrypoint] model ready; accepting invocations"
wait "${API_PID}"
