# CHIMERA Task-1 GC Agent (minimized)

Agente de orquestación fija para la Task 1 (decisión de biopsia, formato Grand
Challenge). El LLM (**Gemma 4 E2B vía Ollama embebida — invariante**) nunca
toma decisiones de subtarea ni emite tool calls: solo redacta el reasoning.

## Pipeline (por caso)

```
structured-prompt.json
  → Tool A get_biopsy_decision_support   (subtask1_ensemble.pkl)
      decision preliminar + probabilidad + vecinos KNN con free_texts ejemplares
  → Tool B get_confidence_estimate       (subtask2_kdm.pkl)   confidence preliminar
  → Tool C get_variable_weights_estimate (subtask3_extratrees.pkl) pesos preliminares
  → form_fill: 1 llamada Gemma E2B → free_text anclado en ejemplares
  → Task1Output (Pydantic); campos duros sobrescritos por los submodelos
  → /output/prostate-biopsy-decision.json + ...-reasoning.json
```

## Estructura

- `src/chimera_agent_baseline/mcp_server.py` — servidor MCP stdio con las 3 tools.
- `src/chimera_agent_baseline/run.py` — orquestador fijo (sin ReAct).
- `src/chimera_agent_baseline/form_fill.py` — prompt-and-parse + garantía dura.
- `models/ollama_compat.py` — ChatOllama (Gemma 3n E2B, localhost).
- `inference.py` + `entrypoint.sh` + `Dockerfile` — contenedor GC autocontenido.

## Desarrollo local

Requisitos: Ollama corriendo con `gemma3n:e2b` (`ollama pull gemma3n:e2b`).

```bash
pip install -e .
python inference.py --input test/input/interf0 --output /tmp/out \
    models-dir=../../training/models
```

## Build GC

```bash
docker build -t chimera-task1-agent .
```

Los pickles de `deploy/tasks/task1/training/models/` se montan en runtime en
`/opt/ml/model` (convención GC) y se localizan vía `paths.models_dir`.
El modelo Gemma debe hornearse en la imagen (`OLLAMA_MODELS`) porque en
evaluación el contenedor corre con `--network none`.
