# CHIMERA Generalist Agent (`deploy/agent/`)

Agente generalista para las **tres tareas** del reto de razonamiento en cáncer de próstata.
Esta carpeta es la implementación productiva; el **baseline funcional conservado** permanece en
`deploy/tasks/task1/gc-agent/` y no debe modificarse.

## Estado

- **Task 1 — Biopsy decision:** implementada de forma conservadora vía `src/chimera_agent/tasks/task1/baseline/` (traslado sin refactor científico del pipeline, modelos, prompt y contrato JSON). Produce los dos archivos GC requeridos:
  - `prostate-biopsy-decision.json`
  - `prostate-biopsy-decision-reasoning.json`
  además de `prediction.json`.
- **Task 2 y Task 3:** placeholders explícitos — fallan de forma controlada con `NotImplementedError`. No se inventan entradas, salidas ni modelos; los espacios quedan listos para añadir cada tarea cuando corresponda.

## Layout

```
deploy/agent/
  configs/
    config.yaml              # base task-agnostic
    tasks/task1.yaml         # override Task 1
    tasks/task2.yaml         # placeholder
    tasks/task3.yaml         # placeholder
  src/chimera_agent/
    dispatch.py              # VALID_TASK_IDS={1,2,3}, validación antes de cargar modelos
    tasks/task1/baseline/    # código conservado de gc-agent
    tasks/task2/__init__.py  # NotImplementedError
    tasks/task3/__init__.py  # NotImplementedError
  templates/prompts/task1/agent_prompt.j2
  inference.py               # merge explícito OmegaConf + staging GC /input
  entrypoint.sh
  Dockerfile
```

## Flujo de ejecución

1. `inference.py` carga `configs/config.yaml` y hace `OmegaConf.merge` explícito con `configs/tasks/taskN.yaml` (no usa Hydra `defaults` implícito).
2. Si detecta `/input/structured-prompt.json` (GC), lo estadifica a `/tmp/gc_cases/task1/agent_input/<case_id>/structured-prompt.json` para que el loader baseline lo encuentre sin cambios.
3. Valida `task_id ∈ {1,2,3}` vía `dispatch.validate_task_id` **antes** de cargar pickles.
4. Despacha al runner de la tarea. Task 1 usa el baseline conservado; Tasks 2/3 abortan con error controlado.

Solo se cargan pickles/joblib desde `cfg.paths.submodels_dir` (`/opt/ml/model` en GC); nunca desde rutas derivadas del input del usuario.

## Docker (GC)

```bash
docker build -f deploy/agent/Dockerfile -t chimera-agent .
docker run --rm -v $PWD/deploy/agent/tests/fixtures:/input -v $PWD/output:/output chimera-agent
```

El `entrypoint.sh` inicia `ollama serve`, espera health-check (`curl`), hace pull solo si el modelo no está bakeado, y lanza `inference.py`.

## Desarrollo local

```bash
pip install -e deploy/agent
pytest -q deploy/agent/tests
```

## Contratos

El contrato JSON de Task 1 se preserva byte-equivalente al baseline en `deploy/tasks/task1/gc-agent/src/chimera_agent_baseline/output/schema.py` (Task1Output). La migración no lo reemplaza por un esquema incompatible.

## Añadir Task 2/3

1. Copiar su código a `src/chimera_agent/tasks/taskN/baseline/`.
2. Añadir `configs/tasks/taskN.yaml` con `agent.task_id` y `paths.submodels_dir` / `mcp.server_module`.
3. Registrar el runner en `dispatch.py` y añadir tests de contrato.
