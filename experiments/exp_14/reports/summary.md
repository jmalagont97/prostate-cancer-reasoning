# Experiment Report: Validación end-to-end del agente Task 3 sobre la cohorte completa
**Experiment**: experiments/exp_14/  
**Project**: prostate-cancer-reasoning  
**Report date**: 2026-08-30  
**Plan date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete

---

## 1. Summary

El agente Task 3 procesó los 75 casos disponibles usando Docker, Ollama, GPU y `gemma4:e2b`. Los 75 casos generaron correctamente los dos archivos JSON exigidos, todos fueron dirigidos a `output/task3/` y no se creó ningún output de Task 1. Sobre los mismos 75 casos anotados, la evaluación exploratoria obtuvo Harrell C-index = 0.9173, `event_score` medio = 0.9067 y `time_score` medio = 0.5920. Estas métricas son in-sample y potencialmente optimistas porque el modelo productivo fue entrenado con esta misma cohorte.

---

## 2. Hypothesis & Verdict

**Hypothesis (from plan):** El agente Task 3 puede procesar de forma reproducible todos los casos disponibles, generar los dos archivos exigidos por el contrato y producir predicciones Cox válidas acompañadas de razonamiento clínico generado por Gemma 4 E2B, sin que Gemma altere `months_to_recurrence` ni `event`.

**Verdict:** ✅ Supported for functional validation; generalization remains untested.

**Evidence:** 75/75 casos produjeron outcome y reasoning válidos, con 0 errores de contrato. La ruta final fue exclusivamente `output/task3/`; el smoke test corregido conservó `months_to_recurrence=14.9` y `event=0`.

---

## 3. Experimental Setup (as run)

- **Dataset**: 75 directorios de `data/raw/train_release/task3/`; todos disponían de ground truth en esta cohorte.
- **Model**: agente Task 3 con tool Cox determinista y reasoning separado por `gemma4:e2b` vía Ollama.
- **Execution**: Docker `chimera-agent:task3-cox`, `--gpus all`, `--network none`, entrada read-only y outputs en `experiments/exp_14/results/agent_all_cases/outputs/`.
- **Hardware**: NVIDIA GeForce RTX 3060; ejecución persistente en `tmux:0`.
- **Deviations from plan**: se detectó y corrigió antes de la ejecución final un defecto de autodetección que podía usar el namespace Task 1. La imagen fue reconstruida y el lote final se ejecutó después de la corrección. El plan esperaba posibles casos no anotados, pero el directorio disponible contenía 75 casos y los 75 estaban anotados.

---

## 4. Version & Artifacts

| Item | Value |
|---|---|
| Git commit base | `d44eb25e607053199b06593ea979d2ca4d8557e2` |
| Git state | Dirty; paths modificados registrados en `results/*/version.txt` |
| Docker image | `sha256:67d383ee34bbf958e7460d514c42c4d29946333f28178bcc5b11a076f4b6a7b5` |
| Model | `gemma4:e2b` |
| Metrics package | `chimera-metrics 0.1.0` |

Artifacts: `agent_all_cases/outputs/task3/`, `agent_all_cases/manifest.json`, `agent_all_cases/version.txt`, `agent_all_cases/logs/docker.log`, `annotated_metrics/metrics.json` y `annotated_metrics/version.txt`.

---

## 5. Results

### 5.1 Coverage and contract

| Check | Result |
|---|---:|
| Cases discovered | 75 |
| Valid outcomes | 75/75 |
| Valid reasonings | 75/75 |
| Contract errors | 0 |
| Output namespace incorrect (`task1`) | No |

### 5.2 Annotated Task 3 metrics

Calculadas exclusivamente mediante `chimera_metrics` sobre los 75 ground truths.

| Metric | Value |
|---|---:|
| Mean `event_score` | 0.9067 |
| Mean censoring-aware `time_score` | 0.5920 |
| Harrell C-index | 0.9173 |
| IPCW AUC @ 12 months | 0.9670 |
| IPCW AUC @ 24 months | 0.9250 |
| IPCW AUC @ 36 months | 0.9223 |
| IPCW AUC @ 60 months | 0.9444 |

Ground truth events: 19 event / 56 censored. Predicted events: 20 event / 55 censored.

---

## 6. Statistical Analysis

No significance test or confidence interval was applied. This is a single exploratory in-sample run, not a comparison between independent models or held-out folds; inferential statistics would give a misleading impression of external validity.

---

## 7. Comparison to Expected Results

| Expected | Observed | Match? |
|---|---|---|
| Todos los casos procesados | 75/75 | ✅ |
| Dos JSON válidos por caso | 150/150 | ✅ |
| Outputs bajo Task 3 | Sí; `task1` no creado | ✅ |
| Predicciones Cox preservadas | Verificado en smoke test y contrato | ✅ |
| Métricas con `chimera_metrics` | C-index, scores y AUC IPCW | ✅ |

---

## 8. Missing Data & Caveats

- No se encontraron casos no anotados en el directorio disponible; por tanto, la cobertura funcional se probó sobre 75 casos, todos anotados.
- Las métricas son in-sample: el modelo productivo fue entrenado sobre la misma cohorte.
- El reasoning de Gemma se validó como JSON independiente y no reemplaza los valores oficiales Cox.
- La ejecución inicial interrumpida produjo outputs temporales en `/tmp` bajo `task1`; no contaminó `data/raw`, `deploy/tasks/task1` ni los resultados finales. La campaña final se ejecutó con la ruta corregida.

---

## 9. Conclusions & Next Steps

- **Establecido:** el flujo Docker Task 3 procesa la cohorte completa, genera el contrato oficial y mantiene la separación Cox→outcome / Gemma→reasoning.
- **Establecido:** la corrección de rutas evita que un batch Task 3 sea despachado al namespace Task 1.
- **Incierto:** el rendimiento fuera de muestra y en los casos oficiales no vistos del challenge.
- **Siguiente paso:** repetir la evaluación sobre una cohorte externa o los casos oficiales de evaluación, manteniendo la misma imagen y protocolo.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Case manifest | ✅ |
| GPU/model/image recorded | ✅ |
| Code/version state recorded | ✅ |
| Docker logs preserved | ✅ |
| Raw outputs preserved | ✅ |
| `chimera_metrics` version recorded | ✅ |
| Seeds | N/A; deterministic inference configuration |
| Checkpoints | N/A; production model baked into image |
| Experiment tracker | N/A |
