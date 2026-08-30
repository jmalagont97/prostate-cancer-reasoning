# Implementation Plan: Ejecución persistente del modelo ganador Task 3 bajo MCCV-30

**Experiment**: `experiments/exp_13/`  
**Project**: `prostate-cancer-reasoning`  
**Author**: Juan Sebastian Malagón Torres  
**Date**: 2026-08-29  
**Status**: Implemented and verified; rerun with event prediction pending

---

## 1. Objetivo de implementación

Implementar el ejecutor de `exp_13` para reproducir la estrategia de `origin/rdacostav_dev` sobre los datos procesados canónicos y evaluar las condiciones definidas en `DESIGN.md` con los 30 folds de `exp_12`.

La ejecución será persistente en una sesión detached denominada `tmux:0`. El programa emitirá mensajes de progreso con `flush=True` y escribirá simultáneamente un log global y logs por condición, de forma que el avance pueda consultarse en tiempo real mediante `tmux capture-pane` o leyendo los archivos de log.

## 2. Entradas canónicas

- `data/processed/task3/clinical_pathology_features.csv`
- `data/processed/task3/survival_labels.csv`
- `data/processed/task3/embeddings.npz`
- `data/processed/task3/mccv_splits_30.csv`
- `data/processed/task3/schema.json`
- `utils/pkg/chimera_metrics/`

La implementación no leerá `data/raw/` ni dependerá de archivos de `origin/rdacostav_dev` durante la ejecución.

## 3. Componentes a implementar

### 3.1 Cargador y validación de datos

- Cargar las tres tablas/artefactos procesados.
- Verificar unicidad y alineación completa por `case_id`.
- Verificar 75 casos, 19 eventos, 30 columnas `split_0`...`split_29` y particiones 60/15.
- Verificar que los embeddings de prostatectomía y sus conteos sean válidos.
- Rechazar silenciosamente cualquier caso o fold inconsistente.

### 3.2 Transformaciones fold-safe

Para cada fold y condición:

- separar entrenamiento y validación mediante la matriz MCCV existente;
- imputar y escalar usando únicamente entrenamiento;
- hacer pooling promedio de los embeddings de prostatectomía;
- ajustar PCA-5 únicamente sobre entrenamiento en las condiciones multimodales;
- transformar la validación con los objetos ajustados en entrenamiento.

No habrá PCA, imputación ni escalado globales.

### 3.3 Modelo

Implementar la estrategia registrada en `rdacostav_dev`:

- modelo Cox proporcional con regularización Ridge `lambda=0.3`;
- representación clínica/patológica concatenada con prostatectomía PCA-5;
- predicción de riesgo para ranking;
- predicción de meses mediante la regla temporal congelada `p=0.90`;
- predicción de `event` mediante `S(60 meses | x) < 0.7`;
- condición de control con estimación temporal media.

Si una dependencia externa del modelo original no está instalada, se documentará el bloqueo y no se sustituirá silenciosamente por otro estimador.

### 3.4 Evaluación centralizada

El ejecutor importará el paquete local mediante `PYTHONPATH=utils/pkg/chimera_metrics` y usará exclusivamente:

```python
from chimera_metrics import evaluate_task3_cohort
```

Para cada fold se enviarán las listas `gt_list`, `pred_list` y `case_ids` a `evaluate_task3_cohort`. Las métricas se copiarán desde el objeto retornado, sin recalcular C-index, AUC, score de evento ni score temporal localmente.

El `event` se calculará a partir de la curva de supervivencia individual del Cox ajustado en cada fold. El horizonte (`60` meses) y el umbral de supervivencia (`0.7`) son políticas congeladas; no se seleccionarán usando la validación.

## 4. Salidas

Por condición:

```text
experiments/exp_13/results/<condition>/
├── fold_metrics.csv
├── predictions.csv
├── summary.json
└── version.txt
```

A nivel global:

```text
experiments/exp_13/results/
├── exp_13_live.log
├── execution_status.json
└── manifest.json
```

Cada fold deberá dejar una línea de progreso con condición, fold, casos, eventos y C-index, por ejemplo:

```text
[2026-08-29T..] condition=prostatectomy_pca5_p90 fold=07/30 status=ok c_index=...
```

El estado se actualizará después de cada fold para permitir recuperación y auditoría.

## 5. Persistencia tmux y tiempo real

- Crear la sesión si `tmux:0` no existe:
  ```bash
  tmux new-session -d -s 0
  ```
- Ejecutar el proceso desde la raíz del repositorio dentro de esa sesión.
- Usar salida Python no bufferizada (`python -u`) y `flush=True`.
- Redirigir stdout y stderr a `experiments/exp_13/results/execution.log` sin ocultar la salida del pane.
- Verificar externamente que la sesión existe, que el proceso está activo y que `execution.log` aumenta durante la ejecución.
- No usar un proceso foreground que dependa de la interfaz Hermes.

## 6. Reproducibilidad

Antes del lanzamiento se registrarán:

- commit actual y estado del working tree;
- hash de cada entrada procesada;
- hash de `mccv_splits_30.csv`;
- configuración completa de condiciones;
- versión de Python y dependencias relevantes;
- identificador de la sesión tmux.

La presencia de cambios no comprometidos se marcará explícitamente como advertencia, sin bloquear la ejecución.

## 7. Verificación de aceptación

La implementación se considerará ejecutada correctamente cuando:

- las cuatro condiciones completen 30 folds cada una;
- cada fold utilice exactamente 60 casos de entrenamiento y 15 de validación;
- exista una predicción held-out por caso y fold;
- `evaluate_task3_cohort` sea la única fuente de métricas;
- estén presentes `fold_metrics.csv`, `predictions.csv`, `summary.json` y `version.txt` por condición;
- el log muestre progreso en tiempo real durante la ejecución;
- `execution_status.json` indique finalización por condición;
- no se modifique `data/raw/`;
- el proceso finalice dentro de `tmux:0` con código de salida verificable.

## 8. Siguiente paso

Tras aprobar este plan de implementación se crearán los scripts, se lanzará la ejecución persistente en `tmux:0` y se monitorizarán los logs hasta completar los 120 entrenamientos/evaluaciones de fold.
