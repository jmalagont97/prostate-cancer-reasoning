# Experiment Design: Replicación MCCV-30 del modelo ganador de Task 3

**Experiment**: `experiments/exp_13/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Implemented and verified; rerun with event prediction pending

---

## 1. Hipótesis

La estrategia ganadora identificada en `origin/rdacostav_dev` —Cox proporcional con regularización Ridge, pooling de embeddings de prostatectomía, PCA de cinco componentes y predicción temporal basada en el percentil 0.90— conservará una ventaja de discriminación frente al baseline clínico-patológico cuando se evalúe sin fuga mediante el MCCV-30 de `exp_12`.

La afirmación se considerará confirmada únicamente si la condición `prostatectomy_pca5_p90` mejora el C-index del baseline `clinical_pathology` en la diferencia pareada predefinida y no depende de un único fold favorable.

> **Referencia no confirmatoria:** el C-index LOOCV=0.8765 reportado en `rdacostav_dev` se tratará como referencia histórica, no como resultado comparable ni como umbral de éxito, porque el protocolo LOOCV y el MCCV-30 son distintos.

## 2. Dataset y procedencia

- **Cohorte:** 75 casos de Task 3, con 19 eventos y 56 censuras.
- **Características clínicas/patológicas:** `data/processed/task3/clinical_pathology_features.csv`.
- **Etiquetas:** `data/processed/task3/survival_labels.csv`, con `case_id`, `months_to_recurrence` y `event`.
- **Embeddings:** `data/processed/task3/embeddings.npz`; la modalidad principal será `prostatectomy`, con forma `(75, 3, 960)` y pooling promedio por caso.
- **Particiones:** `data/processed/task3/mccv_splits_30.csv`, generada y validada por `exp_12`.
- **Esquema:** `data/processed/task3/schema.json`.
- **Fuente de la estrategia:** `origin/rdacostav_dev:task3_solution/`, usada como referencia conceptual y de hiperparámetros, no como fuente de datos en tiempo de ejecución.

La implementación deberá comprobar que todos los `case_id` están alineados entre características, etiquetas, embeddings y particiones antes de iniciar cualquier fold.

## 3. Protocolo MCCV-30

Se reutilizará exactamente la matriz de particiones de `exp_12`; no se regenerarán folds durante la evaluación del modelo.

- 30 folds deterministas: `split_0` a `split_29`.
- Por fold: 60 casos de entrenamiento y 15 de validación.
- Cada validación: 4 eventos y 11 censuras.
- La columna `case_id` será la única clave de unión.
- Las predicciones de validación se conservarán fold por fold y se concatenarán únicamente para los resúmenes globales.

Para cada fold, toda transformación aprendida —imputación, escalado, PCA y estimación del modelo— se ajustará únicamente con los 60 casos de entrenamiento. La validación se utilizará exclusivamente para producir predicciones y métricas.

## 4. Estrategia del modelo de referencia

La condición confirmatoria conservará la estrategia de `rdacostav_dev`:

1. extraer las representaciones de prostatectomía;
2. aplicar pooling promedio sobre los tres vectores disponibles por caso;
3. ajustar PCA con `n_components=5` dentro del entrenamiento de cada fold;
4. combinar la representación PCA con las variables clínicas y patológicas preprocesadas;
5. ajustar un modelo Cox proporcional con regularización Ridge `lambda=0.3`;
6. producir el riesgo/puntuación de ordenación del modelo;
7. producir `months_to_recurrence` mediante el percentil temporal predefinido `p=0.90`.

Los hiperparámetros `n_components=5`, `lambda=0.3` y `p=0.90` quedan congelados antes de observar los resultados MCCV-30. No se realizará una búsqueda de percentiles o de regularización sobre los folds de validación.

## 5. Condiciones experimentales

| Condición | Predictores y procedimiento | Función |
|---|---|---|
| `clinical_pathology` | Variables clínicas y patológicas, sin embeddings | Baseline clínico-patológico |
| `prostatectomy_pca5` | Clínicas/patología + prostatectomía pooled + PCA-5; Cox Ridge `lambda=0.3` | Aislar el aporte de la modalidad principal |
| `prostatectomy_pca5_p90` | Igual que la anterior, con estimación temporal `p=0.90` | **Condición confirmatoria del modelo ganador** |
| `prostatectomy_pca5_mean` | Igual que la anterior, usando la estimación temporal media predefinida | Control de la regla de estimación temporal |

Todos los modelos se entrenarán con los mismos 30 folds, la misma cohorte y el mismo protocolo de transformaciones para que las diferencias sean pareadas.

## 6. Evaluación centralizada

Toda métrica deberá provenir de `utils/pkg/chimera_metrics`; no se permitirá implementar métricas paralelas dentro de `exp_13`.

La interfaz principal será:

```python
from chimera_metrics import evaluate_task3_cohort
```

La evaluación de cada condición deberá pasar al paquete las predicciones y las etiquetas correspondientes al fold, conservando como mínimo:

```text
case_id
months_to_recurrence
event
```

Métricas:

- **Primaria:** Harrell C-index, obtenido de `evaluate_task3_cohort`.
- **Secundarias:** `mean_event_score`, `mean_time_score` y AUC dependiente del tiempo en 12, 24, 36 y 60 meses, únicamente cuando el paquete las considere estimables.
- **Agregadas:** media, mediana, desviación estándar e intervalo de confianza del 95% sobre los 30 resultados por fold.
- **Comparación:** diferencia pareada por fold contra `clinical_pathology`, proporción de folds favorables y prueba pareada no paramétrica como análisis secundario.

La salida binaria `event` se derivará de la curva de supervivencia de Cox, siguiendo la política actualizada de `rdacostav_dev`: horizonte de 60 meses y regla `event=1` si `S(60|x) < 0.7`, sin ajustar el umbral con la validación. El C-index y las métricas secundarias seguirán siendo calculados exclusivamente por `chimera_metrics`.

## 7. Artefactos y trazabilidad

Los resultados se guardarán en:

```text
experiments/exp_13/results/<condition>/
```

Cada condición deberá incluir, como mínimo:

```text
fold_metrics.csv          # una fila por fold y métricas del paquete
predictions.csv           # predicciones held-out con case_id y fold
summary.json              # agregados y estado de estimabilidad
version.txt               # estado exacto del código usado
```

También se conservará un manifiesto global con:

- hash de `mccv_splits_30.csv`;
- hash o versión de los artefactos procesados;
- commit de código y estado del working tree;
- configuración congelada de cada condición;
- número de casos, eventos y censuras por fold.

## 8. Reglas de decisión

- **Confirmación principal:** `prostatectomy_pca5_p90` supera a `clinical_pathology` en el C-index medio y en la estimación agrupada, con diferencia pareada positiva y sin que la mejora esté concentrada en un único fold.
- **Resultado inconcluso:** la ventaja aparece en el promedio, pero el intervalo de confianza incluye cero, el análisis pareado no es consistente o la métrica no es estimable en suficientes folds.
- **Refutación:** el baseline obtiene igual o mejor C-index, o la condición multimodal empeora consistentemente.
- El valor LOOCV=0.8765 de `rdacostav_dev` no se usará para seleccionar hiperparámetros ni para declarar éxito estadístico.

## 9. Riesgos y mitigaciones

- ⚠️ **Fuga de información por PCA/imputación global:** ajustar cada transformación dentro del train de cada fold y registrar el alcance de ajuste.
- ⚠️ **Sobreajuste de `p=0.90`:** tratarlo como hiperparámetro congelado por replicación; no volver a seleccionarlo con MCCV-30.
- ⚠️ **Pocos eventos:** 19 eventos producen incertidumbre elevada; reportar distribución completa de folds e intervalos de confianza, no solo el promedio.
- ⚠️ **Incompatibilidad de `event`:** el modelo original no documenta necesariamente una predicción binaria; no fabricar etiquetas y marcar las métricas no estimables.
- ⚠️ **Desalineación de modalidades:** verificar `case_id`, máscaras `*_counts` y número de vectores antes del entrenamiento.
- ⚠️ **Dependencia indebida de otra implementación:** importar datos y métricas desde las rutas canónicas actuales; usar `rdacostav_dev` solo como referencia de estrategia.

## 10. Criterios de reproducibilidad

- [ ] Se reutiliza sin cambios `data/processed/task3/mccv_splits_30.csv`.
- [ ] Se verifican 75 casos, 30 folds, 60/15 por fold y 4 eventos por validación.
- [ ] Se fijan y registran las semillas y la configuración de cada condición.
- [ ] PCA, imputación y escalado se ajustan dentro de cada fold.
- [ ] Todas las métricas proceden de `utils/pkg/chimera_metrics`.
- [ ] Se guardan predicciones held-out y métricas por fold.
- [ ] Se registra `version.txt` por condición.
- [ ] Se registra la versión/hash de datos procesados y particiones.
- [ ] Se comprueba que `data/raw/` no se modifica.
- [ ] Se documenta el estado del working tree, incluyendo cambios no comprometidos.

## 11. Próximo paso

Revisar y aprobar este diseño. Después se preparará `experiments/exp_13/IMPLEMENTATION.md`, donde se especificarán el adaptador del modelo de `rdacostav_dev`, el cargador de los artefactos procesados y la integración obligatoria con `chimera_metrics`, antes de ejecutar las cuatro condiciones.
