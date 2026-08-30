# Implementation Plan: MCCV-30 estratificado para supervivencia en Task 3 (`exp_12`)

**Experiment**: `experiments/exp_12/`  
**Project**: `prostate-cancer-reasoning`  
**Author**: Juan Sebastian Malagón Torres  
**Date**: 2026-08-29  
**Status**: Implemented and verified

---

## 1. Alcance

Implementar un generador reproducible de 30 particiones MCCV para los 75 casos de Task 3. El generador leerá las etiquetas procesadas, construirá particiones 80/20 y escribirá el artefacto canónico en `data/processed/task3/mccv_splits_30.csv`.

No entrenará modelos, no calculará C-index y no ajustará PCA, imputación, escalado ni ninguna transformación de predictores. La evaluación de modelos será una fase posterior y utilizará `chimera_metrics`.

## 2. Entradas, salidas y dependencias

**Entrada:**

```text
data/processed/task3/survival_labels.csv
```

Columnas requeridas: `case_id`, `months_to_recurrence`, `event`.

**Salidas:**

```text
data/processed/task3/mccv_splits_30.csv
experiments/exp_12/results/mccv_diagnostics.json
experiments/exp_12/results/version.txt
```

El script usará únicamente la biblioteca estándar, NumPy y pandas ya disponibles en el proyecto, y resolverá las rutas por defecto desde la raíz del repositorio.

## 3. Generación determinista

Para cada `split_b`, con `b = 0,...,29`, se usará la semilla explícita `42000 + 1000*b`. Se generarán candidatos con 4 eventos y 11 censuras en validación (15 casos); los 60 restantes serán entrenamiento. Se evaluarán hasta 2000 candidatos deterministas por fold y se conservará el de menor función de pérdida de balance.

La función de pérdida combinará:

1. desviación de la tasa de eventos entre train y validation;
2. discrepancia de frecuencias entre bins de seguimiento;
3. diferencia entre medias y cuantiles de `months_to_recurrence` entre ambos subconjuntos.

Los bins temporales se calcularán una sola vez a partir de los tiempos de entrada para controlar la composición de los folds. No se usarán para entrenar ningún modelo.

## 4. Validaciones obligatorias

Antes de escribir el CSV se comprobará:

- 75 `case_id` únicos y ordenados de forma lexicográfica;
- exactamente 30 columnas `split_0`–`split_29`;
- exactamente 60 valores `train` y 15 valores `val` por columna;
- exactamente 4 eventos y 11 censuras en cada validación;
- complementareidad train/validation en cada fold;
- todos los tiempos numéricos y no negativos;
- todos los eventos en `{0, 1}`;
- ninguna etiqueta o característica clínica se incorpora al archivo de splits;
- ausencia de modificaciones en `data/raw/`.

Los diagnósticos registrarán la semilla, el número de eventos/censuras, la media temporal, los bins temporales, la pérdida de balance y los identificadores de validación de cada fold.

## 5. Verificación de determinismo

Se ejecutará el generador dos veces con la misma entrada y configuración. Se compararán el SHA-256 y el contenido lógico de `mccv_splits_30.csv`; cualquier diferencia invalidará el resultado. También se validará que el script no dependa del directorio de trabajo actual.

## 6. Criterios de aceptación

La implementación se considerará correcta cuando:

1. genere `data/processed/task3/mccv_splits_30.csv` con forma `(75, 31)`;
2. cada fold contenga 60 casos de entrenamiento, 15 de validación, 4 eventos y 11 censuras en validation;
3. los diagnósticos cubran los 30 folds;
4. dos ejecuciones produzcan el mismo hash;
5. el artefacto sea compatible con el formato matricial de Task 1, adaptado de `pid` a `case_id` y sin categoría `unlabeled`;
6. `python -m py_compile`, la validación del script y `git diff --check` terminen correctamente.

## 7. Secuencia de ejecución

```text
1. Ejecutar el modo de generación.
2. Ejecutar el modo de validación sobre el CSV generado.
3. Repetir la generación y comparar hashes.
4. Inspeccionar los diagnósticos de los 30 folds.
5. Actualizar `DESIGN.md`, `INDEX.md`, `.logbook.md` y `.discussion.md` solo con resultados observados.
```

## 8. Límites

Este plan no incluye el entrenamiento del modelo Cox, selección de hiperparámetros, ablation study, agregación de predicciones ni cálculo del C-index. Esos pasos requerirán el ejecutor de modelado posterior y deberán usar exclusivamente los folds generados y transformaciones ajustadas dentro de cada conjunto de entrenamiento.
