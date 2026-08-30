# Experiment Design: MCCV-30 estratificado para predicción de recurrencia bioquímica en Task 3

**Experiment**: `experiments/exp_12/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres  
**Status**: In Progress

---

## 1. Hipótesis y objetivo

**Hipótesis falsable:** un protocolo determinista de Monte Carlo Cross-Validation con 30 particiones, estratificado conjuntamente por estado de recurrencia y distribución de seguimiento, permitirá comparar de forma más estable y reproducible las variantes de un modelo de supervivencia para Task 3 que una única partición aleatoria; además, la incorporación de representaciones de prostatectomía mejorará el C-index frente al baseline clínico-patológico sin introducir fuga de información.

El objetivo primario es generar el artefacto de particiones y establecer un protocolo de evaluación repetible. El diseño no presupone que la hipótesis se confirme.

## 2. Conjunto de datos y unidad de análisis

- **Tarea**: CHIMERA-Agent Task 3, predicción del tiempo hasta recurrencia bioquímica o último seguimiento después de prostatectomía.
- **Datos**: `data/processed/task3/clinical_pathology_features.csv`, `survival_labels.csv` y `embeddings.npz`.
- **Cohorte**: 75 casos completos, 19 eventos observados y 56 observaciones censuradas, sujetos a verificación por el manifiesto de `exp_11`.
- **Unidad de análisis**: un paciente/caso identificado por `case_id`; ninguna modalidad puede separar un mismo caso entre entrenamiento y validación.
- **Etiqueta de supervivencia**: `months_to_recurrence` como tiempo observado y `event` como indicador binario, donde `1` representa recurrencia y `0` censura.
- **Preprocesamiento previo**: únicamente extracción, reconciliación por `case_id` y validación. No se usarán PCA, imputación, normalización aprendida ni selección de variables ajustada a toda la cohorte.

## 3. Diseño MCCV-30

Se generarán **30 particiones Monte Carlo fijas**, `split_0` a `split_29`, usando semillas deterministas registradas en el artefacto. Cada partición tendrá la misma proporción que el protocolo de Task 1:

- **80% entrenamiento**: 60 casos.
- **20% validación**: 15 casos.
- **Sin casos `unlabeled`**: todos los 75 casos de Task 3 tienen etiqueta de supervivencia para este protocolo.

La estratificación se realizará sobre la etiqueta de supervivencia, no sobre las características de entrada del modelo:

1. Preservar en cada validación la presencia de eventos y censuras.
2. Mantener aproximadamente la tasa global de eventos (`19/75 = 25.3%`), con objetivo de 3–5 eventos y al menos 3 censuras por fold de validación.
3. Equilibrar la distribución de `months_to_recurrence` mediante bins de tiempo definidos de forma reproducible y evaluados separadamente para eventos y censuras. Los bins solo controlan la composición de los folds; no se usan para entrenar el modelo.
4. Rechazar particiones sin variación suficiente para calcular C-index o con una representación temporal claramente desbalanceada.
5. Seleccionar cada partición mediante una función de balance predefinida y no mediante el resultado de ningún modelo.

La validación de cada partición debe confirmar que las 15 filas `val` y las 60 filas `train` son complementarias y que todos los `case_id` aparecen exactamente una vez en cada fold.

## 4. Artefacto de particiones esperado

El resultado canónico será:

```text
data/processed/task3/mccv_splits_30.csv
```

Con exactamente 75 filas y 31 columnas:

```text
case_id,split_0,split_1,...,split_29
```

Cada celda de las columnas de split tendrá únicamente `train` o `val`. Para cada `split_b` se debe cumplir:

```text
count(train) = 60
count(val)   = 15
count(train) + count(val) = 75
```

Los diagnósticos de balance, las semillas, el hash del dataset de entrada y la versión del código se almacenarán bajo `experiments/exp_12/results/`. El CSV será determinista: la misma entrada y configuración deben producir byte-for-byte el mismo archivo.

## 5. Condiciones de modelado que se compararán posteriormente

Estas condiciones forman parte del diseño de evaluación, pero su implementación se definirá en un plan separado después de aprobar este diseño.

| Condición | Predictores | Propósito |
|---|---|---|
| `clinical_pathology` | Variables clínicas y patológicas estructuradas | Baseline interpretable y de bajo riesgo de sobreajuste |
| `prostatectomy_only` | Baseline + embeddings de prostatectomía | Probar la contribución de la modalidad priorizada por `rdacostav_dev` |
| `mri_prostatectomy` | Baseline + MRI + prostatectomía | Medir el valor incremental de MRI junto con patología |
| `all_modalities` | Baseline + MRI + biopsia + prostatectomía | Evaluar la fusión multimodal completa y posibles modalidades redundantes |

En cada fold, cualquier imputación, escalado, pooling aprendido, reducción dimensional, selección de variables y ajuste del modelo se realizará **exclusivamente con los 60 casos de entrenamiento**. La validación solo se transformará usando los parámetros ajustados en entrenamiento.

## 6. Métricas y protocolo de evaluación

- **Métrica primaria**: Harrell's concordance index (C-index) sobre cada conjunto de validación, respetando la censura.
- **Métricas secundarias**: concordancia del evento, proximidad temporal consciente de censura y AUC dependiente del tiempo a 12, 24, 36 y 60 meses cuando exista soporte suficiente en el fold.
- **Resumen principal**: media, desviación estándar, mediana e intervalo de confianza del 95% de las 30 puntuaciones por condición.
- **Comparación entre condiciones**: comparación pareada por `split_b`, usando diferencias de C-index y Wilcoxon signed-rank como análisis no paramétrico; no se tratarán folds como observaciones independientes para ignorar el emparejamiento.
- **Estimación complementaria**: agregación de predicciones de validación por caso, reportando cuántas veces fue validado cada `case_id`; esta agregación no reemplaza el resumen por fold.
- **Evaluación oficial**: la salida de inferencia debe conservar ambos campos requeridos, `months_to_recurrence` y `event`, además de `case_id` según el contrato oficial de Task 3.

La evaluación debe usar la implementación centralizada del proyecto (`chimera_metrics`) y no código de métricas duplicado dentro de `experiments/`.

## 7. Reglas de decisión

- **Confirmación de estabilidad del MCCV**: todas las particiones cumplen el tamaño 60/15, contienen eventos y censuras suficientes, y el diagnóstico de balance no muestra folds atípicos.
- **Apoyo a la hipótesis multimodal**: `prostatectomy_only` supera a `clinical_pathology` en la media del C-index y la diferencia pareada es consistente en la mayoría de folds, sin depender de un único fold extremo.
- **Resultado no concluyente**: la mejora media es pequeña, el intervalo incluye cero o la variabilidad entre folds domina el efecto; no se seleccionará una modalidad solo por el mejor fold.
- **Rechazo metodológico**: cualquier fuga de información, ajuste de PCA/imputación fuera del entrenamiento, desalineación de `case_id` o ausencia de `event` invalida el resultado y obliga a repetir el protocolo.
- **Selección final**: ningún hiperparámetro se escogerá observando el conjunto de validación y luego se reportará como confirmatorio. Si se requiere ajuste, deberá declararse un procedimiento anidado o una fase exploratoria independiente.

## 8. Riesgos y mitigaciones

- **Pocos eventos (19)**: el C-index por fold puede presentar alta varianza. Se reportarán los 30 folds, intervalos de confianza y estabilidad de la distribución, no solo la media.
- **Dependencia entre folds**: MCCV produce solapamiento entre conjuntos de entrenamiento y validación. Se usarán comparaciones pareadas por split y se evitará interpretar los 30 valores como muestras independientes.
- **Sobreajuste por selección de hiperparámetros**: se fijarán los hiperparámetros antes del análisis confirmatorio o se empleará validación anidada.
- **Fuga por transformaciones multimodales**: PCA, imputación, escalado y pooling ajustable se entrenarán dentro de cada fold; las etiquetas nunca entrarán en la matriz de características.
- **Modalidades con bolsas de tamaño variable**: se conservarán las máscaras y conteos, verificando que el caso y la modalidad permanezcan alineados.
- **Métrica no definida en un fold**: el generador rechazará particiones con composición insuficiente y el evaluador registrará explícitamente cualquier métrica no estimable, sin reemplazarla silenciosamente por cero.

## 9. Reproducibilidad y criterios de aceptación

- [x] Se fija una semilla explícita por fold y se registra la configuración completa.
- [x] Se guarda `mccv_splits_30.csv` bajo `data/processed/task3/`.
- [x] Se verifica la forma 75 × 31 y los conteos 60/15 en las 30 columnas.
- [x] Se verifica el rango de eventos/censuras por fold y la distribución temporal.
- [x] Se registra el hash de los archivos de entrada y la versión exacta del código.
- [x] Una segunda ejecución produce el mismo CSV y los mismos diagnósticos.
- [x] Se confirma que no se modificó `data/raw/`.
- [ ] Las métricas provienen de `chimera_metrics` y las predicciones conservan `case_id`, `months_to_recurrence` y `event`.
- [ ] El estado de Git se registra junto con cada ejecución; los resultados con cambios sin commit se marcan como no reproducibles desde un SHA limpio.

## 10. Próximo paso

La generación MCCV-30 fue ejecutada y verificada. El siguiente paso es implementar el ejecutor de las condiciones de modelado y evaluar las predicciones usando exclusivamente las particiones registradas.
