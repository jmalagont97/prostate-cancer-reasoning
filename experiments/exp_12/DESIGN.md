# Experiment Design: Late Multimodal Fusion — Top-1 Winner per Modality (exp_12)
**Experiment**: experiments/exp_12/ · **Project**: pathology-reasoning · **Date**: 2026-08-17 · **Status**: Approved

---

## 1. Hypothesis

La fusión tardía de las tres modalidades (tabular, MRI, texto) promediando sus probabilidades de los modelos ganadores individuales mejora el rendimiento frente a cualquier modalidad individual, siempre que los errores sean parcialmente independientes. La combinación tabular + texto es la candidata más fuerte.

## 2. Modalidades y modelos congelados

Cada modalidad utiliza exactamente el modelo ganador previamente seleccionado. No hay búsqueda de hiperparámetros dentro de exp_12.

### Tabular (exp_5)
- **Variables**: 21 tras tau_0.60 (intersección MCCV).
- **Pruning**: Spearman correlation pruning, clustering complete linkage, representantes por regla de variables esenciales.
- **Preprocesamiento**: zero-fill + indicadores de ausencia, one-hot, MinMax.
- **KNN**: k=1, cosine, uniform, confidence_weighted (fuzzy v2).

### MRI (exp_9)
- **Embedding**: 1024 dimensiones de images.csv.
- **PCA**: n_components=1 (fit per fold en entrenamiento).
- **KNN**: k=1, euclidean, distance, confidence_weighted (fuzzy v2).

### Texto (exp_10 corregido)
- **Preprocesamiento textual**: lowercase, remove special chars, remove numeric tokens (digits + written numbers), remove stopwords (protect negations), lemmatize (spaCy en_core_web_sm v3.8.0).
- **TF-IDF**: max_features=2000, ngram_range=(1,1), norm=l2.
- **KNN**: k=3, cosine, distance, confidence_weighted (fuzzy v2).

## 3. Protocolo MCCV (selección)

50 splits estratificados, 70/18. Para cada split:
1. Ajustar cada modelo solo con el conjunto de entrenamiento.
2. Generar probabilidad para los 18 casos de validación.
3. Para las 7 combinaciones no vacías de modalidades, promediar sus probabilidades.
4. Calcular métricas sobre las 18 predicciones de validación.

Combinaciones:
- T, M, X, T+M, T+X, M+X, T+M+X

## 4. Regla de selección

```text
F1_macro → brier_score (↓) → F1_yes → balanced_accuracy → MCC
```

La selección se basa exclusivamente en las métricas MCCV.

## 5. Protocolo LOO (final)

88 folds con hiperparámetros congelados. Ajuste desde cero de cada modalidad incluida en la combinación ganadora, en cada fold. La intersección de variables tabulares se computa a partir de las 50 selecciones MCCV (misma convención de exp_5).

## 6. Métricas por combinación

- F1_macro (selección), brier_score, F1_yes, balanced_accuracy, MCC, sensibilidad, especificidad, PR-AUC, ROC-AUC, ECE, matriz de confusión.
- Diversidad: correlación probabilística entre modalidades, tasa de desacuerdo de etiquetas.

## 7. Artefactos

- summary_selection.json, config_log.json, fusion_report.json.
- Por combinación ganadora: oof_predictions_mccv.csv, oof_predictions_loo.csv, confusion_matrices.json, validation_report.json.
- Por modalidad: oof_predictions_mccv.csv y oof_predictions_loo.csv individuales.

## 8. Validaciones

- Cohorte usable_labeled 88 casos.
- 50 splits MCCV, cada uno 70/18 y ambas clases presentes.
- 88 folds LOO, exactamente 1 caso por fold.
- Sin fuga de datos: pruning, PCA y TF-IDF ajustados solo en entrenamiento.
- Probabilidades en [0, 1], promediar solo probabilidades (umbral 0.5).
