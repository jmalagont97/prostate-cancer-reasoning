# Implementation Plan: Late Multimodal Fusion — exp_12
**Experiment**: experiments/exp_12/ · **Script**: experiments/exp_12/scripts/run_late_fusion_experiment.py · **Runtime**: tmux session 0 · **Date**: 2026-08-17

---

## 1. Overview

Script único y autónomo que reentrena desde cero los tres modelos ganadores de exp_5 (tabular), exp_9 (MRI) y exp_10 corregido (texto), promedia sus probabilidades por combinación de modalidades, evalúa en MCCV y selectiona la mejor combinación. La combinación ganadora se evalúa con LOO.

Coste:
- MCCV: 3 modelos × 50 splits = 150 entrenamientos fold-locales
- LOO: 3 modelos × 88 folds = 264 entrenamientos fold-locales

## 2. File Structure

```text
experiments/exp_12/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_late_fusion_experiment.py
├── results/
│   ├── summary_selection.json
│   ├── config_log.json
│   ├── fusion_report.json
│   ├── tabular_mccv.csv
│   ├── mri_mccv.csv
│   ├── text_mccv.csv
│   ├── <combination>/
│   │   ├── metrics_mccv.json
│   │   ├── metrics_loo.json
│   │   ├── oof_predictions_mccv.csv
│   │   ├── oof_predictions_loo.csv
│   │   ├── diversity_mccv.json
│   │   └── validation_report.json
│   └── tabular_loo.csv (si T en ganadora)
│   └── mri_loo.csv (si M en ganadora)
│   └── text_loo.csv (si X en ganadora)
└── run_output.log
```

## 3. Data Contracts

### Entradas (data/chimera26/preprocessed/task1/)
| Archivo | Uso | Notas |
|---|---|---|
| main_tabular.csv (195 × 28+) | Variables clínicas | case_id + 27 features |
| full_prompt_narrative.csv (195 × 2) | Narrativa clínica | case_id + txt_full_prompt_narrative |
| images.csv (195 × 1025) | Embedding MRI | case_id + mri_emb_0..1023 |
| ground_truth.csv (195 × 27) | Targets | case_id + target_biopsy_decision_binary, target_confidence |
| mccv_loocv_splits.csv (195 × 56) | Particiones fijas | cohort_status, loocv_fold, mccv_split_00..49 |

### Salidas
- Métricas por combinación (MCCV y LOO).
- Probabilidades out-of-fold alineadas por (split/fold, case_id).

## 4. Module Details

### 4.1 Data Loader
- Filtrar a cohort_status == "usable_labeled" (N=88).
- Alinear tabular, imágenes, texto, ground_truth y splits por case_id.

### 4.2 Tabular Pipeline (exp_5)
- Pruning Spearman tau=0.60 fold-local en MCCV.
- Para LOO: intersección de variables seleccionadas en los 50 splits MCCV.
- Preprocesamiento: missing indicators, fillna, OHE, MinMaxScaler fit-on-train-only.
- KNN: k=1, cosine, uniform, confidence_weighted.

### 4.3 MRI Pipeline (exp_9)
- PCA n_components=1, svd_solver='full', fit-on-train-only.
- KNN: k=1, euclidean, distance, confidence_weighted.

### 4.4 Text Pipeline (exp_10 corregido)
- SpaCy en_core_web_sm v3.8.0.
- Preprocesamiento: lowercase, regex remove special chars, remove numeric tokens, protect negations, lemmatize.
- TF-IDF: max_features=2000, norm=l2.
- KNN: k=3, cosine, distance, confidence_weighted.

### 4.5 Fusion
- Combinaciones: ['T','M','X','T+M','T+X','M+X','T+M+X']
- Promedio de probabilidades de las modalidades incluidas.
- Umbral binario: 0.5.

## 5. Evaluation

### MCCV
- 50 splits, 70/18.
- Métricas calculadas sobre 18 casos de validación por split.
- Selección: F1_macro → brier_score → F1_yes → balanced_accuracy → MCC.

### LOO
- 88 folds.
- Solo para la combinación ganadora.
- Reentrenamiento desde cero por fold.
- Métricas calculadas sobre las 88 predicciones out-of-fold.

## 6. Validation Checks (hard)

| Check | Regla |
|---|---|
| Cohorte usable | 88 casos, 54 yes / 34 no |
| MCCV splits | 50, 70/18, ambas clases |
| LOO folds | 88, 1 caso por fold |
| Sin fuga | Pruning, PCA y TF-IDF solo ajustados en train |
| Probabilidades | Todas en [0, 1] |
| Promedio | Solo probabilidades (no etiquetas), umbral 0.5 |

## 7. Execution

```bash
tmux send-keys -t 0 "conda activate histo-DL && python3 experiments/exp_12/scripts/run_late_fusion_experiment.py 2>&1 | tee experiments/exp_12/run_output.log" C-m
```

## 8. Environment

- conda `histo-DL` (Python 3.11.15)
- scikit-learn 1.9.0, pandas 3.0.3, numpy, scipy 1.17.1, spacy en_core_web_sm 3.8.0
