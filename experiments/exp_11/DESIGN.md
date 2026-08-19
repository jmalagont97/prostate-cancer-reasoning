# Experiment Design: TF-IDF + TruncatedSVD + KNN on Clinical Narrative Text
**Experiment**: experiments/exp_11/  
**Project**: pathology-reasoning  
**Date**: 2026-08-17  
**Status**: Draft (Corrected)

## 1. Hypothesis
SVD resuelve la geometría de TF-IDF cuando se conserva el vocabulario completo, mejorando el Macro-F1 MCCV en al menos 0.02 respecto a exp_10.

## 2. Representación
- **Vocabulario**: Completo (`max_features=None`)
- **Reducción**: TruncatedSVD (`random_state=42`, `n_iter=5`, `algorithm="randomized"`)
- **Post-SVD**: L2 normalization
- **Pipeline**: `texto → spaCy → TF-IDF → TruncatedSVD → L2 norm → KNN`
- **Preprocessing**: lowercase, remove special chars (keep hyphens), remove numeric tokens, remove stopwords (protect negations), lemmatize

## 3. Grid
- **n_components**: `None` (sin SVD), `1`, `20`, `40`, `60`
- **KNN**: 72 configuraciones (mismas que exp_10)
- **Total**: 5 × 72 × 50 = **18.000 evaluaciones MCCV**
- **LOO**: 88 folds solo para la configuración ganadora

## 4. Baselines
| Baseline | MCCV F1_macro | LOO F1_macro |
|----------|--------------|-------------|
| Majority (no text) | 0.380 | 0.380 |
| exp_10 corregido (nosvd, full vocab) | 0.646 | 0.661 |
| exp_5 tabular (external) | 0.667 | 0.689 |

## 5. Selección
F1_macro → brier_score ↓ → F1_yes → balanced_accuracy → MCC → nombre determinista.
