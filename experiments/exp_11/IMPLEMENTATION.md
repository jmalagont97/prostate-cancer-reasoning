# Implementation Plan: exp_11 — TF-IDF + TruncatedSVD + KNN

**Experiment**: experiments/exp_11/  
**Project**: pathology-reasoning  
**Date**: 2026-08-17  
**Status**: Complete

## Script
`experiments/exp_11/scripts/run_tfidf_svd_knn_experiment.py`

Single self-contained script (no external src/ dependencies). Based on exp_10's `run_tfidf_knn_experiment.py`.

### Key Changes from exp_10
1. **TruncatedSVD** after TF-IDF, before KNN:
   - `n_components ∈ {None, 1, 20, 40, 60}` (None = no SVD control)
   - `random_state=42`, `n_iter=5`, `algorithm="randomized"`
2. **L2 normalization** after SVD (fixed, not a hyperparameter)
3. **Config name** includes SVD info: `tfidf_mfall_svd{comp}_knn_...` or `tfidf_mfall_nosvd_knn_...`
4. **Grid**: 5 SVD conditions × 72 KNN = 360 configs
5. **Variance explained** reported per representation (diagnostic only)
6. **max_features=None** always (full vocabulary)

### Artefacts
- `results/<winner>/config_log.json` (all 360 configs)
- `results/<winner>/metrics_mccv.json`
- `results/<winner>/metrics_loo.json`
- `results/<winner>/hyperparameters.json` (includes SVD params)
- `results/<winner>/oof_predictions_mccv.csv`
- `results/<winner>/oof_predictions_loo.csv`
- `results/<winner>/confusion_matrices.json`
- `results/<winner>/validation_report.json` (all_passed: true)
- `results/<winner>/variance_explained.json`
- `results/summary_selection.json`
- `results/config_log.json`
