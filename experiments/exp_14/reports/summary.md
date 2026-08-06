# MRI Fuzzy KNN Sweep & LOOCV (exp_14) Summary Report

**Date**: 2026-08-05  
**Model**: Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) on MRI Embeddings  
**Dataset**: Labeled Complete-Case MRI Dataset ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Configuration Found**:
  - **Representation**: `embedkit_unsup`  
  - `n_neighbors` (k): `3`  
  - `weights`: `uniform`  
  - `metric`: `euclidean`  
  - **Mean Validation Macro-F1**: `0.5422` ($	ext{std} = 0.1130$)  

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
- **Out-of-Fold Macro-F1**: **`0.5335`**  
- **Out-of-Fold Accuracy**: **`0.5682`** (50/88 correct)  
- **Sensitivity (Yes class)**: **`0.6852`** (37/56 correct)  
- **Specificity (No class)**: **`0.3824`** (13/32 correct)  
- **AUROC**: **`0.5387`**  
- **Brier Calibration Score**: **`0.2623`**  

### 2x2 Confusion Matrix Counts:
| Ground Truth \ Predicted | No Biopsy | Biopsy |
|:---|:---:|:---:|
| **No Biopsy** ($N=32$) | **13** | 21 |
| **Biopsy** ($N=56$) | 17 | **37** |

## Comparison against Baseline (exp_6 Standard MRI KNN)
| Model / Harness | Representation | Hyperparameters (k, w, m) | MCCV Mean Macro-F1 | LOOCV Macro-F1 | LOOCV Accuracy | Sensitivity | Specificity | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exp_6` (Standard KNN)** | embedkit_sup | k=3, uniform, euclidean | 0.5469 | 0.5335 | 56.82% | 0.6852 | 0.3824 | — |
| **`exp_14` (Fuzzy KNN)** | **embedkit_unsup** | k=3, uniform, euclidean | **0.5422** | **0.5335** | **56.82%** | **0.6852** | **0.3824** | **0.5387** |
