# Tabular Fuzzy KNN Sweep & LOOCV (exp_13) Summary Report

**Date**: 2026-08-05  
**Model**: Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`)  
**Dataset**: Labeled Complete-Case Tabular Clinical Data ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Hyperparameters Found**:
  - `n_neighbors` (k): `1`  
  - `weights`: `uniform`  
  - `metric`: `euclidean`  
  - **Mean Validation Macro-F1**: `0.6117` ($	ext{std} = 0.1144$)  

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
- **Out-of-Fold Macro-F1**: **`0.6364`**  
- **Out-of-Fold Accuracy**: **`0.6591`** (58/88 correct)  
- **Sensitivity (Yes class)**: **`0.7407`** (40/56 correct)  
- **Specificity (No class)**: **`0.5294`** (18/32 correct)  
- **AUROC**: **`0.6304`**  
- **Brier Calibration Score**: **`0.2908`**  

### 2x2 Confusion Matrix Counts:
| Ground Truth \ Predicted | No Biopsy | Biopsy |
|:---|:---:|:---:|
| **No Biopsy** ($N=32$) | **18** | 16 |
| **Biopsy** ($N=56$) | 14 | **40** |

## Comparison against Baseline (exp_5 Standard Tabular KNN)
| Model / Harness | Hyperparameters (k, w, m) | MCCV Mean Macro-F1 | LOOCV Macro-F1 | LOOCV Accuracy | Sensitivity | Specificity | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exp_5` (Standard KNN)** | k=3, uniform, euclidean | 0.6218 | 0.6333 | 68.18% | 0.8519 | 0.4118 | — |
| **`exp_13` (Fuzzy KNN)** | k=1, uniform, euclidean | **0.6117** | **0.6364** | **65.91%** | **0.7407** | **0.5294** | **0.6304** |
