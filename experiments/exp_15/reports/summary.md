# Text Fuzzy KNN Sweep & LOOCV (exp_15) Summary Report

**Date**: 2026-08-05  
**Model**: Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) on TF-IDF Text Prompts  
**Dataset**: Labeled Complete-Case Text Dataset ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Configuration Found**:
  - **Vocabulary Size (`max_features`)**: `None`  
  - **Representation**: `pca`  
  - `n_neighbors` (k): `3`  
  - `weights`: `uniform`  
  - `metric`: `cosine`  
  - **Mean Validation Macro-F1**: `0.6393` ($	ext{std} = 0.1105$)  

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
- **Out-of-Fold Macro-F1**: **`0.6558`**  
- **Out-of-Fold Accuracy**: **`0.6932`** (61/88 correct)  
- **Sensitivity (Yes class)**: **`0.8333`** (45/56 correct)  
- **Specificity (No class)**: **`0.4706`** (16/32 correct)  
- **AUROC**: **`0.6868`**  
- **Brier Calibration Score**: **`0.2195`**  

### 2x2 Confusion Matrix Counts:
| Ground Truth \ Predicted | No Biopsy | Biopsy |
|:---|:---:|:---:|
| **No Biopsy** ($N=32$) | **16** | 18 |
| **Biopsy** ($N=56$) | 9 | **45** |

## Comparison against Baseline (exp_7 Standard Text KNN)
| Model / Harness | max_features | Representation | Hyperparameters (k, w, m) | MCCV Mean Macro-F1 | LOOCV Macro-F1 | LOOCV Accuracy | Sensitivity | Specificity | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exp_7` (Standard KNN)** | 500 | pca | k=1, uniform, cosine | 0.6329 | 0.6988 | 71.59% | 0.7778 | 0.6176 | — |
| **`exp_15` (Fuzzy KNN)** | **None** | **pca** | k=3, uniform, cosine | **0.6393** | **0.6558** | **69.32%** | **0.8333** | **0.4706** | **0.6868** |
