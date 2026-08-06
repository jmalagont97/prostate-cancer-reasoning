# Class-Weighted Composite Fuzzy ICI Diagnostic Confidence Prediction (exp_17) Summary Report

**Date**: 2026-08-05  
**Model**: Class-Weighted Decision Tree Meta-Thresholding on Composite Fuzzy ICI (`class_weight='balanced'`)  
**Dataset**: Labeled Reasoning Cohort ($N_{labeled} = 88$)  

## Phase A: Learned Balanced Meta-Thresholds (100 MCCV Splits)
- **Meta-Threshold 1 ($ar{\tau}_1$, Uncertain / Borderline)**: `0.0180` ($	ext{std} = 0.0318$)  
- **Meta-Threshold 2 ($ar{\tau}_2$, Borderline / Clear)**: `0.1266` ($	ext{std} = 0.0612$)  

## Phase B: Frozen LOOCV Out-of-Fold Evaluation (88 Folds)
- **3-Class Macro-F1**: **`0.4470`**  
- **Accuracy**: **`0.5795`** (51/88 correct)  
- **Spearman Rank Correlation ($ho$)**: **`0.2790`** (p-value: `8.4867e-03`)  

### 3x3 Confusion Matrix Counts:
| Ground Truth \ Predicted | Uncertain | Borderline | Clear |
|:---|:---:|:---:|:---:|
| **Uncertain** ($N=15$) | **2** | 5 | 7 |
| **Borderline** ($N=18$) | 2 | **10** | 6 |
| **Clear** ($N=55$) | 1 | 16 | **39** |

## Comparison against Baseline (exp_10 Hard Composite ICI Meta-Thresholding)
| Experiment | ICI Formulation | Decision Tree Weighting | Meta ar{\tau}_1 | Meta ar{\tau}_2 | LOOCV Macro-F1 | LOOCV Accuracy | Spearman ho |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exp_10` (Hard Composite ICI)** | Discrete Hard | Balanced | 0.0669 | 0.2960 | **0.3691** | **39.77%** | **0.1228** |
| **`exp_17` (Fuzzy Composite ICI)** | **Continuous Soft** | **Balanced** | **0.0180** | **0.1266** | **0.4470** | **57.95%** | **0.2790** |
