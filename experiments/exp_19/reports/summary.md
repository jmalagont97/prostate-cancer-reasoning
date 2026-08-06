# Class-Weighted Composite Hybrid ICI Diagnostic Confidence Prediction (exp_19) Summary Report

**Date**: 2026-08-05  
**Model**: Class-Weighted Decision Tree Meta-Thresholding on Composite Hybrid ICI (`class_weight='balanced'`)  
**Dataset**: Labeled Reasoning Cohort ($N_{labeled} = 88$)  

## Phase A: Learned Balanced Meta-Thresholds (100 MCCV Splits)
- **Meta-Threshold 1 ($ar{\tau}_1$, Uncertain / Borderline)**: `0.1169` ($	ext{std} = 0.1629$)  
- **Meta-Threshold 2 ($ar{\tau}_2$, Borderline / Clear)**: `0.4338` ($	ext{std} = 0.2055$)  

## Phase B: Frozen LOOCV Out-of-Fold Evaluation (88 Folds)
- **3-Class Macro-F1**: **`0.3885`**  
- **Accuracy**: **`0.4205`** (37/88 correct)  
- **Spearman Rank Correlation ($ho$)**: **`0.1681`** (p-value: `1.1745e-01`)  

### 3x3 Confusion Matrix Counts:
| Ground Truth \ Predicted | Uncertain | Borderline | Clear |
|:---|:---:|:---:|:---:|
| **Uncertain** ($N=15$) | **8** | 3 | 3 |
| **Borderline** ($N=18$) | 9 | **6** | 3 |
| **Clear** ($N=55$) | 24 | 9 | **23** |

## Comparison across All ICI Diagnostic Confidence Experiments
| Experiment | Multimodal Sources | ICI Formulation | Meta ar{\tau}_1 | Meta ar{\tau}_2 | LOOCV Macro-F1 | LOOCV Accuracy | Spearman ho |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exp_10`** | All Hard KNN | Hard Composite | 0.0669 | 0.2960 | 0.3691 | 39.77% | 0.1228 |
| **`exp_17`** | All Fuzzy KNN | Fuzzy Composite | 0.0180 | 0.1266 | 0.4470 | 57.95% | 0.2790 |
| **`exp_19`** | **Hybrid KNN (Tab Fuzzy + MRI/Text Hard)** | **Hybrid Composite** | **0.1169** | **0.4338** | **0.3885** | **42.05%** | **0.1681** |
