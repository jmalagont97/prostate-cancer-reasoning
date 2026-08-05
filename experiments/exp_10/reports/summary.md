# Balanced Out-of-Fold Diagnostic Confidence Prediction (exp_10) Summary Report

**Date**: 2026-08-05  
**Model**: Class-Weighted Out-of-Fold ICI + Decision Tree Meta-Thresholding (MCCV $\to$ LOOCV)  
**Dataset**: Labeled Reasoning Cohort ($N_{labeled} = 88$)  

## Phase A: Learned Balanced Meta-Thresholds (100 MCCV Splits)
- **Meta-Threshold 1 ($ar{\tau}_1$, Uncertain / Borderline)**: `0.0669` ($	ext{std} = 0.0769$)  
- **Meta-Threshold 2 ($ar{\tau}_2$, Borderline / Clear)**: `0.2960` ($	ext{std} = 0.2023$)  

## Phase B: Frozen LOOCV Out-of-Fold Evaluation (88 Folds)
- **3-Class Macro-F1**: **`0.3691`**  
- **Accuracy**: **`0.3977`** (35.0/88 correct)  
- **Spearman Rank Correlation ($ho$)**: **`0.1228`** (p-value: `2.5447e-01`)  

### 3x3 Confusion Matrix Counts:
| Ground Truth \ Predicted | Uncertain | Borderline | Clear |
|:---|:---:|:---:|:---:|
| **Uncertain** | 6 | 3 | 5 |
| **Borderline** | 7 | 8 | 3 |
| **Clear** | 17 | 18 | 21 |
