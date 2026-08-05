# Dynamic Out-of-Fold Diagnostic Confidence Prediction (exp_11) Summary Report

**Date**: 2026-08-05  
**Model**: Pure Dynamic LOOCV Decision Tree Thresholding ($ICI \to \text{confidence}$)  
**Dataset**: Labeled Reasoning Cohort ($N_{labeled} = 88$)  

## Dynamic Threshold Stability across 88 LOOCV Folds
- **Mean Dynamic $\tau_1$ (Uncertain / Borderline)**: `0.0966` ($	ext{std} = 0.0000$)  
- **Mean Dynamic $\tau_2$ (Borderline / Clear)**: `0.2892` ($	ext{std} = 0.1202$)  

## Out-of-Fold Evaluation Metrics (88 Folds)
- **3-Class Macro-F1**: **`0.3388`**  
- **Accuracy**: **`0.3636`** (32.0/88 correct)  
- **Spearman Rank Correlation ($ho$)**: **`0.1228`** (p-value: `2.5447e-01`)  

### 3x3 Confusion Matrix Counts:
| Ground Truth \ Predicted | Uncertain | Borderline | Clear |
|:---|:---:|:---:|:---:|
| **Uncertain** | 6 | 4 | 4 |
| **Borderline** | 7 | 7 | 4 |
| **Clear** | 17 | 20 | 19 |
