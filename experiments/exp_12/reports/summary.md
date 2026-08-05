# 3D Probability State Vector Diagnostic Confidence Prediction (exp_12) Summary Report

**Date**: 2026-08-05  
**Model**: Class-Weighted 3D Probability Vector Decision Tree ($p = [p_{tab}, p_{mri}, p_{text}] \to \text{confidence}$)  
**Dataset**: Labeled Reasoning Cohort ($N_{labeled} = 88$)  

## Modal Feature Importances
- **Tabular Probability ($p_{\text{tab}}$)**: `0.6184` ($	ext{std} = 0.1607$)  
- **MRI Probability ($p_{\text{mri}}$)**: `0.3347` ($	ext{std} = 0.0967$)  
- **Text Probability ($p_{\text{text}}$)**: `0.0469` ($	ext{std} = 0.1012$)  

## Out-of-Fold Evaluation Metrics (88 LOOCV Folds)
- **3-Class Macro-F1**: **`0.3331`**  
- **Accuracy**: **`0.3750`** (33.0/88 correct)  
- **Spearman Rank Correlation ($ho$)**: **`0.1228`** (p-value: `2.5447e-01`)  

### 3x3 Confusion Matrix Counts:
| Ground Truth \ Predicted | Uncertain | Borderline | Clear |
|:---|:---:|:---:|:---:|
| **Uncertain** | 5 | 7 | 2 |
| **Borderline** | 5 | 6 | 7 |
| **Clear** | 18 | 16 | 22 |
