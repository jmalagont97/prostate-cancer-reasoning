# Multimodal Fuzzy KNN Late Fusion LOOCV (exp_16) Summary Report

**Date**: 2026-08-05  
**Model**: Late-Fusion Soft-Voting Ensemble (Tabular Fuzzy KNN + MRI Fuzzy KNN + Text Fuzzy KNN)  
**Dataset**: Labeled Complete-Case Cohort ($N_{labeled} = 88$)  

## Comparative Ensemble Performance Across Conditions (LOOCV 88 Folds)

| Condition | Weights (Tabular, MRI, Text) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Unimodal-Tabular` | `[1.00, 0.00, 0.00]` | **0.6364** | **65.91%** | **0.7407** | **0.5294** | **0.6304** | **0.2908** |
| `Unimodal-MRI` | `[0.00, 1.00, 0.00]` | **0.5487** | **57.95%** | **0.6852** | **0.4118** | **0.5539** | **0.2579** |
| `Unimodal-Text` | `[0.00, 0.00, 1.00]` | **0.6558** | **69.32%** | **0.8333** | **0.4706** | **0.6868** | **0.2195** |
| `Equal-Trimodal-Fusion` | `[0.33, 0.33, 0.33]` | **0.6143** | **65.91%** | **0.8148** | **0.4118** | **0.7252** | **0.2030** |
| `Bimodal-Tabular-Text` | `[0.50, 0.00, 0.50]` | **0.6317** | **65.91%** | **0.7593** | **0.5000** | **0.7121** | **0.2083** |
| `Bimodal-Tabular-MRI` | `[0.50, 0.50, 0.00]` | **0.6460** | **68.18%** | **0.8148** | **0.4706** | **0.6481** | **0.2312** |
| `Bimodal-Text-MRI` | `[0.00, 0.50, 0.50]` | **0.6497** | **69.32%** | **0.8519** | **0.4412** | **0.7021** | **0.2094** |
| `Optimal-Weighted-Trimodal` | `[0.15, 0.55, 0.30]` | **0.6813** | **71.59%** | **0.8519** | **0.5000** | **0.7053** | **0.2093** |


## Comparison against Baseline (exp_8 Standard Multimodal KNN Late Fusion)
| Strategy | Optimal Weights | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`exp_8` Standard KNN Fusion** | `[0.25, 0.41, 0.34]` | 0.7171 | 75.00% | 0.8889 | 0.5294 | 0.7715 | Hard Voting |
| **`exp_16` Fuzzy KNN Fusion** | `[0.15, 0.55, 0.30]` | **0.6813** | **71.59%** | **0.8519** | **0.5000** | **0.7053** | **Soft Targets Calibrated** |
