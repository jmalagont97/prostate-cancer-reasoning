# Hybrid Multimodal Late Fusion LOOCV (exp_18) Summary Report

**Date**: 2026-08-05  
**Model**: Hybrid Late-Fusion Soft-Voting Ensemble (Tabular Fuzzy KNN + MRI Hard KNN + Text Hard KNN)  
**Dataset**: Labeled Complete-Case Cohort ($N_{labeled} = 88$)  

## Comparative Ensemble Performance Across Hybrid Conditions (LOOCV 88 Folds)

| Condition | Weights (Tabular Fuzzy, MRI Hard, Text Hard) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Unimodal-Tabular-Fuzzy` | `[1.00, 0.00, 0.00]` | **0.6364** | **65.91%** | **0.7407** | **0.5294** | **0.6304** | **0.2908** |
| `Unimodal-MRI-Hard` | `[0.00, 1.00, 0.00]` | **0.5335** | **56.82%** | **0.6852** | **0.3824** | **0.5517** | **0.2879** |
| `Unimodal-Text-Hard` | `[0.00, 0.00, 1.00]` | **0.6645** | **68.18%** | **0.7407** | **0.5882** | **0.6645** | **0.3182** |
| `Equal-Hybrid-Fusion` | `[0.33, 0.33, 0.33]` | **0.6111** | **64.77%** | **0.7778** | **0.4412** | **0.7369** | **0.1997** |
| `Bimodal-TabularFuzzy-TextHard` | `[0.50, 0.00, 0.50]` | **0.5906** | **65.91%** | **0.8704** | **0.3235** | **0.7440** | **0.2091** |
| `Bimodal-TabularFuzzy-MRIHard` | `[0.50, 0.50, 0.00]` | **0.6207** | **65.91%** | **0.7963** | **0.4412** | **0.6315** | **0.2429** |
| `Bimodal-TextHard-MRIHard` | `[0.00, 0.50, 0.50]` | **0.6558** | **69.32%** | **0.8333** | **0.4706** | **0.7059** | **0.2216** |
| `Optimal-Weighted-Hybrid` | `[0.05, 0.50, 0.45]` | **0.6713** | **70.45%** | **0.8333** | **0.5000** | **0.7334** | **0.2151** |


## Comparison against Baselines (exp_8 Standard Hard Fusion vs exp_16 All-Fuzzy Fusion)
| Experiment / Fusion Strategy | Optimal Weights (Tab, MRI, Text) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`exp_8` All Hard KNN Fusion** | `[0.25, 0.41, 0.34]` | 0.7171 | 75.00% | 0.8889 | 0.5294 | 0.7715 | All Hard Voting |
| **`exp_16` All Fuzzy KNN Fusion** | `[0.15, 0.55, 0.30]` | 0.6813 | 71.59% | 0.8519 | 0.5000 | 0.7053 | All Soft Targets |
| **`exp_18` Hybrid KNN Fusion** | `[0.05, 0.50, 0.45]` | **0.6713** | **70.45%** | **0.8333** | **0.5000** | **0.7334** | **Hybrid Optimal Combination** |
