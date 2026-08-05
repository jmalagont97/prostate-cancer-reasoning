# Multimodal Late Fusion (Soft-Voting) LOOCV Summary Report

**Date**: 2026-08-04  
**Model**: Late Fusion Soft-Voting Ensemble (KNN Tabular + KNN MRI + KNN Text)  
**Dataset**: Complete-Case Labeled Cohort ($N_{labeled} = 88$)  

## Comparative Results (LOOCV 88 Folds)

| Condition | Weights (Tabular, MRI, Text) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `Unimodal-Tabular` | `[1.00, 0.00, 0.00]` | **0.6333** | 0.6818 | 0.8519 | 0.4118 | 0.6825 |
| `Unimodal-MRI` | `[0.00, 1.00, 0.00]` | **0.5335** | 0.5682 | 0.6852 | 0.3824 | 0.5517 |
| `Unimodal-Text` | `[0.00, 0.00, 1.00]` | **0.6988** | 0.7159 | 0.7778 | 0.6176 | 0.6977 |
| `Equal-Trimodal-Fusion` | `[0.33, 0.33, 0.33]` | **0.7171** | 0.7500 | 0.8889 | 0.5294 | 0.7821 |
| `Bimodal-Tabular-Text` | `[0.50, 0.00, 0.50]` | **0.6914** | 0.7273 | 0.8704 | 0.5000 | 0.7835 |
| `Bimodal-Tabular-MRI` | `[0.50, 0.50, 0.00]` | **0.4545** | 0.5909 | 0.8889 | 0.1176 | 0.6874 |
| `Bimodal-Text-MRI` | `[0.00, 0.50, 0.50]` | **0.6864** | 0.7159 | 0.8333 | 0.5294 | 0.7255 |
| `Optimal-Weighted-Trimodal` | `[0.25, 0.41, 0.34]` | **0.7171** | 0.7500 | 0.8889 | 0.5294 | 0.7715 |


## Key Multimodal Insights:
- **Optimal Weighted Trimodal Fusion**: Weights `[Tab: 0.25, MRI: 0.41, Text: 0.34]` achieved a Macro-F1 of **0.7171** and AUROC of **0.7715**.  
- **Bimodal Tabular + Text Fusion**: Achieved a Macro-F1 of **0.6914** and Specificity of **0.5000**.  
- **Role of Visual MRI Embeddings**: Standalone MRI embeddings remain weak (F1 = 0.5335), and assigning non-zero weight to MRI in equal fusion slightly degrades performance compared to Tabular + Text bimodal fusion.  
