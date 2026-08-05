# MRI Representations Model Selection (exp_6) Summary Report

**Date**: 2026-08-04  
**Model**: K-Nearest Neighbors Classifier on MRI Embeddings  
**Dataset**: Labeled Complete-Case MRI Dataset ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Configuration Found**:
  - **Representation**: `embedkit_sup`  
  - `n_neighbors` (k): `3`  
  - `weights`: `uniform`  
  - `metric`: `euclidean`  
  - **Frozen EmbedKit Target Dimension**: `384`  
  - **Mean Validation Macro-F1**: `0.5469`  

### Mean Dynamic Dimensions Logged for EmbedKit:
- Unsupervised mode dimension: 384.0 (std: 0.0)  
- Supervised mode dimension: 384.0 (std: 0.0)  

### Top 5 Hyperparameter Configurations:
| Rank | Representation | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | embedkit_sup | 3 | uniform | euclidean | 0.5469 | 0.5867 | 0.7036 | 0.4029 |
| 2 | embedkit_unsup | 3 | uniform | cosine | 0.5469 | 0.5867 | 0.7036 | 0.4029 |
| 3 | embedkit_sup | 3 | uniform | cosine | 0.5469 | 0.5867 | 0.7036 | 0.4029 |
| 4 | embedkit_sup | 3 | distance | euclidean | 0.5469 | 0.5867 | 0.7036 | 0.4029 |
| 5 | embedkit_unsup | 3 | uniform | euclidean | 0.5469 | 0.5867 | 0.7036 | 0.4029 |

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
The optimal representation and KNN configuration was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:

- **Final Macro-F1**: `0.5335`  
- **Final Accuracy**: `0.5682` (50 correct out of 88 cases)  
- **Sensitivity (Yes class)**: `0.6852` (Correctly identified 37 out of 56 yes cases)  
- **Specificity (No class)**: `0.3824` (Correctly identified 13 out of 32 no cases)  

### Confusion Matrix Counts:
- True Negatives (TN): `13`  
- False Positives (FP): `21`  
- False Negatives (FN): `17`  
- True Positives (TP): `37`  

