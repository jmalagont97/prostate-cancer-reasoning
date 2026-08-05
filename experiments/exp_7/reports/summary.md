# Text TF-IDF Representations Model Selection (exp_7) Summary Report

**Date**: 2026-08-04  
**Model**: K-Nearest Neighbors Classifier on Clinical Text TF-IDF  
**Dataset**: Labeled Complete-Case Text Dataset ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Configuration Found**:
  - **Vocabulary Size (`max_features`)**: `500`  
  - **Representation**: `pca`  
  - `n_neighbors` (k): `1`  
  - `weights`: `uniform`  
  - `metric`: `cosine`  
  - **Mean Validation Macro-F1**: `0.6329`  

### Top 5 Hyperparameter Configurations:
| Rank | max_features | Representation | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 500 | pca | 1 | uniform | cosine | 0.6329 | 0.6611 | 0.7464 | 0.5271 |
| 2 | 500 | pca | 1 | distance | cosine | 0.6329 | 0.6611 | 0.7464 | 0.5271 |
| 3 | None | raw | 3 | uniform | cosine | 0.6300 | 0.6761 | 0.8209 | 0.4486 |
| 4 | None | raw | 3 | distance | cosine | 0.6300 | 0.6761 | 0.8209 | 0.4486 |
| 5 | None | raw | 3 | uniform | euclidean | 0.6300 | 0.6761 | 0.8209 | 0.4486 |

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
The optimal text representation and KNN configuration was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:

- **Final Macro-F1**: `0.6988`  
- **Final Accuracy**: `0.7159` (63 correct out of 88 cases)  
- **Sensitivity (Yes class)**: `0.7778` (Correctly identified 42 out of 56 yes cases)  
- **Specificity (No class)**: `0.6176` (Correctly identified 21 out of 32 no cases)  

### Confusion Matrix Counts:
- True Negatives (TN): `21`  
- False Positives (FP): `13`  
- False Negatives (FN): `12`  
- True Positives (TP): `42`  

