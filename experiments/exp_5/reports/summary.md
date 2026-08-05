# Tabular KNN Model Selection (exp_5) Summary Report

**Date**: 2026-08-04  
**Model**: K-Nearest Neighbors Classifier  
**Dataset**: Labeled Complete-Case Tabular Clinical Data ($N_{labeled} = 88$)  

## Phase A: 100-Split MCCV Grid Search Results
- **Best Hyperparameters Found**:
  - `n_neighbors` (k): `3`  
  - `weights`: `uniform`  
  - `metric`: `euclidean`  
  - **Mean Validation Macro-F1**: `0.6218`  

### Top 5 Hyperparameter Configurations:
| Rank | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3 | uniform | euclidean | 0.6218 | 0.6717 | 0.8245 | 0.4314 |
| 2 | 3 | uniform | manhattan | 0.6181 | 0.6650 | 0.8100 | 0.4371 |
| 3 | 3 | distance | euclidean | 0.6146 | 0.6617 | 0.8064 | 0.4343 |
| 4 | 1 | uniform | euclidean | 0.6117 | 0.6444 | 0.7436 | 0.4886 |
| 5 | 1 | distance | euclidean | 0.6117 | 0.6444 | 0.7436 | 0.4886 |

## Phase B: Leave-One-Out (LOOCV) Generalization Performance
The optimal hyperparameter configuration ($k=3$, weights=uniform, metric=euclidean) was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:

- **Final Macro-F1**: `0.6333`  
- **Final Accuracy**: `0.6818` (60 correct out of 88 cases)  
- **Sensitivity (Yes class)**: `0.8519` (Correctly identified 46 out of 56 yes cases)  
- **Specificity (No class)**: `0.4118` (Correctly identified 14 out of 32 no cases)  

### Confusion Matrix Counts:
- True Negatives (TN): `14`  
- False Positives (FP): `20`  
- False Negatives (FN): `8`  
- True Positives (TP): `46`  

## Preprocessing Pipeline Details
- **Numerical variables**: scaled dynamically per fold using `MinMaxScaler` onto $[0, 1]$ interval.  
- **Categorical variables**: one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`.  
- **Feature space**: concatenated output size = 8 features (7 numerical, 1 category one-hot column for `dre`).  
