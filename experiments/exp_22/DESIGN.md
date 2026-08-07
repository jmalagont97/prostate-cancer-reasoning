# Experiment Design: Multivariate SHAP Vector Input Decision Tree Clinical Relevance Attribution
**Experiment**: experiments/exp_22/ · **Project**: pathology-reasoning · **Date**: 2026-08-06 · **Status**: Complete

---

## 1. Hypothesis
For each target clinical feature $j \in \{\text{age}, \text{psa}, \text{vol}, \text{pirads}, \text{psad}, \text{dre}\}$, training a multivariate `DecisionTreeClassifier(max_depth=3, class_weight='balanced')` taking the **full 6-dimensional SHAP vector** $\boldsymbol{\Phi}^{(i)} = [|\phi_{\text{age}}^{(i)}|, |\phi_{\text{psa}}^{(i)}|, |\phi_{\text{vol}}^{(i)}|, |\phi_{\text{pirads}}^{(i)}|, |\phi_{\text{psad}}^{(i)}|, |\phi_{\text{dre}}^{(i)}|]$ as input directly in Leave-One-Out Cross-Validation (LOOCV $N=88$) without 1D scalar thresholding or MCCV meta-averaging will capture cross-feature contextual interactions and significantly improve out-of-fold 4-class Macro-F1, Accuracy, and Spearman rank correlations ($\rho$) over `exp_21` (1D SHAP) and `exp_20` (1D Perturbation).

## 2. Experimental Setup
- **Cohort**: Complete-Case Labeled Training Cohort ($N=88$ complete-case patients).
- **Underlying Model**: Distance-Weighted Tabular Fuzzy KNN (`exp_13`): $k^*=1$, `uniform`, `euclidean`, MinMaxScaler + OneHotEncoder DRE.
- **SHAP Feature Attribution Engine**: `shap.KernelExplainer` on the fitted Tabular Fuzzy KNN model to extract absolute Shapley value vector $\boldsymbol{\Phi}^{(i)} \in \mathbb{R}^6$ per patient $i$.
- **Multivariate Input Attribution Classifier Architecture**:
  - Input Feature Space: Full 6-dimensional vector $\mathbf{X}^{(i)} = [|\phi_{\text{age}}^{(i)}|, |\phi_{\text{psa}}^{(i)}|, |\phi_{\text{vol}}^{(i)}|, |\phi_{\text{pirads}}^{(i)}|, |\phi_{\text{psad}}^{(i)}|, |\phi_{\text{dre}}^{(i)}|]$.
  - Target per Feature $j$: Ground-truth urologist discrete importance class $y_{j}^{(i)} \in \{\text{not\_used}=0, \text{noted}=1, \text{important}=2, \text{decisive}=3\}$.
  - Classifier Model: `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)`.
- **Validation Protocol**:
  - Direct Leave-One-Out Cross-Validation (LOOCV $N=88$ folds).
  - In each fold $i$:
    1. Train unimodal Tabular Fuzzy KNN on 87 training cases and compute SHAP vectors $\boldsymbol{\Phi}^{(k)}$ for $k \in \text{train}$.
    2. Fit 6 independent multivariate decision trees (one per target feature $j$) taking $\boldsymbol{\Phi}^{(\text{train})} \in \mathbb{R}^{87 \times 6}$ to predict $y_j^{(\text{train})}$.
    3. Predict patient $i$'s out-of-fold weight classes $\hat{y}_{j, i}$ using validation sample SHAP vector $\boldsymbol{\Phi}^{(i)}$.

## 3. File Layout for This Experiment
```
experiments/exp_22/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← LOOCV multivariate SHAP decision tree training & evaluation script
├── results/
│   ├── oof_multivariate_shap_predictions.csv ← out-of-fold predictions per patient
│   ├── feature_attribution_metrics.json      ← Spearman rho, p-values, Macro-F1, Accuracy per feature
│   └── feature_confusion_matrices.json       ← raw 4x4 confusion matrices per feature
└── reports/
    ├── figures/
    │   ├── cm_age.png                        ← 4x4 confusion matrix image for AGE
    │   ├── cm_psa.png                        ← 4x4 confusion matrix image for PSA
    │   ├── cm_vol.png                        ← 4x4 confusion matrix image for VOL
    │   ├── cm_pirads.png                     ← 4x4 confusion matrix image for PIRADS
    │   ├── cm_psad.png                       ← 4x4 confusion matrix image for PSAD
    │   ├── cm_dre.png                        ← 4x4 confusion matrix image for DRE
    │   ├── confusion_matrices_all_features.png ← 2x3 grid image of all 6 confusion matrices
    │   └── histograms_combined_f1_rho_exp22.png ← F1-Macro & |Rho| bar charts
    └── summary.md                            ← final summary report
```

## 4. Evaluation Protocol & Explicit Deliverables per Variable
For each of the 6 clinical variables ($j \in \{\text{age}, \text{psa}, \text{vol}, \text{pirads}, \text{psad}, \text{dre}\}$), the evaluation must be performed **independently**:
1. **Independent 4-Class Macro-F1 Score**: Calculated per feature $j$ by comparing out-of-fold predicted classes $\hat{y}_{j, i}$ vs ground-truth urologist annotations $y_{j, i}$.
2. **Independent Spearman Rank Correlation ($\rho$) & p-value**: Tested per feature $j$ to measure rank agreement between predictions $\hat{y}_{j, i}$ and human annotations $y_{j, i}$.
3. **Independent 4x4 Confusion Matrix PNG Images**: Generated and saved per feature (`cm_age.png`, `cm_psa.png`, `cm_vol.png`, `cm_pirads.png`, `cm_psad.png`, `cm_dre.png`) and consolidated into a 2x3 grid plot (`confusion_matrices_all_features.png`).
4. **Independent Accuracy, Sensitivity, and Specificity**: Calculated per class for each feature $j$.
5. **Baseline Benchmark Comparisons**: Compare independent Macro-F1 and Spearman $\rho$ against Uniform Random ($F1 \approx 0.17$), Stratified Random ($F1 \approx 0.25$), 1D Perturbation (`exp_20`), and 1D SHAP (`exp_21`).
6. **JSON Output Files**: `results/feature_attribution_metrics.json` and `results/feature_confusion_matrices.json`.

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Config and scripts saved in `scripts/`
- [ ] Quantitative metrics exported to JSON files in `results/`
- [ ] Confusion matrices saved as PNG images in `reports/figures/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan (hypothesis, conditions, metrics, decision rules).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_22`.
