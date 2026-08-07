# Experiment Design: Clinical Feature Relevance Attribution via SHAP Shapley Values
**Experiment**: experiments/exp_21/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Computing sample-level Shapley attributions $|\phi_j^{(i)}|$ using SHAP (`KernelExplainer`) on the Distance-Weighted Tabular Fuzzy KNN model (`exp_13`) and learning feature-independent decision tree meta-thresholds in Phase A (100 MCCV splits) will resolve feature collinearities (e.g. `psa`, `psad`, `psap`) and significantly improve out-of-fold 4-class Macro-F1, Accuracy, and Spearman rank correlations ($\rho$) against expert urologist annotations over the 1D perturbation baseline (`exp_20`).

## 2. Experimental Setup
- **Cohort**: Complete-Case Labeled Training Cohort ($N=88$ complete-case patients).
- **Underlying Model**: Distance-Weighted Tabular Fuzzy KNN (`exp_13`): $k^*=1$, `uniform`, `euclidean`, MinMaxScaler + OneHotEncoder DRE.
- **SHAP Attribution Engine**:
  - `shap.KernelExplainer` fitted on the training split feature matrix to compute Shapley values $\phi_{j}^{(i)} \in [-\infty, +\infty]$ for all 8 tabular features ($j \in \{\text{age}, \text{psa}, \text{vol}, \text{pirads}, \text{psad}, \text{psav}, \text{psap}, \text{dre}\}$).
  - Absolute Shapley magnitude $|\phi_j^{(i)}|$ used as the continuous feature relevance input.
- **Two-Phase Validation Harness**:
  - **Phase A (100-Split MCCV Meta-Threshold Learning)**:
    For each split $s \in \{1, \dots, 100\}$ and each feature $j$:
    Fit a 1D `DecisionTreeClassifier(max_depth=3, class_weight='balanced')` on train set $|\phi_{j, s}^{(train)}|$ to predict urologist ground-truth importance classes (`not_used`=0, `noted`=1, `important`=2, `decisive`=3).
    Extract cut-point thresholds $(\tau_{1,s}^{(j)}, \tau_{2,s}^{(j)}, \tau_{3,s}^{(j)})$.
    Compute mean feature-independent meta-thresholds:
    $$\bar{\tau}_1^{(j)} = \frac{1}{100}\sum_{s=1}^{100} \tau_{1,s}^{(j)}, \quad \bar{\tau}_2^{(j)} = \frac{1}{100}\sum_{s=1}^{100} \tau_{2,s}^{(j)}, \quad \bar{\tau}_3^{(j)} = \frac{1}{100}\sum_{s=1}^{100} \tau_{3,s}^{(j)}$$
  - **Phase B (Frozen Out-of-Fold LOOCV Evaluation)**:
    Under LOOCV ($N=88$ folds), compute out-of-fold Shapley magnitudes $|\phi_{j, i}^{\text{OOF}}|$ and apply the frozen meta-thresholds $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$ to predict discrete feature importance:
    $$\hat{y}_{j, i} = \begin{cases} \text{not\_used} (0) & \text{if } |\phi_{j, i}^{\text{OOF}}| < \bar{\tau}_1^{(j)} \\ \text{noted} (1) & \text{if } \bar{\tau}_1^{(j)} \le |\phi_{j, i}^{\text{OOF}}| < \bar{\tau}_2^{(j)} \\ \text{important} (2) & \text{if } \bar{\tau}_2^{(j)} \le |\phi_{j, i}^{\text{OOF}}| < \bar{\tau}_3^{(j)} \\ \text{decisive} (3) & \text{if } |\phi_{j, i}^{\text{OOF}}| \ge \bar{\tau}_3^{(j)} \end{cases}$$

## 3. File Layout for This Experiment
```
experiments/exp_21/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← SHAP extraction, MCCV Phase A & LOOCV Phase B evaluation script
├── results/
│   ├── oof_shap_attributions.csv      ← out-of-fold SHAP values & predictions per patient
│   ├── meta_thresholds_shap.json      ← learned meta-thresholds per feature
│   ├── feature_attribution_metrics.json ← Spearman rho, p-values, Macro-F1, Accuracy per feature
│   ├── feature_confusion_matrices.json  ← raw 4x4 confusion matrices per feature
│   └── git_commit.txt                 ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── cm_dre.png                 ← 4x4 confusion matrix image for DRE
    │   ├── cm_psav.png                ← 4x4 confusion matrix image for PSAV
    │   ├── confusion_matrices_all_features.png ← 2x4 grid image of all 8 confusion matrices
    │   └── shap_summary_bar.png       ← SHAP global feature importance bar plot
    └── summary.md                     ← final summary report
```

## 4. Evaluation Protocol & Deliverables
1. **Per-Feature Ordinal Metrics**: Spearman Rank Correlation ($\rho$) and exact $p$-value against urologist annotations (`weight_*`).
2. **Per-Feature Classification Metrics**: 4-Class Macro-F1, Overall Accuracy, Per-Class Sensitivity, Per-Class Specificity.
3. **Confusion Matrix PNG Images**: Individual 4x4 image per feature and combined 2x4 grid image saved in `reports/figures/`.
4. **JSON Output Files**: `results/feature_attribution_metrics.json` and `results/feature_confusion_matrices.json`.

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Config and scripts saved in `scripts/`
- [ ] Quantitative metrics exported to JSON files in `results/`
- [ ] Confusion matrices saved as PNG images in `reports/figures/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_21`.
