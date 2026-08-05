# Experiment Design: Out-of-Fold Diagnostic Confidence Prediction via Multimodal Probability State Vector p = [p_tab, p_mri, p_text]
**Experiment**: experiments/exp_12/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Replacing the 1D scalar $ICI$ with the full 3D unimodal output probability vector $\mathbf{p} = [p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}]^T$ as input to a class-weighted Decision Tree (`DecisionTreeClassifier(max_depth=3, class_weight='balanced')`) trained locally within each Leave-One-Out (LOOCV) fold will preserve multi-dimensional modal interactions and modal dominance, significantly improving out-of-fold 3-class Macro-F1 (target $\ge 0.50$) compared to 1D $ICI$ baselines without data leakage or optimism bias.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
  - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` target column)
  - Cohort: $N=88$ labeled complete-case patients with valid `confidence` annotations (`clear`: 56, `borderline`: 18, `uncertain`: 14).
- **Validation Harness (End-to-End LOOCV - 88 Folds)**:
  - For each fold $i \in \{1, \dots, 88\}$:
    - Train Tabular, MRI, and Text KNN models on 87 training cases.
    - Generate unimodal output probabilities $\mathbf{p}_{\text{train}} = [p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}] \in \mathbb{R}^{87 \times 3}$ for the 87 training cases.
    - Fit `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)` on $\mathbf{p}_{\text{train}}$ vs `confidence` of the 87 training cases.
    - Predict unimodal probabilities $\mathbf{p}_{\text{test}}^{(i)} \in \mathbb{R}^3$ for held-out test patient $i$.
    - Classify confidence of test patient $i$ applying the fold's decision tree model.

## 3. Mathematical Formulation
For each patient $i$, the uncertainty state vector is:
$$\mathbf{p}^{(i)} = \begin{bmatrix} p_{\text{tab}}^{(i)} \\ p_{\text{mri}}^{(i)} \\ p_{\text{text}}^{(i)} \end{bmatrix} \in [0, 1]^3$$

Inverse Class Weighting in Decision Tree:
$$w_k = \frac{N_{\text{train}}}{3 \cdot N_{k, \text{train}}}$$
where $w_{\text{clear}} \approx 0.52$, $w_{\text{borderline}} \approx 1.63$, and $w_{\text{uncertain}} \approx 2.10$.

## 4. File Layout for This Experiment
```
experiments/exp_12/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← 3D probability vector LOOCV evaluation script
├── results/
│   ├── feature_importances.json       ← mean feature importances of p_tab, p_mri, p_text
│   ├── loocv_confidence_metrics.json  ← 3-class F1-Macro, Accuracy, Spearman rho
│   └── loocv_confidence_predictions.csv ← patient-level probabilities, predicted & ground-truth confidence
└── reports/
    ├── figures/
    │   ├── feature_importance_bar.png  ← bar plot of modal probability feature importances
    │   └── confusion_matrix_3class.png ← 3x3 confusion matrix of confidence predictions
    └── summary.md             ← write-up of 3D probability vector confidence prediction results
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 3-class Macro-F1 score under LOOCV.
- **Secondary Metrics**: Accuracy, Spearman rank correlation ($\rho$), 3x3 Confusion Matrix, Modal Feature Importances.

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Feature importances saved in `results/feature_importances.json`
- [ ] Predictions and metrics logged to `results/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run the experiment.
