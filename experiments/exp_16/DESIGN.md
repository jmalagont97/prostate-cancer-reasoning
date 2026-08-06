# Experiment Design: Multimodal Fuzzy KNN Late Fusion Soft-Voting LOOCV Evaluation
**Experiment**: experiments/exp_16/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Combining continuous calibrated soft probabilities ($\tilde{p}_{\text{tab}}, \tilde{p}_{\text{mri}}, \tilde{p}_{\text{text}}$) from the optimal unimodal Fuzzy KNN Regressors (`exp_13` Tabular, `exp_14` MRI, `exp_15` Text) via weighted late-fusion soft voting in LOOCV will leverage complementary cross-modal information, outperforming standard late-fusion KNN (`exp_8` LOOCV Macro-F1 = **0.7171**) and individual unimodal Fuzzy KNN models.

## 2. Experimental Setup
- **Unimodal Optimal Fuzzy KNN Models (Frozen)**:
  1. **Tabular Fuzzy KNN (`exp_13`)**: $k^*=1$, `weights='uniform'`, `metric='euclidean'`, trained on soft targets $\tilde{y} \in [0.0, 1.0]$.
  2. **MRI Fuzzy KNN (`exp_14`)**: Representation `embedkit_unsup` (384D), $k^*=3$, `weights='uniform'`, `metric='euclidean'`, trained on soft targets $\tilde{y} \in [0.0, 1.0]$.
  3. **Text Fuzzy KNN (`exp_15`)**: `max_features=None`, Representation `pca` (90% variance), $k^*=3$, `weights='uniform'`, `metric='cosine'`, trained on soft targets $\tilde{y} \in [0.0, 1.0]$.
- **Cohort**: $N=88$ labeled complete-case cohort for LOOCV final evaluation.
- **Late Fusion Mechanism**:
  - For each fold $i$ in LOOCV, train all three unimodal Fuzzy KNN models on the 87 training samples and predict soft probability vector $\mathbf{\tilde{p}}_i = [\tilde{p}_{\text{tab}, i}, \tilde{p}_{\text{mri}, i}, \tilde{p}_{\text{text}, i}]^T \in [0.0, 1.0]^3$.
  - Compute late-fusion soft probability:
    $$\tilde{p}_{\text{fusion}, i} = w_{\text{tab}} \cdot \tilde{p}_{\text{tab}, i} + w_{\text{mri}} \cdot \tilde{p}_{\text{mri}, i} + w_{\text{text}} \cdot \tilde{p}_{\text{text}, i}, \quad \text{where } \sum_{m} w_m = 1, \; w_m \ge 0$$
  - Binary decision threshold: $\tilde{p}_{\text{fusion}, i} \ge 0.50 \implies \hat{y}_i = 1$.
- **Ensemble Conditions Evaluated**:
  1. `Unimodal-Tabular` ($w = [1.00, 0.00, 0.00]$)
  2. `Unimodal-MRI` ($w = [0.00, 1.00, 0.00]$)
  3. `Unimodal-Text` ($w = [0.00, 0.00, 1.00]$)
  4. `Equal-Trimodal-Fusion` ($w = [0.333, 0.333, 0.333]$)
  5. `Bimodal-Tabular-Text` ($w = [0.50, 0.00, 0.50]$)
  6. `Bimodal-Tabular-MRI` ($w = [0.50, 0.50, 0.00]$)
  7. `Bimodal-Text-MRI` ($w = [0.00, 0.50, 0.50]$)
  8. `Optimal-Weighted-Trimodal` (Grid search over $w \in \Delta^2$ with step size 0.05 maximizing LOOCV Macro-F1).

## 3. File Layout for This Experiment
```
experiments/exp_16/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← unimodal Fuzzy KNN retraining + Late Fusion LOOCV script
├── results/
│   ├── best_fusion_weights.json     ← optimal modality fusion weights [w_tab, w_mri, w_text]
│   ├── fusion_grid_results.csv      ← metrics across weight combinations
│   ├── loocv_metrics.json           ← LOOCV metrics across all fusion conditions
│   ├── oof_predictions.csv          ← out-of-fold soft probabilities per modality & ensemble
│   └── git_commit.txt               ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── confusion_matrix.png     ← LOOCV 2x2 confusion matrix of optimal fusion
    │   └── roc_curves.png           ← comparative ROC curves across modalities & fusion
    └── summary.md                   ← final report contrasting exp_16 vs exp_8
```

## 4. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 2-class Macro-F1 under LOOCV.
- **Baseline to Beat**: `exp_8` Multimodal Standard KNN Late Fusion (Equal Trimodal Macro-F1 = **0.7171**, Optimal Weighted Macro-F1 = **0.7171**, Accuracy = **75.00%**, Sensitivity = **0.8889**, Specificity = **0.5294**).
- **Secondary Metrics**: Accuracy, Sensitivity (Recall), Specificity, AUROC, Brier Score (calibration).

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for EmbedKit and PCA)
- [ ] Config and scripts saved in `scripts/`
- [ ] Fusion grid results logged to `results/fusion_grid_results.csv`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_16`.
