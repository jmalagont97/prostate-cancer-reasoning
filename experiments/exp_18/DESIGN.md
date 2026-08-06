# Experiment Design: Hybrid Multimodal Late Fusion LOOCV Evaluation (Tabular Fuzzy KNN + MRI Standard Hard KNN + Text Standard Hard KNN)
**Experiment**: experiments/exp_18/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Combining predicted probabilities from the optimal **Tabular Fuzzy KNN** model (`exp_13`, which improved tabular specificity by +11.76%) with optimal **MRI Standard Hard KNN** (`exp_6`, EmbedKit supervised) and **Text Standard Hard KNN** (`exp_7`, max_features=500, PCA 90%) via late-fusion soft voting in LOOCV will leverage modality-specific strengths, outperforming all prior multimodal ensembles (`exp_8` Macro-F1 = **0.7171** and `exp_16` Macro-F1 = **0.6813**).

## 2. Experimental Setup
- **Unimodal Optimal Models (Frozen)**:
  1. **Tabular Fuzzy KNN (`exp_13`)**: `MinMaxScaler` + `OneHotEncoder` + `KNeighborsRegressor(k=1, uniform, euclidean)` trained on continuous soft targets $\tilde{y} \in [0.0, 1.0]$. Outputs $\tilde{p}_{\text{tab\_fuzzy}}$.
  2. **MRI Standard Hard KNN (`exp_6`)**: `MinMaxScaler` + `EmbedKit(mode="supervised", target_dim=384)` + `KNeighborsClassifier(k=3, uniform, euclidean)` trained on hard labels $y \in \{0, 1\}$. Outputs $p_{\text{mri\_hard}}$.
  3. **Text Standard Hard KNN (`exp_7`)**: `TfidfVectorizer(max_features=500)` + `MinMaxScaler` + `PCA(90%)` + `KNeighborsClassifier(k=1, uniform, cosine)` trained on hard labels $y \in \{0, 1\}$. Outputs $p_{\text{text\_hard}}$.
- **Cohort**: $N=88$ labeled complete-case cohort for LOOCV final evaluation.
- **Late Fusion Combination**:
  - For each fold $i \in [1..88]$ in LOOCV, retrain all three unimodal models strictly on the 87 training samples and predict hybrid probability vector $\mathbf{p}_i = [\tilde{p}_{\text{tab\_fuzzy}, i}, p_{\text{mri\_hard}, i}, p_{\text{text\_hard}, i}]^T$.
  - Compute hybrid late-fusion probability:
    $$p_{\text{hybrid}, i} = w_{\text{tab}} \cdot \tilde{p}_{\text{tab\_fuzzy}, i} + w_{\text{mri}} \cdot p_{\text{mri\_hard}, i} + w_{\text{text}} \cdot p_{\text{text\_hard}, i}, \quad \text{where } \sum_{m} w_m = 1, \; w_m \ge 0$$
  - Binary decision threshold: $p_{\text{hybrid}, i} \ge 0.50 \implies \hat{y}_i = 1$.
- **Ensemble Conditions Evaluated**:
  1. `Unimodal-Tabular-Fuzzy` ($w = [1.00, 0.00, 0.00]$)
  2. `Unimodal-MRI-Hard` ($w = [0.00, 1.00, 0.00]$)
  3. `Unimodal-Text-Hard` ($w = [0.00, 0.00, 1.00]$)
  4. `Equal-Hybrid-Fusion` ($w = [0.333, 0.333, 0.333]$)
  5. `Bimodal-TabularFuzzy-TextHard` ($w = [0.50, 0.00, 0.50]$)
  6. `Bimodal-TabularFuzzy-MRIHard` ($w = [0.50, 0.50, 0.00]$)
  7. `Bimodal-TextHard-MRIHard` ($w = [0.00, 0.50, 0.50]$)
  8. `Optimal-Weighted-Hybrid` (Grid search over $w \in \Delta^2$ with step size 0.05 maximizing LOOCV Macro-F1).

## 3. File Layout for This Experiment
```
experiments/exp_18/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← hybrid unimodal retraining + Late Fusion LOOCV script
├── results/
│   ├── best_fusion_weights.json     ← optimal modality fusion weights [w_tab, w_mri, w_text]
│   ├── fusion_grid_results.csv      ← metrics across weight combinations
│   ├── loocv_metrics.json           ← LOOCV metrics across all hybrid fusion conditions
│   ├── oof_predictions.csv          ← out-of-fold probabilities per modality & ensemble
│   └── git_commit.txt               ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── confusion_matrix.png     ← LOOCV 2x2 confusion matrix of optimal hybrid fusion
    │   └── roc_curves.png           ← comparative ROC curves across modalities & fusion
    └── summary.md                   ← final report contrasting exp_18 vs exp_8 and exp_16
```

## 4. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 2-class Macro-F1 under LOOCV.
- **Baselines to Beat**:
  - `exp_8` Multimodal Standard KNN Fusion (LOOCV Macro-F1 = **0.7171**, Accuracy = **75.00%**).
  - `exp_16` Multimodal Fuzzy KNN Fusion (LOOCV Macro-F1 = **0.6813**, Accuracy = **71.59%**).
- **Secondary Metrics**: Accuracy, Sensitivity (Recall), Specificity, AUROC, Brier Score (calibration).

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for EmbedKit and PCA)
- [ ] Config and scripts saved in `scripts/`
- [ ] Fusion grid results logged to `results/fusion_grid_results.csv`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_18`.
