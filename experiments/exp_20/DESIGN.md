# Experiment Design: Clinical Feature Relevance Attribution via Mode/Median Perturbation (MCCV & LOOCV)
**Experiment**: experiments/exp_20/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Measuring the absolute probability displacement $\Delta p_{i, j} = |\tilde{p}_{\text{base}, i} - \tilde{p}_{\text{perturbed}, i}^{(j)}|$ resulting from masking feature $j$ with its training set mode/median using Tabular Fuzzy KNN (`exp_13`), and categorizing displacement into 4 discrete ordinal relevance levels (`not_used`=0, `noted`=1, `important`=2, `decisive`=3) via Class-Weighted Decision Tree meta-thresholds $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$ learned over 100 MCCV splits, will achieve statistically significant Spearman rank correlations ($\rho$) against expert urologist annotations (`weight_*` in `clinical_reasoning.csv`) without data leakage.

## 2. Experimental Setup
- **Core Model**: Tabular Fuzzy KNN ($k^*=1$, `uniform`, `euclidean`) trained on continuous soft targets $\tilde{y}_k \in [0.0, 1.0]$.
- **Cohort**: $N=88$ labeled complete-case cohort.
- **Clinical Target Features ($j \in \{1..10\}$)**:
  `age`, `psa`, `vol`, `pirads`, `dre`, `psad`, `psav`, `psap`, `comorbidity`, `cspca`.
- **Perturbation Procedure**:
  - For categorical features (`dre`), replace feature value with training set mode $\hat{x}_{j, \text{train}}^{\text{mode}}$.
  - For continuous features (`age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`), replace feature value with training set median $\hat{x}_{j, \text{train}}^{\text{median}}$.
  - Compute displacement: $\Delta p_{i, j} = |\tilde{p}_{\text{base}, i} - \tilde{p}_{\text{perturbed}, i}^{(j)}|$.
- **Two-Phase Harness (Leak-Free & Feature-Independent)**:
  - **Phase A (100-Split MCCV Feature-Independent Meta-Threshold Learning)**:
    - For each split $s \in [1..100]$, fit Tabular Fuzzy KNN on $X_{\text{train}}$ (70% cohort).
    - Compute displacements $\Delta p_{s, k, j}$ on $X_{\text{train}}$.
    - **CRITICAL**: For **EACH INDIVIDUAL FEATURE $j \in \{1..10\}$**, train an **INDEPENDENT 1D `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)`** strictly on feature $j$'s training displacements $\Delta p_{s, k, j}$ vs `weight_*[j]` ground truth annotations to extract feature-specific cut-points $(\tau_{1, s}^{(j)}, \tau_{2, s}^{(j)}, \tau_{3, s}^{(j)})$.
    - Compute 10 independent feature-specific average meta-threshold tuples: $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$.
  - **Phase B (Frozen LOOCV Out-of-Fold Evaluation - 88 Folds)**:
    - Freeze the 10 independent meta-threshold sets $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$.
    - For each held-out test patient $i \in [1..88]$ in LOOCV, retrain Fuzzy KNN on 87 cases.
    - Compute out-of-fold feature displacement $\Delta p_{i, j}$.
    - Predict ordinal relevance level (0..3):
      - $\Delta p_{i, j} < \bar{\tau}_1^{(j)} \implies \text{not\_used}$ (0)
      - $\bar{\tau}_1^{(j)} \le \Delta p_{i, j} < \bar{\tau}_2^{(j)} \implies \text{noted}$ (1)
      - $\bar{\tau}_2^{(j)} \le \Delta p_{i, j} < \bar{\tau}_3^{(j)} \implies \text{important}$ (2)
      - $\Delta p_{i, j} \ge \bar{\tau}_3^{(j)} \implies \text{decisive}$ (3)
    - Evaluate Spearman rank correlation ($\rho$), Accuracy, and 4-class Macro-F1 per clinical feature.

## 3. File Layout for This Experiment
```
experiments/exp_20/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← feature perturbation + MCCV Decision Tree meta-thresholding + LOOCV script
├── results/
│   ├── meta_thresholds.json         ← learned feature-specific balanced cut-points \bar{\tau}_{1..3}^{(j)}
│   ├── feature_attribution_metrics.json ← per-feature Spearman \rho, F1, Accuracy metrics
│   ├── oof_feature_attributions.csv  ← patient-level displacements and predicted vs GT weights
│   └── git_commit.txt               ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── feature_displacement_distributions.png ← displacement distributions per feature
    │   └── feature_importance_bar.png             ← per-feature Spearman correlation & F1 summary
    └── summary.md                   ← final report summarizing clinical relevance alignment
```

## 4. Evaluation Protocol & Decision Rules
- **Primary Metric**: Spearman Rank Correlation ($\rho$) per feature $j$ between predicted relevance levels (0..3) and expert annotations (`weight_*`).
- **Success Threshold**: Statistically significant positive rank correlation ($p < 0.05$) across major clinical features (`psa`, `pirads`, `vol`, `dre`).
- **Secondary Metrics**: 4-class Macro-F1 and Accuracy per clinical feature.

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for Decision Tree fitting)
- [ ] Config and scripts saved in `scripts/`
- [ ] Meta-thresholds logged to `results/meta_thresholds.json`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_20`.
