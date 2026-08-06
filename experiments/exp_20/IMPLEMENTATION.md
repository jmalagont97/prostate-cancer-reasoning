# Implementation Plan: Clinical Feature Relevance Attribution via Mode/Median Perturbation (MCCV & LOOCV)
**Experiment**: experiments/exp_20/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_20/scripts/train.py`
This script implements the leak-free Two-Phase Feature Relevance Attribution pipeline:

1. **Load Datasets & Targets**:
   - Tabular Clinical Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Clinical Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` and `weight_*` columns).
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - MCCV Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).
   - Filter to labeled complete cases ($N=88$).

2. **Map Ordinal Ground Truth Relevance Annotations (0..3)**:
   - Scale: `not_used` $\implies 0$, `noted` $\implies 1$, `important` $\implies 2$, `decisive` $\implies 3$.
   - Target columns evaluated ($j \in \{1..10\}$):
     `weight_age`, `weight_psa`, `weight_vol`, `weight_pirads`, `weight_dre`, `weight_psad`, `weight_psav`, `weight_psap`, `weight_comorbidity`, `weight_cspca`.

3. **Construct Soft Targets ($\tilde{y}_k$) for Tabular Fuzzy KNN**:
   - `clear` $\implies c_k = 1.00$, `borderline` $\implies c_k = 0.50$, `uncertain` $\implies c_k = 0.25$.
   - $y=1 \implies \tilde{y} = 0.50 + 0.50 \cdot c_k$, $y=0 \implies \tilde{y} = 0.50 - 0.50 \cdot c_k$.

4. **Phase A (100 MCCV Splits Feature-Independent Meta-Threshold Learning)**:
   - For each split $s \in [1..100]$, train Tabular Fuzzy KNN ($k=1$, uniform, euclidean) on $X_{\text{train}}$.
   - Compute baseline predictions $\tilde{p}_{\text{base}, k}$.
   - For each feature $j \in \{1..10\}$:
     - Perturb feature $j$ using training set median (continuous) or mode (categorical) $\hat{x}_{j, \text{train}}^{\text{mode/median}}$.
     - Predict perturbed soft probability $\tilde{p}_{\text{perturbed}, k}^{(j)}$.
     - Calculate displacement $\Delta p_{s, k, j} = |\tilde{p}_{\text{base}, k} - \tilde{p}_{\text{perturbed}, k}^{(j)}|$.
     - **CRITICAL**: Fit an **INDEPENDENT 1D `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)`** strictly on feature $j$'s training displacements $\Delta p_{s, k, j}$ vs `weight_*[j]` ground truth annotations to extract feature-specific cut-points $(\tau_{1, s}^{(j)}, \tau_{2, s}^{(j)}, \tau_{3, s}^{(j)})$.
   - Average cut-points over 100 splits to yield 10 independent feature-specific meta-threshold sets: $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$.

5. **Phase B (Frozen LOOCV Out-of-Fold Evaluation - 88 Folds)**:
   - Freeze the 10 meta-threshold sets $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$.
   - For each held-out test patient $i \in [1..88]$ in LOOCV, retrain Tabular Fuzzy KNN on 87 cases.
   - Compute out-of-fold feature displacement $\Delta p_{i, j}$.
   - Apply frozen meta-thresholds to predict 4-class ordinal relevance level (0..3).
   - Evaluate Spearman rank correlation ($\rho$), Accuracy, and 4-class Macro-F1 per feature.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_20/scripts/train.py
```
