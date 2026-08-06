# Implementation Plan: MRI Fuzzy KNN Representation Sweep & LOOCV (Uncertainty-Guided Soft Targets)
**Experiment**: experiments/exp_14/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_14/scripts/train.py`
This script implements the MRI representation sweep and Fuzzy KNN evaluation pipeline:

1. **Load & Align Datasets**:
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv` (1024 features).
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - MCCV Split Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).
   - Filter to labeled complete cases (`df_dec["biopsy_decision"] != "NONE"`, $N=88$).

2. **Construct Uncertainty-Guided Soft Targets ($\tilde{y}_j$)**:
   - Expert certainty weights derived from `confidence`:
     - `clear`: $c_j = 1.00$
     - `borderline`: $c_j = 0.50$
     - `uncertain`: $c_j = 0.25$
   - Continuous soft target mapping:
     - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j$
     - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j$

3. **MRI Representation Learners**:
   - **Raw**: `MinMaxScaler` onto $[0, 1]$.
   - **PCA (90%)**: `MinMaxScaler` + `PCA(n_components=0.90, random_state=42)`.
   - **EmbedKit Unsup**: `MinMaxScaler` + `EmbedKit(mode="unsupervised", target_dim=384, epochs=60, random_state=42)`.
   - **EmbedKit Sup**: `MinMaxScaler` + `EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42)` trained using soft targets $\tilde{y}_{\text{train}}$.

4. **Phase A (100 MCCV Splits Representation & Parameter Sweep)**:
   - Sweep 4 representations $\times$ 8 $k$ values ($1..21$) $\times$ 2 weights (`uniform`, `distance`) $\times$ 2 metrics (`euclidean`, `cosine`) = 128 configurations.
   - Select winning configuration maximizing mean validation Macro-F1.

5. **Phase B (LOOCV 88 Folds Final Evaluation)**:
   - Evaluate winning representation + Fuzzy KNN configuration in LOOCV over the 88 labeled complete cases.
   - Decision threshold $\tilde{p}_i \ge 0.50 \implies \hat{y}_i = 1$.
   - Compute out-of-fold Macro-F1, Accuracy, Sensitivity, Specificity, AUROC, Brier Score, and 2x2 confusion matrix.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_14/scripts/train.py
```
