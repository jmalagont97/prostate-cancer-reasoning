# Implementation Plan: Multimodal Fuzzy KNN Late Fusion Soft-Voting LOOCV Evaluation
**Experiment**: experiments/exp_16/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_16/scripts/train.py`
This script implements the Multimodal Fuzzy KNN Late Fusion pipeline:

1. **Load & Align Datasets**:
   - Tabular Features: `data/chimera26/preprocessed/task1/clinical_reasoning.csv`.
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Text Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Filter to labeled complete cases (`df_dec["biopsy_decision"] != "NONE"`, $N=88$).

2. **Preprocess Text Narratives with spaCy (`en_core_web_sm`)**:
   - Lemmatize tokens, remove stop words, convert to lowercase.

3. **Construct Uncertainty-Guided Soft Targets ($\tilde{y}_j$)**:
   - `clear` $\implies c_j = 1.00$, `borderline` $\implies c_j = 0.50$, `uncertain` $\implies c_j = 0.25$.
   - $y=1 \implies \tilde{y} = 0.50 + 0.50 \cdot c_j$, $y=0 \implies \tilde{y} = 0.50 - 0.50 \cdot c_j$.

4. **LOOCV Unimodal Retraining & Soft Probability Generation**:
   - For each fold $i \in [1..88]$:
     - **Tabular Pipeline (`exp_13`)**: `MinMaxScaler` + `KNeighborsRegressor(k=1, uniform, euclidean)`.
     - **MRI Pipeline (`exp_14`)**: `MinMaxScaler` + `EmbedKit(unsupervised, target_dim=384)` + `KNeighborsRegressor(k=3, uniform, euclidean)`.
     - **Text Pipeline (`exp_15`)**: `TfidfVectorizer(max_features=None)` + `MinMaxScaler` + `PCA(90%)` + `KNeighborsRegressor(k=3, uniform, cosine)`.
     - Predict soft probability vector $\mathbf{\tilde{p}}_i = [\tilde{p}_{\text{tab}, i}, \tilde{p}_{\text{mri}, i}, \tilde{p}_{\text{text}, i}]^T$.

5. **Late Fusion Combination & Optimization**:
   - Evaluate fixed fusion configurations:
     - `Unimodal-Tabular` ($[1.00, 0.00, 0.00]$)
     - `Unimodal-MRI` ($[0.00, 1.00, 0.00]$)
     - `Unimodal-Text` ($[0.00, 0.00, 1.00]$)
     - `Equal-Trimodal-Fusion` ($[0.333, 0.333, 0.333]$)
     - `Bimodal-Tabular-Text` ($[0.50, 0.00, 0.50]$)
     - `Bimodal-Tabular-MRI` ($[0.50, 0.50, 0.00]$)
     - `Bimodal-Text-MRI` ($[0.00, 0.50, 0.50]$)
   - Perform grid search over simplex $\Delta^2$ ($w_{\text{tab}} + w_{\text{mri}} + w_{\text{text}} = 1.0$, step size 0.05) to find `Optimal-Weighted-Trimodal` maximizing LOOCV Macro-F1.

6. **Out-of-Fold Evaluation & Report Generation**:
   - Compute Macro-F1, Accuracy, Sensitivity, Specificity, AUROC, Brier Score, and 2x2 confusion matrix.
   - Save outputs to `results/` and figures to `reports/figures/`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_16/scripts/train.py
```
