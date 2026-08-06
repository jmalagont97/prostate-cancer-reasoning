# Implementation Plan: Clinical Text TF-IDF Fuzzy KNN Representation & Vocabulary Sweep (MCCV) & LOOCV Evaluation
**Experiment**: experiments/exp_15/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_15/scripts/train.py`
This script implements the text vocabulary size + representation sweep and Fuzzy KNN evaluation pipeline:

1. **Load & Align Datasets**:
   - Clinical Text Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - MCCV Split Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).
   - Filter to labeled complete cases (`df_dec["biopsy_decision"] != "NONE"`, $N=88$).

2. **Preprocess Text Narratives with spaCy (`en_core_web_sm`)**:
   - Convert text to lowercase, filter non-alphanumeric tokens (`token.is_alpha`), remove English stop words (`token.is_stop`), apply morphological lemmatization (`token.lemma_`).

3. **Construct Uncertainty-Guided Soft Targets ($\tilde{y}_j$)**:
   - Expert certainty weights derived from `confidence`:
     - `clear`: $c_j = 1.00$
     - `borderline`: $c_j = 0.50$
     - `uncertain`: $c_j = 0.25$
   - Continuous soft target mapping:
     - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j$
     - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j$

4. **TF-IDF & Text Representation Learners**:
   - Sweep `max_features` $\in [100, 300, 500, 1000, \text{None}]$.
   - **Raw**: L2-normalized TF-IDF vector directly.
   - **PCA (90%)**: `PCA(n_components=0.90, random_state=42)`.
   - **EmbedKit Unsup**: `EmbedKit(mode="unsupervised", target_dim=384, epochs=60, random_state=42)`.
   - **EmbedKit Sup**: `EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42)` trained using soft targets $\tilde{y}_{\text{train}}$.

5. **Phase A (100 MCCV Splits Vocabulary, Representation & Parameter Sweep)**:
   - Sweep 5 vocabulary sizes $\times$ 4 representations $\times$ 8 $k$ values ($1..21$) $\times$ 2 weights (`uniform`, `distance`) $\times$ 2 metrics (`euclidean`, `cosine`) = 640 configurations.
   - Select winning configuration maximizing mean validation Macro-F1.

6. **Phase B (LOOCV 88 Folds Final Evaluation)**:
   - Evaluate winning vocabulary size + representation + Fuzzy KNN configuration in LOOCV over the 88 labeled complete cases.
   - Decision threshold $\tilde{p}_i \ge 0.50 \implies \hat{y}_i = 1$.
   - Compute out-of-fold Macro-F1, Accuracy, Sensitivity, Specificity, AUROC, Brier Score, and 2x2 confusion matrix.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_15/scripts/train.py
```
