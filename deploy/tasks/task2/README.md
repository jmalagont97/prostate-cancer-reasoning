# Task 2 Production Models and Inference Suite

This module provides 100% self-contained, deployable model artifacts for **CHIMERA-Agent Challenge Task 2** (Post-Biopsy Treatment Decision & Multimodal Reasoning).

---

## Architecture & Champion Models

1. **Subtask 2.1 (`subtask1_ensemble.pkl`)**:
   - **Model**: `Subtask1FuzzySVREnsemble` (exp_17).
   - **Fusing**: Late fusion simplex ($w_{\text{tab}} = 0.50, w_{\text{mri}} = 0.05, w_{\text{txt}} = 0.45$).
   - **Neighbor Retrieval**: kNN ($k=3$) multi-modal union of closest cases with original free-text clinical reasoning retrieval.

2. **Subtask 2.2 (`subtask2_knn.pkl`)**:
   - **Model**: `Subtask2FuzzyKNNClassifier` (exp_12).
   - **Fusing**: Multimodal late fusion ($w_{\text{mri}} = 0.55, w_{\text{txt}} = 0.45$) for 3-class ordinal confidence (`clear`, `borderline`, `uncertain`).

3. **Subtask 2.3 (`subtask3_extratrees.pkl`)**:
   - **Model**: `Subtask3MultiVariableEnsemble` (exp_12).
   - **Space**: Joint 98D (Tabular 32D + MRI PCA 16D + Text TF-IDF 50D).
   - **Ensemble**: 11 balanced meta-classifiers for `['not_used', 'noted', 'important', 'decisive']`.

---

## Usage

```bash
# Train and serialize all models on 100% of data
python deploy/tasks/task2/training/train_all.py

# Verify end-to-end inference on raw cases
python deploy/tasks/task2/training/verify_models.py
```
