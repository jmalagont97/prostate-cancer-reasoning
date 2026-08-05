# Implementation Plan: Multimodal Late Fusion Soft-Voting LOOCV Evaluation
**Experiment**: experiments/exp_8/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_8/scripts/train.py`
This script implements the Leave-One-Out Cross-Validation (LOOCV) loop for Late Fusion Soft Voting across the 3 optimal unimodal models:
1. **Load Data**:
   - Tabular dataset: `data/chimera26/preprocessed/task1/tabular_imputed.csv`.
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Biopsy decision targets: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - MCCV design splits: `experiments/exp_4/results/mccv_design.csv`.
2. **Alignment & Audit**:
   - Align by `patient_id` and filter to the 88 complete-case labeled cohort.
3. **LOOCV Unimodal Out-of-Fold Probability Generation (88 folds)**:
   - For each fold:
     - **Tabular Model**: Fit `MinMaxScaler` on numerical features, `OneHotEncoder` on `dre`, train KNN ($k=3$, `uniform`, `euclidean`) on 87 cases, predict probability $P_{\text{tabular}}$.
     - **MRI Model**: Fit `MinMaxScaler` and `EmbedKit(mode="supervised", target_dim=384)` on 87 cases, train KNN ($k=3$, `uniform`, `euclidean`), predict probability $P_{\text{mri}}$.
     - **Text Model**: Process text with spaCy (`en_core_web_sm`: lowercasing, stop words removal, punctuation removal, lemmatization), fit `TfidfVectorizer(max_features=500, norm='l2')` and `PCA(n_components=0.90)` on 87 cases, train KNN ($k=1$, `uniform`, `cosine`), predict probability $P_{\text{text}}$.
     - Store $P_{\text{tabular}}$, $P_{\text{mri}}$, and $P_{\text{text}}$ in an out-of-fold matrix.
4. **Late Fusion Soft-Voting Evaluation**:
   - **Equal Trimodal**: $P = \frac{1}{3} (P_{\text{tabular}} + P_{\text{mri}} + P_{\text{text}})$.
   - **Weighted Trimodal**: Grid sweep weights $w_{\text{tab}}, w_{\text{mri}}, w_{\text{text}} \in [0, 1]$ (step 0.05) subject to $\sum w = 1$.
   - **Bimodal Ablations**:
     - Tabular + Text ($w_{\text{mri}} = 0$)
     - Tabular + MRI ($w_{\text{text}} = 0$)
     - Text + MRI ($w_{\text{tab}} = 0$)
5. **Metrics & Outputs**:
   - Compute Macro-F1, accuracy, sensitivity, specificity, and AUROC for all fusion conditions.
   - Save metrics to `results/loocv_metrics.json` and predictions to `results/loocv_predictions.csv`.
   - Generate ROC curves comparing unimodal vs bimodal vs trimodal to `reports/figures/roc_curves.png`.
   - Generate confusion matrix plot to `reports/figures/confusion_matrix.png`.
   - Generate summary report `reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_8/scripts/train.py
```
