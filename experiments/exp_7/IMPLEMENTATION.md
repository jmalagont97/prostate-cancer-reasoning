# Implementation Plan: Text TF-IDF KNN Representation & Vocabulary Sweep & LOOCV
**Experiment**: experiments/exp_7/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_7/scripts/train.py`
This script will implement the text vectorization, representation pre-computation, KNN sweep, and final LOOCV evaluation:
1. **Load Data**:
   - Clinical prompts text from `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Biopsy decision labels from `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Split partitions from `experiments/exp_4/results/mccv_design.csv`.
2. **Exclusion Audit & spaCy Text Preprocessing**:
   - Align patients and drop the 5 incomplete patients, leaving 190 complete cases (88 labeled, 102 unlabeled).
   - Preprocess all 190 clinical prompt texts using spaCy (`en_core_web_sm`): convert tokens to lower case (`token.lemma_.lower()`), remove punctuation/special characters (`token.is_alpha`), remove English stop words (`token.is_stop`), and apply lemmatization (`token.lemma_`).
3. **Data Pre-Computation per Split (Phase A)**:
   - For each `max_features` in `[100, 300, 500, 1000, None]`:
     - Fit `TfidfVectorizer(max_features=max_features, norm='l2')` strictly on the training partition text.
     - Transform training and validation text features.
     - Apply representation techniques to the TF-IDF vectors:
       - **Raw**: TF-IDF vectors directly.
       - **PCA**: Fit PCA (90% variance) on train TF-IDF, transform val.
       - **EmbedKit Unsupervised**: Fit `EmbedKit(mode="self_supervised", target_dim="auto", epochs=60, random_state=42)` on train TF-IDF and project val. Log resolved target dimension.
       - **EmbedKit Supervised**: Fit `EmbedKit(mode="supervised", target_dim="auto", epochs=60, random_state=42)` on train TF-IDF and project val. Log resolved target dimension.
       - **Correlation Pruning**: Greedily select non-collinear features ($|r| \le \theta$) for $\theta \in [0.70, 0.80, 0.90, 0.95]$.
4. **Grid Search Sweep over 100 Splits**:
   - Evaluate KNN parameter combinations ($k \in \{1, 3, 5, 7, 9, 11, 15, 21\}$, weights $\in \{\text{uniform}, \text{distance}\}$, metric $\in \{\text{euclidean}, \text{cosine}\}$).
   - Save average validation metrics (Macro-F1, accuracy, sensitivity, specificity) to `experiments/exp_7/results/grid_search_results.csv`.
   - Save the best configuration maximizing mean validation Macro-F1 to `experiments/exp_7/results/best_hparams.json`.
5. **Leave-One-Out Cross-Validation (LOOCV) final evaluation (Phase B)**:
   - Freeze optimal `max_features`, representation technique, and KNN parameters.
   - If an EmbedKit mode won, freeze `target_dim` to the mode (most frequent value) of the resolved target dimensions across the 100 MCCV splits of the winning configuration.
   - Execute LOOCV loop over the 88 complete cases:
     - Fit `TfidfVectorizer` and representation transformation on 87 cases, project validation case.
     - Train optimal KNN and predict label and probability.
   - Save LOOCV metrics to `experiments/exp_7/results/loocv_metrics.json` and out-of-fold predictions to `experiments/exp_7/results/loocv_predictions.csv`.
6. **Reports & Visualizations**:
   - Generate validation curves comparing vocabulary sizes and representations to `reports/figures/grid_search_curves.png`.
   - Generate `reports/figures/confusion_matrix.png` for LOOCV predictions.
   - Generate summary report `experiments/exp_7/reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_7/scripts/train.py
```
