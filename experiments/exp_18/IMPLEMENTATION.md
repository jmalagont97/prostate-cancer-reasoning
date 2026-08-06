# Implementation Plan: Hybrid Multimodal Late Fusion LOOCV Evaluation (Tabular Fuzzy KNN + MRI Standard Hard KNN + Text Standard Hard KNN)
**Experiment**: experiments/exp_18/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_18/scripts/train.py`
This script implements the Hybrid Multimodal Late Fusion pipeline:

1. **Load & Align Datasets**:
   - Tabular Clinical Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Clinical Reasoning Targets: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Text Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Filter to labeled complete cases (`df_dec["biopsy_decision"] != "NONE"`, $N=88$).

2. **Preprocess Text Narratives with spaCy (`en_core_web_sm`)**:
   - Lemmatize tokens, remove stop words, convert to lowercase.

3. **Construct Soft Targets ($\tilde{y}_j$) for Tabular Fuzzy KNN**:
   - `clear` $\implies c_j = 1.00$, `borderline` $\implies c_j = 0.50$, `uncertain` $\implies c_j = 0.25$.
   - $y=1 \implies \tilde{y} = 0.50 + 0.50 \cdot c_j$, $y=0 \implies \tilde{y} = 0.50 - 0.50 \cdot c_j$.

4. **LOOCV Retraining & Probability Generation**:
   - For each fold $i \in [1..88]$ in LOOCV:
     - **Tabular Fuzzy Pipeline (`exp_13`)**: `MinMaxScaler` + `OneHotEncoder` + `KNeighborsRegressor(k=1, uniform, euclidean)` trained on $\tilde{y}_{\text{train}}$. Predicts soft probability $\tilde{p}_{\text{tab\_fuzzy}, i}$.
     - **MRI Standard Hard Pipeline (`exp_6`)**: `MinMaxScaler` + `EmbedKit(supervised, target_dim=384)` + `KNeighborsClassifier(k=3, uniform, euclidean)` trained on $y_{\text{train}} \in \{0, 1\}$. Predicts hard probability $p_{\text{mri\_hard}, i} = P(y=1 | X_{\text{mri}})$.
     - **Text Standard Hard Pipeline (`exp_7`)**: `TfidfVectorizer(max_features=500)` + `MinMaxScaler` + `PCA(90%)` + `KNeighborsClassifier(k=1, uniform, cosine)` trained on $y_{\text{train}} \in \{0, 1\}$. Predicts hard probability $p_{\text{text\_hard}, i} = P(y=1 | X_{\text{text}})$.

5. **Hybrid Late Fusion Optimization**:
   - Evaluate fixed ensemble conditions and perform grid search over simplex $\Delta^2$ ($w_{\text{tab}} + w_{\text{mri}} + w_{\text{text}} = 1.0$, step size 0.05) to find `Optimal-Weighted-Hybrid` maximizing LOOCV Macro-F1.

6. **Out-of-Fold Evaluation & Reporting**:
   - Compute Macro-F1, Accuracy, Sensitivity, Specificity, AUROC, Brier Score, and 2x2 confusion matrix.
   - Save outputs to `results/` and figures to `reports/figures/`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_18/scripts/train.py
```
