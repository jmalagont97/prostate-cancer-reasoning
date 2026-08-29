# Implementation Plan: Replication of Composite Fuzzy ICI for Clinical Confidence (exp_7)

**Experiment**: `experiments/exp_7/`  
**Design Document**: `experiments/exp_7/DESIGN.md`  
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md`  
**Status**: Ready for Execution  

---

## 1. Scope & Execution Flow

This implementation replicates the **Composite Fuzzy ICI** architecture (historical `exp_17` / `exp_16`) under the canonical multimodal representation (`exp_2`, `exp_4`) and standardized **MCCV-30 protocol** (`data/processed/task1/mccv_splits_30.csv`).

### Pipeline Workflow:
1. **Data Ingestion (Canonical & Leak-Free)**: Load `clinical_tabular.csv`, `clinical_texts.csv`, `mri_embeddings.csv`, and `ground_truth_task1.csv` ($N=91$ labeled cohort). Include the canonical pre-biopsy `bx` field; exclude pathology-derived post-outcome fields.
2. **Unimodal Model Instantiation (Frozen exp_4)**:
   - **Tabular ($T$)**: `TabularPreprocessor(spearman_tau=None)` + `CustomKNNPredictor(k=5, metric="euclidean", weights="distance", is_fuzzy=True)`.
   - **MRI ($M$)**: `MRIPreprocessor(n_components=16)` + `CustomKNNPredictor(k=1, metric="euclidean", weights="uniform", is_fuzzy=False)`.
   - **Text ($X$)**: `TextPreprocessor(max_features=1000, n_components=None)` + `CustomKNNPredictor(k=3, metric="cosine", weights="uniform", is_fuzzy=True)`.
3. **Inner-Fold Training Probability Generation & ICI Calculation**:
   - For each fold in MCCV-30 ($N_{\text{tr}}=73, N_{\text{val}}=18$):
     - Fit unimodal preprocessors and models on the training fold.
     - Compute inner cross-validation / LOOCV probability predictions $(p_T, p_M, p_X)$ on the training fold without leakage.
     - Compute train continuous Composite ICI:
       $$\text{ICI} = 2 \cdot |\bar{p} - 0.5| \cdot (1 - 2\sigma)$$
     - Fit `DecisionTreeRegressor` (with balanced sample weights) or `DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=5, class_weight='balanced', random_state=42)` on $(\text{ICI}_{\text{train}}, y_{\text{confidence, train}})$.
4. **Validation Inference & Evaluation**:
   - Compute validation unimodal probabilities $(p_T, p_M, p_X)$ on $N_{\text{val}}=18$.
   - Compute validation $\text{ICI}_{\text{val}}$.
   - Predict confidence classes $\{\text{uncertain: 0}, \text{borderline: 1}, \text{clear: 2}\}$ using the fitted decision tree.
5. **Conditions Evaluated (MCCV-30)**:
   - **$C_0$**: Baseline Modal (`clear`).
   - **$C_1$**: Canonical Unweighted Composite Fuzzy ICI ($\bar{p} = \frac{p_T + p_M + p_X}{3}$, balanced tree).
   - **$C_2$**: Optimal Weighted Composite Fuzzy ICI ($\bar{p}_w = 0.55 p_T + 0.20 p_M + 0.25 p_X$, balanced tree).
   - **$C_3$**: Unweighted Composite Fuzzy ICI with Uniform Tree (`class_weight=None`).
6. **Metrics & Statistical Testing**:
   - Compute Macro-F1 (3 classes), Balanced Mean Ordinal Error ($\text{MOE}_{\text{abs}}$), Ordinal Agreement Score ($S_{\text{conf}}$ via `chimera_metrics.task1`), Spearman rank correlation ($\rho, p$-value), Accuracy, and Wilcoxon signed-rank tests vs $C_0$.
   - Save pooled predictions, metrics JSON/CSV, confusion matrix plots, and `version.txt`.

---

## 2. Directory & File Manifest

```
experiments/exp_7/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── models.py                  # Unimodal preprocessors, CustomKNNPredictor, ICI computation, DecisionTree Meta-fitters
│   └── run_exp7_fuzzy_ici.py      # MCCV-30 execution pipeline, live progress reporting, chimera_metrics evaluator
├── results/
│   ├── C0_Baseline_Majority/
│   ├── C1_Canonical_Fuzzy_ICI/
│   ├── C2_Optimal_Weighted_ICI/
│   ├── C3_Unweighted_UniformTree/
│   ├── aggregate_metrics.json
│   ├── summary_metrics.csv
│   ├── wilcoxon_tests.json
│   ├── split_predictions.json
│   ├── version.txt
│   └── execution.log
└── reports/
    ├── summary.md
    └── figures/
        ├── confusion_matrices.png
        └── ici_distribution.png
```
