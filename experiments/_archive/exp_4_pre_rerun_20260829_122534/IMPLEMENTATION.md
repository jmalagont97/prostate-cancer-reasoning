# Implementation Details: Standardized MCCV-30 Unimodal Sweep and Late Fusion Benchmark (exp_4)

**Experiment**: `experiments/exp_4/`  
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md`  
**Evaluation Standard**: `chimera_metrics.task1.compute_biopsy_decision_gate`  
**Date**: 2026-08-23  

---

## 1. Architecture & Code Structure

* **`scripts/models.py`:**
  * `TabularPreprocessor`: Leak-safe fold-isolated preprocessor with missingness indicator vectors, median/mode imputation, complete-linkage Spearman correlation hierarchical clustering ($\tau \in \{0.50, 0.60, 0.70, 0.80, \text{None}\}$), and MinMaxScaler.
  * `MRIPreprocessor`: 1024-D embedding validator and fold-isolated PCA dimensional reducer ($n_{\text{comp}} \in \{4, 8, 16, 32, 64, \text{None}\}$).
  * `TextPreprocessor`: Clinical narrative preprocessor preserving clinical negations (`no`, `not`, `without`, `absence`, `denies`, `negative`), TF-IDF vectorizer ($max\_features \in \{500, 1000, 2000, \text{None}\}$), and TruncatedSVD ($n_{\text{comp}} \in \{4, 10, 20, 40, \text{None}\}$).
  * `CustomKNN`: Dual Standard Classifier / Fuzzy Regressor KNN with metric support ($\{\text{cosine}, \text{euclidean}, \text{manhattan}\}$), kernel weights ($\{\text{uniform}, \text{distance}\}$), and compact neighbor space ($k \in \{1, 3, 5\}$).

* **BX correction:** `TabularPreprocessor` now consumes the canonical numeric `bx` column from `clinical_tabular.csv` and its fold-safe missingness indicator; the obsolete `cli_bx` field is not used.

* **`scripts/run_exp4_mccv_evaluation.py`:**
  * Canonical execution runner evaluating 180 Tabular, 144 MRI, and 480 Clinical Text configurations across 30 deterministic MCCV folds.
  * Late fusion grid sweep over $\mathbf{w} \in \Delta^2$ (step 0.05) and decision threshold $\theta \in [0.20, 0.80]$ (step 0.02).
  * Paired two-sided Wilcoxon signed-rank tests against the official ReAct baseline ($C_0$, `exp_1`).

---

## 2. Governance Compliance

1. **No External Metric Code:** All metrics ($S_{\text{gate}}$, $\text{Macro-F1}$, $F_{1\text{-yes}}$, $F_{1\text{-no}}$, $\text{Sensitivity}$, $\text{Specificity}$) are computed directly via `chimera_metrics`.
2. **Fixed Deterministic Partitions:** Partitions loaded from `data/processed/task1/mccv_splits_30.csv`.
3. **Traceability:** Outputs saved in `results/aggregate_metrics.json`, `results/split_metrics.json`, `results/split_predictions.json`, `results/optimized_configs.json`, and `results/wilcoxon_tests.json`.
