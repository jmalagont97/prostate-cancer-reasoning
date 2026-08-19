# Implementation Plan: Decision Risk Theory for Clinical Confidence Prediction (exp_17)

**Experiment**: `experiments/exp_17/`  
**Design**: `experiments/exp_17/DESIGN.md`  
**Date**: 2026-08-18  
**Status**: Approved  

---

## 1. Overview & Architecture

This implementation executes `exp_17`, evaluating a 50-condition grid of Decision Risk metrics \(\Omega(c_{\text{fn}}, \lambda)\) discretized by a 1D `DecisionTreeClassifier` to predict `target_confidence` (uncertain / borderline / clear) on $N=88$ usable cases.

The script is self-contained in `experiments/exp_17/scripts/run_decision_risk_experiment.py` and uses `tmux` session 0 for persistent background execution with real-time log outputs.

---

## 2. Key Components & Implementation Details

### A. Data & Inputs
- `inputs.csv` ($195 \times 1077$), `ground_truth.csv` ($195 \times 27$), `mccv_loocv_splits.csv` ($195 \times 56$).
- Target: `target_confidence` mapped to ordinal integer: `{uncertain: 0, borderline: 1, clear: 2}`.
- Decision Target: `target_biopsy_decision_binary` $\in \{0, 1\}$.
- Base model weights: `c_clear = 1.0`, `c_borderline = 0.5`, `c_uncertain = 0.25`.

### B. Frozen Base Multimodal Ensemble (Subtask 1.1)
- **Tabular ($T$):** 21 frozen variables (exp_5), zero-fill + indicators, MinMax, OHE. `ConfidenceWeightedKNN(k=1, metric='cosine', weights='uniform', variant='confidence_weighted')`.
- **MRI ($M$):** 1024-dim embedding, PCA $n_{\text{components}}=1$ fit on train fold only. `ConfidenceWeightedKNN(k=1, metric='euclidean', weights='distance', variant='confidence_weighted')`.
- **Text ($X$):** spaCy `en_core_web_sm` preprocessing (numeric removal, negation-protected stopwords), TF-IDF $\text{max\_features}=2000$. `ConfidenceWeightedKNN(k=3, metric='cosine', weights='distance', variant='confidence_weighted')`.

### C. Inner OOF Protocol & Continuous Decision Risk Calculation
- For each split of 50 MCCV splits (70 train / 18 val):
  1. Perform inner 3-fold Stratified K-Fold on the 70 training cases to obtain out-of-fold $p_T, p_M, p_X$ for training cases.
  2. Train base KNN models on full 70-case train set; predict $p_T, p_M, p_X$ on 18 val cases.
  3. Calculate continuous decision risk \(\Omega(c_{\text{fn}}, \lambda)\):
     - \(\bar{p} = (p_T + p_M + p_X) / 3\)
     - \(\sigma = \text{std}(p_T, p_M, p_X, \text{ddof}=0)\)
     - \(R_{\text{margen}} = \frac{\min(\bar{p} c_{\text{fn}}, \; (1-\bar{p})(1-c_{\text{fn}}))}{c_{\text{fn}} (1 - c_{\text{fn}})}\)
     - \(R_{\text{conflicto}} = 2 \cdot \sigma\)
     - \(\Omega = (1 - \lambda) \cdot R_{\text{margen}} + \lambda \cdot R_{\text{conflicto}}\)

### D. 1D Decision Tree & Grid Evaluation
- `DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=5, random_state=42, class_weight=cw)` fit on \(\Omega_{\text{train}} \to \text{target\_confidence}_{\text{train}}\).
- Grid: $c_{\text{fn}} \in \{0.20, 0.35, 0.50, 0.65, 0.80\} \times \lambda \in \{0.00, 0.25, 0.50, 0.75, 1.00\} \times \text{class\_weight} \in \{\text{None}, \text{"balanced"}\}$ (50 conditions).
- Primary metric: \(\text{MOE}_{\text{abs}}\) (Balanced Ordinal Error).
- Secondary metric (tiebreaker): \(\text{F1}_{\text{macro}}\).

### E. LOO Evaluation & Output Generation
- Single winning MCCV condition evaluated on 88 LOO folds (inner 3-fold OOF inside each 87-case train fold).
- Outputs saved to `experiments/exp_17/results/`:
  - `summary.json`, `evaluation_scorecard.csv`, `per_fold.csv`, `predictions_mccv.csv`, `predictions_loo.csv`, `confusion_matrices.json`, `git_commit.txt`.
- Visualizations saved to `experiments/exp_17/reports/figures/`:
  - `confusion_matrices_mccv.png`, `confusion_matrix_loo_selected.png`, `confusion_matrix_loo_selected_normalized.png`.

---

## 3. Execution Plan

1. Create `experiments/exp_17/scripts/run_decision_risk_experiment.py`.
2. Launch script inside `tmux` session 0:
   `tmux send-keys -t 0 "python3 experiments/exp_17/scripts/run_decision_risk_experiment.py" C-m`
3. Monitor progress in real time.
