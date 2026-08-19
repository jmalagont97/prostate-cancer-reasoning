# Implementation Plan: SHAP / Distance-Attribution Exhaustive Threshold Optimization for Clinical Relevance (exp_19)

**Experiment**: `experiments/exp_19/`  
**Design**: `experiments/exp_19/DESIGN.md`  
**Date**: 2026-08-18  
**Status**: Approved  

---

## 1. Overview & Architecture

This implementation executes `exp_19`, evaluating an 8-condition grid of feature attributions \(\psi_{i, k}\) discretized via **Global Exhaustive 3-Threshold Optimization** \((\tau_{1, k}^*, \tau_{2, k}^*, \tau_{3, k}^*)\) per variable for the 10 official clinical relevance targets on $N=88$ usable cases.

The script is self-contained in `experiments/exp_19/scripts/run_shap_relevance_experiment.py` and runs on `tmux` session 0 using the `histo-DL` environment (`/home/jmalagont/miniconda3/envs/histo-DL/bin/python`) for persistent background execution with real-time log outputs.

---

## 2. Key Components & Implementation Details

### A. Data & Targets
- `inputs.csv` ($195 \times 1077$), `ground_truth.csv` ($195 \times 27$), `mccv_loocv_splits.csv` ($195 \times 56$).
- Targets: 10 official relevance variables `target_code_weight_*` $\in \{0, 1, 2, 3\}$ (`not_used`, `noted`, `important`, `decisive`).
  Variables: `age`, `fh`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`.
- Section Targets: `target_reveal_sequence_json` ($\in 6 \text{ sections}$).

### B. Frozen Tabular Base Model (Subtask 1.1)
- 21 frozen variables (exp_5), zero-fill + indicators, MinMax, OHE.
- `ConfidenceWeightedKNN(k=1, metric='cosine', weights='uniform', variant='confidence_weighted')`.

### C. Feature Attribution Methods \(\psi_{i, k}\)
- **`knn_distance_attribution`:** Difference magnitude $|x_{i, k} - x_{\text{nn}, k}|$ to the nearest neighbor.
- **`shap_kernel`:** `shap.KernelExplainer` local attribution on output probability $p_{\text{biopsy}}$.

### D. Global Exhaustive 3-Threshold Optimizer
- `fit_exhaustive_3thresholds(psi_train_k, y_train_k, mode)`:
  - Generate candidate thresholds from sorted unique values of \(\psi_{k}\).
  - Evaluate all valid 3-threshold tuples \((\tau_1, \tau_2, \tau_3)\) with \(\tau_1 < \tau_2 < \tau_3\).
  - Categorize \(\psi_k \to \hat{c}_{i, k} \in \{0, 1, 2, 3\}\).
  - Minimize loss: \(\text{loss} = \text{MOE}_{\text{abs}}(y_{\text{train}, k}, \hat{c}_k) - 0.001 \cdot \text{F1}_{\text{macro}}(y_{\text{train}, k}, \hat{c}_k)\).
  - Vectorized NumPy execution for ultra-fast fold processing.

### E. Section Reveal Sequence Derivation
- Rule-based section activation from predicted 10 relevance weights \(\hat{\mathbf{c}}\):
  - `radiology_report`: \(\max(\text{pirads}, \text{psad}, \text{vol}, \text{cspca}) \ge 1\)
  - `laboratory_results`: \(\text{dre} \ge 1\)
  - `psa_trend`: \(\text{psa} \ge 1\)
  - `family_history`: \(\text{fh} \ge 1\)
  - `pathology_report`: \(\text{bx} \ge 1\)
  - `previous_notes`: \(\text{comorbidity} \ge 1 \text{ or } \text{age} \ge 2\)

### F. Grid Evaluation & LOO Final Audit
- Grid: 8 conditions across attribution methods $\times$ threshold modes $\times$ scaling options.
- Primary metric: \(\text{MOE}_{\text{abs}}^{\text{weights}}\) across the 10 variables.
- Secondary metric: Section \(\text{F1}_{\text{macro}}\).
- Outputs saved to `experiments/exp_19/results/` and figures saved to `experiments/exp_19/reports/figures/`.

---

## 3. Execution Plan

1. Create `experiments/exp_19/scripts/run_shap_relevance_experiment.py`.
2. Launch script inside `tmux` session 0 using `histo-DL` environment:
   `tmux send-keys -t 0 "/home/jmalagont/miniconda3/envs/histo-DL/bin/python experiments/exp_19/scripts/run_shap_relevance_experiment.py 2>&1 | tee experiments/exp_19/results/output.log" C-m`
3. Monitor progress in real time.
