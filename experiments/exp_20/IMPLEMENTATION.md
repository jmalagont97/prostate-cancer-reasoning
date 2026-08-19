# Implementation Plan: Permutation SHAP Significance Thresholding (Fast Vectorized Engine)

**Experiment**: `experiments/exp_20/`  
**Target File**: `experiments/exp_20/scripts/run_permuted_shap_experiment.py`  

---

## 1. Overview
This script implements **Fast Vectorized Permutation SHAP Significance Thresholding** for CHIMERA Task 1 Subtask 1.3 (Clinical Relevance Weights & Section Reveal Sequences).

It evaluates 5 experimental conditions:
1. `cond_1_raw_shap`: Raw SHAP $\phi_{i, k}$ without gating (Baseline control matching `exp_19`).
2. `cond_2_zscore_nogate`: Z-score $Z_{i, k}$ without gating.
3. `cond_3_zscore_gated`: P-value gated ($p_{i, k} \ge 0.05 \implies 0$) + Z-score thresholding.
4. `cond_4_pvalue_desc`: Significance descriptor $S_{i, k} = 1 - p_{i, k}$.
5. `cond_5_pvalue_gated`: P-value gated ($p_{i, k} \ge 0.05 \implies 0$) + Significance descriptor $S_{i, k}$ thresholding.

---

## 2. Key Components
- **Dataset Loading & Preprocessing**: Identical to `exp_19` ($N=88$ clean cohort, 21 pruned features $T_{21}$, `ConfidenceWeightedKNN` baseline).
- **Primary SHAP Extraction**: Compute observed Kernel SHAP attributions $\Phi_{\text{obs}}$ once per training split.
- **Fast Vectorized Permutation Engine ($B=500$)**:
  Perform $B=500$ matrix shuffles directly on attribution matrix $\Phi_{\text{train}}$ in NumPy:
  $$p_{i, k} = \frac{\sum_{b=1}^{B} \mathbb{I}(|\Phi_{i, k}^{(\pi_b)}| \ge |\Phi_{i, k}^{\text{obs}}|) + 1}{B + 1}, \quad S_{i, k} = 1 - p_{i, k}, \quad Z_{i, k} = \frac{\Phi_{i, k}^{\text{obs}} - \mu_{k}^{\text{null}}}{\sigma_{k}^{\text{null}} + \epsilon}$$
- **Global 3-Threshold Grid Search**: Exhaustive 3-threshold tuple $(\tau_{1, k}^*, \tau_{2, k}^*, \tau_{3, k}^*)$ per target variable on 50 MCCV splits (70 train / 18 val).
- **88-Fold LOOCV Audit**: Evaluate out-of-fold section reveal sequence Macro F1 ($\text{F1}_{\text{sections}}^{\text{LOO}}$) and relevance weights ordinal error ($\text{MOE}_{\text{abs}}^{\text{weights}}$).

---

## 3. Real-Time Output & Artifacts
- Real-time logging with explicit `flush=True` print statements showing percentage progress per phase.
- Saves results to `experiments/exp_20/results/summary.json`.
- Records git commit hash in `experiments/exp_20/results/git_commit.txt`.
