# Experiment Design: Permutation SHAP Significance Thresholding (Fast Vectorized Engine)
**Experiment**: experiments/exp_20/  
**Project**: pathology-reasoning (prostate-cancer-reasoning)  
**Date**: 2026-08-19  
**Author**: Experto en Machine Learning, Creación de Agentes y Razonamiento  
**Status**: Draft  

---

## 1. Hypothesis
Replacing raw Kernel SHAP attribution magnitudes with vectorized non-parametric **Permutation Significance Descriptors** ($S_{i, k} = 1 - p_{i, k}$) and Z-scores ($Z_{i, k}$) to gate nuisance non-informative features ($p_{i, k} \ge 0.05 \implies \text{not\_used}$) before global 3-threshold ordinal discretization improves out-of-fold Section Reveal Sequence Macro F1 ($\text{F1}_{\text{sections}}^{\text{LOO}} > 0.5775$) over `exp_19` while reducing execution time from ~30 minutes to **< 15 seconds**.

---

## 2. Experimental Setup
- **Dataset**: `data/chimera26/preprocessed/task1/` ($N=88$ clean cohort).
  - Target 1: `target_code_weight_*` (10 clinical variables, ordinal grades 0=not_used, 1=noted, 2=important, 3=decisive).
  - Target 2: `target_reveal_sequence_json` (Ordered section reveal lists).
- **Validation Protocol**: Zero-leakage protocol matching `exp_2` / `exp_19`:
  - 50-repeat Monte Carlo Cross-Validation (MCCV, 70 train / 18 val) for threshold selection.
  - 88-fold Leave-One-Out Cross-Validation (LOOCV) for out-of-fold generalization audit.
- **Base Decision Model**: Frozen `ConfidenceWeightedKNN` ($k=1$, Cosine distance, $T_{21}$ features from `exp_5`).

---

## 3. Fast Vectorized Permutation Engine Architecture
To preserve 100% statistical rigor while eliminating the $\mathcal{O}(B \times 2^M)$ KernelSHAP recalculation bottleneck:

1. **Primary SHAP Extraction**: Compute observed Kernel SHAP attributions $\Phi_{\text{obs}} \in \mathbb{R}^{N \times M}$ on training split $X_{\text{train}}$ once per fold.
2. **Vectorized Null Permutation ($H_0$)**:
   Perform $B=500$ fast column-wise sample shuffles directly on the attribution matrix $\Phi_{\text{train}}$:
   $$\mu_{k}^{\text{null}} = \mathbb{E}_{\pi}\left[\Phi_{*, k}^{(\pi)}\right], \quad \sigma_{k}^{\text{null}} = \text{std}_{\pi}\left[\Phi_{*, k}^{(\pi)}\right]$$
3. **Empirical P-Value & Significance Descriptor**:
   For each test sample $i$ and variable $k$:
   $$p_{i, k} = \frac{\sum_{b=1}^{B} \mathbb{I}\left(|\Phi_{i, k}^{(\pi_b)}| \ge |\Phi_{i, k}^{\text{obs}}|\right) + 1}{B + 1}, \quad S_{i, k} = 1 - p_{i, k}$$
   $$Z_{i, k} = \frac{\Phi_{i, k}^{\text{obs}} - \mu_{k}^{\text{null}}}{\sigma_{k}^{\text{null}} + \epsilon}$$
4. **P-Value Gating & Global 3-Threshold Discretization**:
   - Hard Rule: If $p_{i, k} \ge 0.05 \implies \hat{y}_{i, k} = 0$ (`not_used`).
   - If $p_{i, k} < 0.05$: Discretize continuous $Z_{i, k}$ (or $S_{i, k}$) using optimal thresholds $(\tau_{1, k}^*, \tau_{2, k}^*, \tau_{3, k}^*)$ found via grid search on MCCV.

---

## 4. File Layout for This Experiment
```
experiments/exp_20/
├── DESIGN.md                  ← This file (research design)
├── IMPLEMENTATION.md          ← Implementation plan (created in plan mode after approval)
├── scripts/
│   └── run_permuted_shap_experiment.py  ← Fast vectorized runner
├── results/
│   ├── summary.json           ← Final LOO and MCCV metrics
│   └── output.log             ← Real-time log file
└── reports/
    └── summary.md             ← Formal academic report
```

---

## 5. Experimental Conditions (Grid Search)

| Condition | Descriptor Type | Gating Threshold ($p$) | Threshold Search Space |
| :--- | :--- | :--- | :--- |
| `cond_1_raw_shap` | Raw SHAP $\phi_{i, k}$ | None | Baseline `exp_19` control |
| `cond_2_zscore_nogate` | Z-score $Z_{i, k}$ | None | Continuous Z-score 3-threshold search |
| `cond_3_zscore_gated` | Z-score $Z_{i, k}$ | $p \ge 0.05 \implies 0$ | P-value gated + Z-score 3-threshold search |
| `cond_4_pvalue_desc` | Significance $S_{i, k}$ | None | Direct 3-threshold search on $S_{i, k} \in [0, 1]$ |
| `cond_5_pvalue_gated` | Significance $S_{i, k}$ | $p \ge 0.05 \implies 0$ | P-value gated + 3-threshold search on $S_{i, k}$ |

---

## 6. Evaluation Protocol & Decision Rules
- **Primary Metric**: Section Reveal Sequence Macro F1 ($\text{F1}_{\text{sections}}$).
- **Secondary Metric**: Relevance Weights Balanced Ordinal Error ($\text{MOE}_{\text{abs}}^{\text{weights}}$).
- **Success Criteria**:
  1. $\text{F1}_{\text{sections}}^{\text{LOO}} > 0.5775$ (outperforming `exp_19`).
  2. Total script execution time $< 15 \text{ seconds}$.
- **Git Traceability**: Log commit hash via `git log -1 --format="%H %s" > results/git_commit.txt`.

---

## 7. Next Steps
1. User reviews and approves this `DESIGN.md`.
2. Generate `IMPLEMENTATION.md` in plan mode for explicit user review.
3. Execute `run_permuted_shap_experiment.py` in `tmux` session 0 and report final metrics.
