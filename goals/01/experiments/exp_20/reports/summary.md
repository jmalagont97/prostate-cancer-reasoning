# Experiment 20 Final Report: Fast Permutation SHAP Significance Thresholding

**Experiment**: `experiments/exp_20/`  
**Date**: 2026-08-19  
**Status**: Complete  
**Verdict**: `✓ PASS` (Condition `cond_4_pvalue_desc` achieves lowest LOO Weight Ordinal Error $\text{MOE}_{\text{abs}} = 0.3000$ while preserving SOTA Section Reveal F1 = 0.5775)

---

## 1. Executive Summary
Experiment 20 evaluated non-parametric **Permutation Significance Descriptors** ($S_{i, k} = 1 - p_{i, k}$) and Z-scores ($Z_{i, k}$) derived from $B=500$ sample-level permutations of SHAP attributions to replace raw max-normalized attributions in CHIMERA Subtask 1.3.

A 5-condition sweep was conducted across 50 MCCV splits (70 train / 18 val) and 88 LOO folds:
1. **Continuous Significance Descriptors ($S_{i, k} = 1 - p_{i, k}$)** without hard binary gating (`cond_4`) improved relevance weight calibration, reducing LOO relevance weight ordinal error to **$\text{MOE}_{\text{abs}}^{\text{weights}} = 0.3000$** (vs $0.3026$ in `exp_19`) while maintaining the SOTA Section Reveal Sequence Macro F1 of **$\text{F1}_{\text{sections}}^{\text{LOO}} = 0.5775$**.
2. **Hard P-Value Gating ($p \ge 0.05 \implies 0$)** (`cond_3` and `cond_5`) caused metric collapse ($\text{F1}_{\text{sections}}^{\text{LOO}} = 0.1177$). Because variables like `fh` and `comorbidity` are annotated as `1 (noted)` in >68% of cohort cases but carry low local SHAP variance, hard p-value gating incorrectly zeroes out essential section reveal triggers.

---

## 2. Quantitative Results & Comparison with `exp_19`

### LOO Out-of-Fold Evaluation Metrics

| Condition | Descriptor Type | Gating Rule | LOO Section Reveal F1 ($\text{F1}_{\text{sections}}$) | LOO Weight MOE Abs ($\text{MOE}_{\text{abs}}^{\text{weights}}$) | LOO Weight Macro F1 | Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`exp_19` Baseline** | Raw SHAP $\phi_{i, k}$ | None | **0.5775** | 0.3026 | 0.2033 | Baseline SOTA |
| `cond_1_raw_shap` | Raw SHAP $\phi_{i, k}$ | None | **0.5775** | 0.3026 | 0.2033 | Control Réplica |
| `cond_2_zscore_nogate` | Z-score $Z_{i, k}$ | None | **0.5775** | 0.3029 | 0.2025 | Competitive |
| **`cond_4_pvalue_desc`** | **Significance $S_{i, k}$** | **None** | **0.5775** | **0.3000** | **0.2053** | **WINNER (`✓ PASS`)** |
| `cond_3_zscore_gated` | Z-score $Z_{i, k}$ | $p \ge 0.05 \implies 0$ | 0.1177 | 0.4414 | 0.0953 | ✗ REJECT (Collapse) |
| `cond_5_pvalue_gated` | Significance $S_{i, k}$ | $p \ge 0.05 \implies 0$ | 0.1177 | 0.4420 | 0.0928 | ✗ REJECT (Collapse) |

---

## 3. Scientific Discussion & Theoretical Findings

1. **Continuous vs. Hard Binary Significance:**
   Transforming raw SHAP values into continuous empirical significance descriptors $S_{i, k} = 1 - p_{i, k}$ standardizes feature attributions across clinical variables with different marginal distributions without imposing artificial binary cutoffs.
2. **Pathology of Hard P-Value Thresholding:**
   In clinical decision trees, background variables (e.g. `family_history` or `comorbidity`) are frequently noted by clinicians ($y=1$) even when their local feature contribution variance is small. Binary $p$-value gating ($p < 0.05$) erroneously classifies these consistent background factors as non-significant noise, collapsing reveal sequences to trivial baselines.

---

## 4. Reproducibility & Artifacts
- **Runner Script**: `experiments/exp_20/scripts/run_permuted_shap_experiment.py`
- **Summary Metrics**: `experiments/exp_20/results/summary.json`
- **Git Commit Hash**: Recorded in `experiments/exp_20/results/git_commit.txt`
