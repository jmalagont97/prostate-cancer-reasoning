# Experiment Summary: Replication of Composite Fuzzy ICI for Clinical Confidence (exp_7)

**Experiment**: `experiments/exp_7/`  
**Date**: 2026-08-24  
**Status**: Complete  
**Verdict**: ⚠️ Informative / Partial (Hypothesis Refined under Frozen exp_4 Modalities)

---

## 1. Executive Summary

Experiment `exp_7` evaluated the post-hoc **Composite Fuzzy ICI (Inconsistency & Confidence Index)** architecture for **Subtask 1.2 (Clinical Confidence Prediction: `uncertain`, `borderline`, `clear`)** using the frozen optimal unimodal models from `exp_4` across the standardized **30-split Multidimensional MCCV** protocol ($N_{\text{tr}}=73, N_{\text{val}}=18$).

All evaluations maintained zero data leakage via strict inner-split cross-fitting on the training partitions to derive out-of-fold unimodal decision probabilities ($p_T, p_M, p_X$) and 1D Decision Tree meta-thresholds ($\tau_1, \tau_2$).

---

## 2. Quantitative Results & Comparison Table

| Condition / Model | Modality Aggregation | Meta-Thresholding Strategy | Macro-F1 (3-class) $\uparrow$ | Accuracy $\uparrow$ | $\mathbf{\text{MOE}_{\text{abs}}}$ (Balanced) $\downarrow$ | $\mathbf{S_{\text{conf}}}$ Score $\uparrow$ | Spearman $\rho$ ($p$-value) | Wilcoxon vs $C_0$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0_Baseline_Majority`** | — | Constant `clear` | 0.2529 ± 0.0000 | 61.11% | 0.5000 ± 0.0000 | 0.7222 ± 0.0000 | N/A | — |
| **`C1_Canonical_Fuzzy_ICI`** | Unweighted $(p_T+p_M+p_X)/3$ | Balanced 1D Tree Regressor | 0.2187 ± 0.0920 | 29.07% | 0.3731 ± 0.0984 | 0.6093 ± 0.0898 | 0.0518 ($p=0.229$) | $p=0.9879$ |
| **`C1_clf_Fuzzy_ICI`** | Unweighted $(p_T+p_M+p_X)/3$ | Balanced 1D Tree Classifier | 0.2215 ± 0.0922 | 29.07% | 0.4481 ± 0.0934 | 0.5065 ± 0.0889 | 0.0518 ($p=0.229$) | $p=0.9786$ |
| **`C2_Optimal_Weighted_ICI`** | Late-Fusion $(0.55/0.20/0.25)$ | Balanced 1D Tree Regressor | 0.2142 ± 0.0745 | 27.78% | 0.3820 ± 0.0896 | 0.5944 ± 0.0833 | 0.0601 ($p=0.163$) | $p=0.9951$ |
| **`C3_Unweighted_UniformTree`** | Unweighted $(p_T+p_M+p_X)/3$ | Uniform 1D Tree Regressor | **0.2892 ± 0.0716** | **60.74%** | **0.4756 ± 0.0546** | **0.7269 ± 0.0546** | 0.0518 ($p=0.229$) | **$p=0.0060$** ⭐ |

---

## 3. Key Findings & Theoretical Insights

1. **Impact of Modality Discretization ($k=1$ in MRI from `exp_4`):**
   - In `exp_4`, the optimal MRI model was selected as `standard_knn_clf` ($k=1$, uniform), which outputs hard discrete probabilities $p_M \in \{0, 1\}$.
   - In historical `exp_17`, all modalities used continuous soft distance-weighted Fuzzy KNN ($k \ge 3$). The presence of discrete $\{0, 1\}$ outputs in MRI collapsed the intermodal variance $\sigma$ into polarized states, reducing the rank correlation of the composite score to $\rho = 0.0518$ ($p = 0.229$).
2. **Balanced Tree Over-Correction:**
   - Under heavy class imbalance (`clear` 63.7%, `borderline` 19.8%, `uncertain` 16.5%), fitting 1D meta-thresholds with `class_weight='balanced'` forces over-prediction of the intermediate `borderline` class (Recall: $0.80$), penalizing overall accuracy ($29.07%$) and Macro-F1 ($0.2187$).
   - Conversely, the unweighted tree (`C3`) achieves higher Macro-F1 ($0.2892$) and significantly beats the majority baseline on split-level paired tests ($p = 0.0060$).
3. **Comparison with Direct Parametric/Memory Estimators (`exp_6`):**
   - The direct scalar KDM approach (`exp_6`, Macro-F1: $0.4836$, $\text{MOE}_{\text{abs}}: 0.3436$) strongly outperforms post-hoc ICI meta-thresholding when unimodal models are frozen for decision optimization rather than confidence calibration.

---

## 4. Acceptance Gate Verification

- [x] **Gate 1 (Structural Validity):** 100% predictions strictly in `{"uncertain", "borderline", "clear"}` across all 30 validation splits.
- [x] **Gate 2 (Zero Leakage):** Meta-thresholds $\tau_1, \tau_2$ and unimodal OOF probabilities fitted exclusively on training splits via inner cross-fitting.
- [x] **Gate 3 (Comparative Benchmark):** Characterized the exact behavior of post-hoc ICI when coupled to `exp_4` frozen decision backbones.
- [ ] **Gate 4 (Target Macro-F1 $\ge 0.44$):** Not achieved with frozen `exp_4` backbones due to $k=1$ hard MRI quantization and tree threshold sensitivity. Direct KDM (`exp_6`) remains the champion for Subtask 1.2.
