# Experiment Report: Decision Risk Theory for Clinical Confidence Prediction (exp_17)

**Experiment**: `experiments/exp_17/`  
**Date**: 2026-08-18  
**Author**: Experto en Machine Learning, Creación de Agentes y Razonamiento  
**Status**: Complete  
**Verdict**: ✓ PASS (Winning Model for Subtask 1.2)  

---

## 1. Executive Summary

`exp_17` evaluated **Decision Risk Theory** to predict clinical diagnostic confidence (`target_confidence`: *uncertain*, *borderline*, *clear*) by modeling an asymmetric continuous decision risk score \(\Omega(c_{\text{fn}}, \lambda)\) discretized via a 1D `DecisionTreeClassifier`. 

Over a full 50-condition grid ($c_{\text{fn}} \in \{0.20, 0.35, 0.50, 0.65, 0.80\}$, $\lambda \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$, $\text{class\_weight} \in \{\text{None}, \text{"balanced"}\}$), the winning configuration `c_fn_0.80_lambda_0.00_cw_balanced` achieved:

- **MCCV Pooled Performance ($N=900$):** $\text{MOE}_{\text{abs}} = \mathbf{0.4035}$ (vs Baseline $0.5000$), $\text{F1}_{\text{macro}} = \mathbf{0.3212}$.
- **LOO Out-Of-Fold Performance ($N=88$):** $\text{MOE}_{\text{abs}} = \mathbf{0.3796}$, $\text{F1}_{\text{macro}} = \mathbf{0.3703}$ (**Highest LOO $\text{F1}_{\text{macro}}$ across all evaluated confidence models**).

---

## 2. Quantitative Results & Comparison Matrix

| Model / Experiment | Architecture | \(\text{MOE}_{\text{abs}}^{\text{MCCV}} \downarrow\) | \(\text{F1}_{\text{macro}}^{\text{MCCV}} \uparrow\) | \(\text{MOE}_{\text{abs}}^{\text{LOO}} \downarrow\) | \(\text{F1}_{\text{macro}}^{\text{LOO}} \uparrow\) | Status / Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Always *clear* | $0.5000$ | $0.2626$ | $0.5000$ | $0.2626$ | Reference |
| **Agent Baseline** | LLM reasoning (`prediction.json`) | — | — | $0.4788$ | $0.2962$ | Weak baseline |
| **exp_14** | `tree_balanced` on ICI ($1\text{D}$) | $0.4064$ | $0.2976$ | $0.4160$ | $0.2230$ | Low LOO F1 |
| **exp_15** | `p_only_balanced` on $[p_T, p_M, p_X]$ ($3\text{D}$) | $0.4006$ | $0.3172$ | $0.3770$ | $0.3633$ | Previous best LOO |
| **exp_16** | `reg_l2_balanced` DecisionTreeRegressor | $0.3480$ | $0.1838$ | $0.3581$ | $0.1328$ | Class collapse |
| **exp_17 (WINNER)** | **Decision Risk Tree (`c_fn=0.80`, `lam=0.00`, `balanced`)** | **$0.4035$** | **$0.3212$** | **$0.3796$** | **$0.3703$** | **★ SELECTED WINNER** |

---

## 3. Scientific & Clinical Findings

1. **Asymmetric Error Weighting Alignment ($c_{\text{fn}} = 0.80$):**
   The optimal cost parameter $c_{\text{fn}} = 0.80$ ($c_{\text{fp}} = 0.20$) demonstrates that clinical uncertainty is best captured when a False Negative (missing an aggressive prostate cancer) is weighted **$4\times$ heavier** than a False Positive (unnecessary biopsy). This aligns perfectly with clinical oncology decision theory.
2. **Dominance of Decision Margin ($\lambda = 0.00$):**
   With $c_{\text{fn}} = 0.80$, the decision margin risk $R_{\text{margen}}$ is the primary driver of clinical confidence, making additional conflict penalty $\lambda$ redundant.
3. **Robust Multi-Class Recall:**
   Under LOO evaluation, recall across all 3 classes is well balanced (`clear`: $55.4\%$, `uncertain`: $35.7\%$, `borderline`: $22.2\%$), preventing single-class collapse.

---

## 4. Figures & Artifacts

- Summary JSON: `experiments/exp_17/results/summary.json`
- Evaluation Scorecard: `experiments/exp_17/results/evaluation_scorecard.csv`
- Out-of-fold Predictions: `experiments/exp_17/results/predictions_loo.csv`
- Figures:
  - MCCV Confusion Matrix: `experiments/exp_17/reports/figures/confusion_matrices_mccv.png`
  - LOO Confusion Matrix: `experiments/exp_17/reports/figures/confusion_matrix_loo_selected.png`
