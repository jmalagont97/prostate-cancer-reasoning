# Experiment Design: Global Exhaustive Threshold Optimization on Decision Risk (exp_18)

**Experiment**: `experiments/exp_18/`  
**Project**: `pathology-reasoning` (CHIMERA Task 1)  
**Date**: 2026-08-18  
**Author**: Experto en Machine Learning, Creación de Agentes y Razonamiento  
**Status**: Draft  

---

## 1. Hypothesis

Replacing rule-based `DecisionTreeClassifier` discretization with **Global Exhaustive Threshold Optimization** \((\tau_1^*, \tau_2^*)\) directly on the continuous Decision Risk score \(\Omega(c_{\text{fn}}, \lambda)\) — finding the exact 2D cut-point pair that minimizes training-fold Balanced Ordinal Error (\(\text{MOE}_{\text{abs}}\)) and maximizes \(\text{F1}_{\text{macro}}\) — improves clinical confidence prediction over `exp_17` (\(\text{MOE}_{\text{abs}}^{\text{MCCV (mean)}} < 0.3949\), \(\text{F1}_{\text{macro}}^{\text{MCCV (pooled)}} > 0.3212\)).

### Null Hypothesis
Exhaustive threshold optimization on \(\Omega\) does not achieve lower average \(\text{MOE}_{\text{abs}}\) than `exp_17` (\(\text{MOE}_{\text{abs}}^{\text{MCCV (mean)}} = 0.3949\)) under 50-repeat Monte Carlo Cross-Validation.

---

## 2. Experimental Setup

### Cohort & Dataset
- **Cohort:** `usable_labeled` cases ($N=88$) from `data/chimera26/preprocessed/task1/inputs.csv`.
- **Splits:** 50 MCCV splits + 88 LOO folds from `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (Seed 42).
- **Target:** `target_confidence` $\in \{0=\text{uncertain}, 1=\text{borderline}, 2=\text{clear}\}$.

### Ground Truth Class Distribution
| Category | Ordinal Code | Count ($N=88$) | Proportion |
| :--- | :---: | :---: | :---: |
| `clear` | 2 | 56 | 63.6% |
| `borderline` | 1 | 18 | 20.5% |
| `uncertain` | 0 | 14 | 15.9% |

### Frozen Multimodal Base Models (Subtask 1.1)
Base models are trained on `target_biopsy_decision_binary` using `ConfidenceWeightedKNN` (identical to exp_4–exp_17):
- **Tabular ($T$):** 21 frozen variables (exp_5 $\tau=0.60$), MinMax, OHE, zero-fill + indicators, $k=1$, cosine, uniform, confidence_weighted.
- **MRI ($M$):** 1024-dim embedding, PCA $n_{\text{components}}=1$ (fit per fold), $k=1$, euclidean, distance, confidence_weighted.
- **Text ($X$):** TF-IDF $\text{max\_features}=2000$, numeric removal + negation protection, $k=3$, cosine, distance, confidence_weighted.

---

## 3. Mathematical Formulation of Decision Risk & Exhaustive Threshold Optimization

### 1. Decision Risk Continuous Score \(\Omega(c_{\text{fn}}, \lambda)\)
- Asymmetric Margin Risk:
  \[
  R_{\text{margen}}(\bar{p}, c_{\text{fn}}) = \frac{\min \big( \bar{p} \cdot c_{\text{fn}}, \; (1 - \bar{p}) \cdot (1 - c_{\text{fn}}) \big)}{c_{\text{fn}} \cdot (1 - c_{\text{fn}})} \quad \in [0, 1]
  \]
- Conflict Risk:
  \[
  R_{\text{conflicto}}(p_T, p_M, p_X) = 2 \cdot \text{std}(p_T, p_M, p_X) \quad \in [0, 1]
  \]
- Combined Continuous Risk Score:
  \[
  \Omega(c_{\text{fn}}, \lambda) = (1 - \lambda) \cdot R_{\text{margen}} + \lambda \cdot R_{\text{conflicto}} \quad \in [0, 1]
  \]

### 2. Global Exhaustive 2D Threshold Search \((\tau_1^*, \tau_2^*)\)
Given \(N_{\text{train}} = 70\) risk scores \(\Omega_i\) in a training fold:
- Extract all unique sorted candidate cut-points between adjacent values of \(\Omega_i\).
- Generate all valid pairs \((\tau_1, \tau_2)\) with \(0 < \tau_1 < \tau_2 < 1\).
- For each candidate pair \((\tau_1, \tau_2)\), evaluate the training predictions:
  \[
  \hat{y}_i(\tau_1, \tau_2) = \begin{cases}
  2 \quad (\text{clear}) & \text{if } \Omega_i < \tau_1 \\
  1 \quad (\text{borderline}) & \text{if } \tau_1 \le \Omega_i < \tau_2 \\
  0 \quad (\text{uncertain}) & \text{if } \Omega_i \ge \tau_2
  \end{cases}
  \]
- Select the globally optimal threshold pair \((\tau_1^*, \tau_2^*)\) on train:
  \[
  (\tau_1^*, \tau_2^*) = \arg\min_{(\tau_1, \tau_2)} \Big[ \text{MOE}_{\text{abs}}\big(y_{\text{train}}, \hat{y}(\tau_1, \tau_2)\big) - 0.001 \cdot \text{F1}_{\text{macro}}\big(y_{\text{train}}, \hat{y}(\tau_1, \tau_2)\big) \Big]
  \]

### 3. Threshold Search Modes
1. `exact_free`: Global search over all valid pairs \((\tau_1, \tau_2)\).
2. `exact_balanced_min_recall`: Global search constrained such that each class has at least $k \ge 3$ training predictions (prevents training-fold class collapse).

---

## 4. File Layout for This Experiment

```
experiments/exp_18/
├── DESIGN.md                  ← This research design document
├── IMPLEMENTATION.md          ← Implementation plan (to be generated in plan mode)
├── scripts/
│   └── run_exhaustive_threshold_experiment.py  ← Self-contained runner script
├── results/
│   ├── summary.json           ← Overall sweep summary and selected config
│   ├── evaluation_scorecard.csv  ← Detailed per-condition MCCV scorecard
│   ├── per_fold.csv           ← Per-fold metrics for selected config
│   ├── predictions_mccv.csv   ← Out-of-fold predictions across 50 MCCV splits
│   ├── predictions_loo.csv    ← Out-of-fold predictions across 88 LOO folds
│   └── confusion_matrices.json← Aggregated confusion matrices
└── reports/
    └── figures/
        ├── confusion_matrices_mccv.png
        ├── confusion_matrix_loo_selected.png
        └── confusion_matrix_loo_selected_normalized.png
```

---

## 5. Baselines

| Baseline / Model | Decision Engine | \(\text{MOE}_{\text{abs}}^{\text{MCCV (mean)}}\) | \(\text{MOE}_{\text{abs}}^{\text{MCCV (pooled)}}\) | \(\text{F1}_{\text{macro}}^{\text{MCCV (pooled)}}\) | \(\text{MOE}_{\text{abs}}^{\text{LOO}}\) | \(\text{F1}_{\text{macro}}^{\text{LOO}}\) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Always Clear** | Majority baseline | $0.5000$ | $0.5000$ | $0.2626$ | $0.5000$ | $0.2626$ |
| **Agent Baseline** | LLM reasoning (`prediction.json`) | — | — | — | $0.4788$ | $0.2962$ |
| **exp_14** | `tree_balanced` on ICI | $0.3986$ | $0.4064$ | $0.2976$ | $0.4160$ | $0.2230$ |
| **exp_15** | `p_only_balanced` on $[p_T, p_M, p_X]$ | $0.3932$ | $0.4006$ | $0.3172$ | $0.3770$ | $0.3633$ |
| **exp_16** | `reg_l2_balanced` DecisionTreeRegressor | $0.3421$ | $0.3480$ | $0.1838$ | $0.3581$ | $0.1328$ |
| **exp_17** | Decision Risk Tree (`c_fn=0.80`, `lam=0.00`) | **$0.3949$** | **$0.4035$** | **$0.3212$** | **$0.3796$** | **$0.3703$** |

---

## 6. Proposed Hyperparameter Grid (50 Conditions)

The experiment evaluates a full factorial grid of **50 conditions**:

1. **False Negative Cost (\(c_{\text{fn}}\)):** 5 values $\in \{0.20, 0.35, 0.50, 0.65, 0.80\}$ ($c_{\text{fp}} = 1 - c_{\text{fn}}$).
2. **Conflict Weight (\(\lambda\)):** 5 values $\in \{0.00, 0.25, 0.50, 0.75, 1.00\}$.
3. **Threshold Search Mode:** 2 options $\in \{\text{"exact\_free"}, \text{"exact\_balanced\_min\_recall"}\}$.

\[
5 \; (c_{\text{fn}}) \times 5 \; (\lambda) \times 2 \; (\text{threshold\_mode}) = 50 \text{ conditions}
\]

---

## 7. Evaluation Protocol & Selection Rules

### Protocol
1. **Outer Loop (MCCV Selection):** 50 stratified splits (70 train / 18 val).
2. **Inner OOF:** 3-fold CV within 70 train cases to generate $p_T, p_M, p_X$ for training without leakage.
3. **Compute Risk Score:** Calculate \(\Omega(c_{\text{fn}}, \lambda)\) for train and val sets.
4. **Exhaustive Threshold Search:** Evaluate all valid pairs \((\tau_1, \tau_2)\) on \(\Omega_{\text{train}} \to \text{target\_confidence}_{\text{train}}\) to find \((\tau_1^*, \tau_2^*)\).
5. **Evaluate Validation Set:** Apply \((\tau_1^*, \tau_2^*)\) to \(\Omega_{\text{val}}\), compute \(\text{MOE}_{\text{abs}}\) (Balanced Ordinal Error) and \(\text{F1}_{\text{macro}}\).
6. **LOO Final Audit:** Evaluate the single winning MCCV condition across 88 LOO folds to generate final out-of-fold metrics and confusion matrices.

### Primary Selection Metric (MCCV Mean & Pooled)
\[
\text{MOE}_{\text{abs}} = \frac{1}{3} \sum_{c \in \{0,1,2\}} \frac{1}{N_c} \sum_{i: y_i=c} \frac{|\hat{y}_i - y_i|}{2} \quad (\text{Lower is better})
\]

### Selection Cascade (MCCV)
1. Strictly superior to baseline (\(\text{MOE}_{\text{abs}}^{\text{pooled}} < 0.5000\)).
2. No zero recall across all 3 classes (eliminates single-class or minority-collapsed models).
3. **Primary Selection:** Minimize \(\text{MOE}_{\text{abs}}^{\text{mean}}\) and \(\text{MOE}_{\text{abs}}^{\text{pooled}}\).
4. **Secondary Tiebreaker:** Maximize \(\text{F1}_{\text{macro}}^{\text{pooled}}\).

---

## 8. Expected Results & Decision Rules

- **If Hypothesis Holds (\(\text{MOE}_{\text{abs}}^{\text{MCCV (mean)}} < 0.3949\) and \(\text{F1}_{\text{macro}}^{\text{MCCV (pooled)}} > 0.3212\)):** Global exhaustive threshold tuning outperforms rule-based trees. Adopt `exp_18` winner as the canonical Subtask 1.2 model.
- **If Hypothesis Fails to Beat exp_17:** Retain `exp_17` as the canonical Subtask 1.2 model, document comparison in `exp_18/reports/summary.md`.

---

## 9. Reproducibility Checklist

- [x] Random seeds fixed (`seed=42`)
- [x] Canonical splits file referenced (`data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`)
- [x] Dataset version / SHA-256 tracked
- [x] Out-of-fold predictions logged to `results/`
- [x] Environment frozen (`histo-DL`)
- [x] Git commit hash logged to `results/git_commit.txt` before execution

---

## 10. Next Steps

1. Approve this experiment design (`experiments/exp_18/DESIGN.md`).
2. Update `experiments/INDEX.md` to register `exp_18`.
3. In **plan mode**, request an **implementation plan** to be saved as `experiments/exp_18/IMPLEMENTATION.md`.
