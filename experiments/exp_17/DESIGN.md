# Experiment Design: Decision Risk Theory for Clinical Confidence Prediction (exp_17)

**Experiment**: `experiments/exp_17/`  
**Project**: `pathology-reasoning` (CHIMERA Task 1)  
**Date**: 2026-08-18  
**Author**: Experto en Machine Learning, Creación de Agentes y Razonamiento  
**Status**: Draft  

---

## 1. Hypothesis

A continuous Decision Risk metric \(\Omega(c_{\text{fn}}, \lambda)\) — derived from the binary biopsy probabilities of the frozen multimodal ensemble (\(p_T, p_M, p_X\)) under asymmetric error costs \(c_{\text{fn}}\) and inter-modality conflict weight \(\lambda\) — discretized by a 1D `DecisionTreeClassifier`, predicts clinical diagnostic confidence (`target_confidence`) significantly better than the majority baseline (\(\text{MOE}_{\text{abs}} < 0.5000\)) and improves upon previous ensemble meta-feature approaches (\(\text{MOE}_{\text{abs}}^{\text{LOO}} < 0.3770\), \(\text{F1}_{\text{macro}}^{\text{LOO}} > 0.3633\)).

### Null Hypothesis
The decision-risk-based tree does not achieve lower \(\text{MOE}_{\text{abs}}\) than predicting the majority class (`clear`) for all cases (\(\text{MOE}_{\text{abs}} = 0.5000\)) under 50-repeat Monte Carlo Cross-Validation.

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
All base models are trained on `target_biopsy_decision_binary` using `ConfidenceWeightedKNN` (matching winning setups from exp_4–exp_12):
- **Tabular ($T$):** 21 frozen variables (exp_5 $\tau=0.60$), MinMax, OHE, zero-fill + indicators, $k=1$, cosine, uniform, confidence_weighted.
- **MRI ($M$):** 1024-dim embedding, PCA $n_{\text{components}}=1$ (fit per fold), $k=1$, euclidean, distance, confidence_weighted.
- **Text ($X$):** TF-IDF $\text{max\_features}=2000$, numeric removal + negation protection, $k=3$, cosine, distance, confidence_weighted.

---

## 3. Mathematical Formulation of Decision Risk \(\Omega(c_{\text{fn}}, \lambda)\)

### 1. Asymmetric Binary Decision Risk (\(R_{\text{margen}}\))
Given the mean decision probability \(\bar{p} = \frac{p_T + p_M + p_X}{3}\) and cost parameter \(c_{\text{fn}} \in (0, 1)\) (with \(c_{\text{fp}} = 1 - c_{\text{fn}}\)):
- Risk of No Biopsy: \(R(0 \mid \bar{p}) = \bar{p} \cdot c_{\text{fn}}\)
- Risk of Biopsy: \(R(1 \mid \bar{p}) = (1 - \bar{p}) \cdot (1 - c_{\text{fn}})\)
- Unavoidable Binary Decision Risk:
  \[
  R^*(\bar{p}, c_{\text{fn}}) = \min \Big( \bar{p} \cdot c_{\text{fn}}, \quad (1 - \bar{p}) \cdot (1 - c_{\text{fn}}) \Big)
  \]
- Normalized Margin Decision Risk (\(R_{\text{margen}} \in [0, 1]\)):
  \[
  R_{\text{margen}}(\bar{p}, c_{\text{fn}}) = \frac{\min \big( \bar{p} \cdot c_{\text{fn}}, \; (1 - \bar{p}) \cdot (1 - c_{\text{fn}}) \big)}{c_{\text{fn}} \cdot (1 - c_{\text{fn}})}
  \]

### 2. Inter-Modality Conflict Risk (\(R_{\text{conflicto}}\))
Measures disagreement across the three modalities:
\[
R_{\text{conflicto}}(p_T, p_M, p_X) = 2 \cdot \text{std}(p_T, p_M, p_X) \quad \in [0, 1]
\]
where \(\text{std}\) is population standard deviation ($\text{ddof}=0$).

### 3. Combined Continuous Risk Score \(\Omega \in [0, 1]\)
\[
\Omega(c_{\text{fn}}, \lambda) = (1 - \lambda) \cdot R_{\text{margen}}(\bar{p}, c_{\text{fn}}) + \lambda \cdot R_{\text{conflicto}}(p_T, p_M, p_X)
\]

### 4. Categorization via 1D Decision Tree
A 1D `DecisionTreeClassifier` is fit on \(\Omega(c_{\text{fn}}, \lambda) \to \text{target\_confidence}\):
- Parameters: `max_depth=2`, `max_leaf_nodes=3`, `min_samples_leaf=5`, `random_state=42`.
- Hyperparameter: `class_weight` \(\in \{\text{None}, \text{"balanced"}\}\).
- Discretizes \(\Omega\) into two decision thresholds \(\tau_1 < \tau_2\), partitioning the risk continuum into:
  \[
  \hat{y} = \begin{cases}
  \text{clear } (2) & \text{if } \Omega < \tau_1 \\
  \text{borderline } (1) & \text{if } \tau_1 \le \Omega < \tau_2 \\
  \text{uncertain } (0) & \text{if } \Omega \ge \tau_2
  \end{cases}
  \]

---

## 4. File Layout for This Experiment

```
experiments/exp_17/
├── DESIGN.md                  ← This research design document
├── IMPLEMENTATION.md          ← Implementation plan (to be generated in plan mode)
├── scripts/
│   └── run_decision_risk_experiment.py  ← Self-contained runner script
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

| Baseline Name | Description | \(\text{MOE}_{\text{abs}}\) | \(\text{F1}_{\text{macro}}\) |
| :--- | :--- | :---: | :---: |
| **Always Clear** | Majority class prediction for all cases | $0.5000$ | $0.2626$ |
| **Agent Baseline** | LLM reasoning baseline (`prediction.json`) | $0.4788$ | $0.2962$ |
| **exp_14 (tree_balanced)** | `DecisionTreeClassifier` on ICI ($1\text{D}$) | $0.4064$ (MCCV) / $0.4160$ (LOO) | $0.2976$ / $0.2230$ |
| **exp_15 (p_only_balanced)**| `DecisionTreeClassifier` on $[p_T, p_M, p_X]$ ($3\text{D}$) | $0.4006$ (MCCV) / $0.3770$ (LOO) | $0.3172$ / $0.3633$ |
| **exp_16 (reg_l2_balanced)**| `DecisionTreeRegressor` on ICI ($1\text{D}$) | $0.3480$ (MCCV) / $0.3581$ (LOO) | $0.1838$ / $0.1328$ (F1 collapsed) |

---

## 6. Proposed Hyperparameter Grid (Conditions)

The experiment evaluates a full factorial grid of **50 conditions**:

1. **False Negative Cost (\(c_{\text{fn}}\)):** 5 values $\in \{0.20, 0.35, 0.50, 0.65, 0.80\}$ (strictly $0 < c_{\text{fn}} < 1$, where $c_{\text{fp}} = 1 - c_{\text{fn}}$).
2. **Conflict Weight (\(\lambda\)):** 5 values $\in \{0.00, 0.25, 0.50, 0.75, 1.00\}$ (covering full range $[0, 1]$).
3. **Class Weighting:** 2 options $\in \{\text{None}, \text{"balanced"}\}$.

\[
5 \; (c_{\text{fn}}) \times 5 \; (\lambda) \times 2 \; (\text{class\_weight}) = 50 \text{ conditions}
\]

---

## 7. Evaluation Protocol & Selection Rules

### Protocol
1. **Outer Loop (MCCV Selection):** 50 stratified splits (70 train / 18 val).
2. **Inner OOF:** 3-fold CV within 70 train cases to generate $p_T, p_M, p_X$ for training without leakage.
3. **Train Base Models:** Fit 3 base KNNs on full 70-case train set; predict $p_T, p_M, p_X$ on 18 val cases.
4. **Compute Risk Score:** Calculate \(\Omega(c_{\text{fn}}, \lambda)\) for train and val sets.
5. **Fit Risk Tree:** Train `DecisionTreeClassifier` on \(\Omega_{\text{train}} \to \text{target\_confidence}_{\text{train}}\).
6. **Evaluate Validation Set:** Predict \(\hat{y}_{\text{val}}\), calculate \(\text{MOE}_{\text{abs}}\) (Balanced Ordinal Error) and \(\text{F1}_{\text{macro}}\).
7. **LOO Final Audit:** Evaluate the single winning MCCV condition across 88 LOO folds to generate final out-of-fold metrics and mandatory confusion matrices.

### Evaluation Policy & Metric Definitions

- **Primary Metric — Balanced Ordinal Error (\(\text{MOE}_{\text{abs}}\)):**
  Macro-averaged normalized absolute ordinal distance across all 3 classes:
  \[
  \text{MOE}_{\text{abs}} = \frac{1}{3} \sum_{c \in \{0,1,2\}} \frac{1}{N_c} \sum_{i: y_i=c} \frac{|\hat{y}_i - y_i|}{2} \quad \in [0, 1] \quad (\text{Lower is better})
  \]
  Ensures equal weighting across ground truth classes regardless of class imbalance ($63.6\%$ clear, $20.5\%$ borderline, $15.9\%$ uncertain).

- **Secondary Metric (Tiebreaker) — Macro F1-Score (\(\text{F1}_{\text{macro}}\)):**
  Unweighted mean of F1-scores across all 3 classes:
  \[
  \text{F1}_{\text{macro}} = \frac{\text{F1}_{\text{uncertain}} + \text{F1}_{\text{borderline}} + \text{F1}_{\text{clear}}}{3} \quad (\text{Higher is better})
  \]

- **Mandatory Confusion Matrix Analysis:**
  To inspect how predictions are distributed across categories and detect off-diagonal error patterns (e.g. uncertain $\leftrightarrow$ clear severe errors vs. adjacent errors), the experiment runner MUST generate:
  1. Pooled 50-split MCCV confusion matrix ($N=900$ total predictions).
  2. Out-of-fold LOO confusion matrix ($N=88$ predictions), both raw counts and row-normalized by true class.
  3. JSON records (`results/confusion_matrices.json`) and visualization figures (`reports/figures/confusion_matrices_mccv.png` and `reports/figures/confusion_matrix_loo_selected.png`).

### Selection Cascade (MCCV)
1. `valid_structure_rate == 100%` (tree produces 2 internal nodes and 3 leaves in all 50 folds).
2. \(\text{MOE}_{\text{abs}} < 0.5000\) (strictly superior to baseline).
3. No zero recall across all 3 classes (eliminates single-class or minority-collapsed models).
4. **Primary Selection:** Minimize Balanced Ordinal Error (\(\text{MOE}_{\text{abs}}\)).
5. **Secondary Tiebreaker:** Maximize \(\text{F1}_{\text{macro}}\).


---

## 8. Expected Results & Decision Rules

- **If Hypothesis Holds (\(\text{MOE}_{\text{abs}}^{\text{LOO}} < 0.3770\) and \(\text{F1}_{\text{macro}}^{\text{LOO}} > 0.3633\)):** Decision Risk Theory provides a superior, physically grounded representation of clinical uncertainty. Select `exp_17` winner as the canonical Subtask 1.2 model and transition to Subtask 1.3 (Clinical Relevance Weights).
- **If Hypothesis Fails to Beat exp_15:** Retain `exp_15` (`p_only_balanced`) as the canonical Subtask 1.2 model, document the comparison in `exp_17/reports/summary.md`, and proceed to Subtask 1.3.

---

## 9. Risks & Mitigations

1. **Hyperparameter Overfitting on $N=88$:**
   - *Mitigation:* The search space is small ($50$ conditions total), 2D continuous risk \(\Omega\) is strictly 1-dimensional for tree splits, and selection is strictly governed by 50-repeat MCCV.
2. **Extreme Class Imbalance in Minority Classes:**
   - *Mitigation:* `class_weight="balanced"` and macro-averaged metrics (\(\text{MOE}_{\text{abs}}\), \(\text{F1}_{\text{macro}}\)) prevent majority-class collapse.
3. **Information Leakage:**
   - *Mitigation:* Inner 3-fold OOF inside training sets ensures \(\Omega\) is computed on out-of-sample predictions during tree training.

---

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`seed=42`)
- [x] Canonical splits file referenced (`data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`)
- [x] Dataset version / SHA-256 tracked
- [x] Out-of-fold predictions logged to `results/`
- [x] Environment frozen
- [x] Git commit hash logged to `results/git_commit.txt` before execution

---

## 11. Next Steps

1. Review and approve this experiment design (`experiments/exp_17/DESIGN.md`).
2. Update `experiments/INDEX.md` to register `exp_17`.
3. In **plan mode**, request an **implementation plan** to be saved as `experiments/exp_17/IMPLEMENTATION.md`.
