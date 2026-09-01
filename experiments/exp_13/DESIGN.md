# Experiment Design: Replication of the Three Task-2 Subtasks with Linear-SVM under MCCV-100 + LOOCV (exp_13)

**Experiment**: `experiments/exp_13/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Date**: 2026-08-29
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Draft — pending PI approval
**Based on**: exp_11 (linear-SVM champion: fuzzy-SVR Macro-F1 0.6166, Brier 0.3894) and exp_12 (three-subtask KNN replication)

---

## 1. Hypotheses

**H1 (Subtask 2.1 — Treatment, 4-class):** The frozen exp_11 fuzzy-SVR configs reproduce Macro-F1 >= 0.6166 and Brier <= 0.3894 under the shared MCCV-100 + LOOCV protocol (reproducibility check, mirroring exp_12 2.1).

**H2 (Subtask 2.2 — Clinical Confidence, 3-class):** Linear-SVM confidence models (hard SVC + soft SVR) produce **better calibration (Brier)** than the KNN baseline (exp_12 2.2: Macro-F1 0.3736), consistent with the exp_11 lesson that SVM Platt / regression probabilities are intrinsically better calibrated, despite the structural `uncertain` n=0. A secondary binary clear-vs-borderline condition is evaluated to isolate the interpretable contrast.

**H3 (Subtask 2.3 — Variable Weights, 11 x 4 ordinal):** Per-variable linear-SVM meta-models (SVC balanced / SVR ordinal) achieve comparable or better per-variable Macro-F1 than the exp_12 RF/ExtraTrees baseline (bx_isup 0.4229 best; pirads 0.1812), with the pre-registered exp_8/9 SHAP-attribution fallback if raw multimodal features remain insufficient.

---

## 2. Experimental Setup

- **Dataset**: CHIMERA Task 2 trimodal complete cohort (N=70) — `data/processed/task2/` (identical to exp_10/11/12).
- **Splits**: EXACT REUSE of `data/processed/task2/mccv_splits_100.csv` (100 stratified 4-class splits, seed=42) — enables paired Wilcoxon vs exp_11 and exp_12.
- **Phase B (LOOCV)**: INCLUDED this time — frozen Phase-A hyperparameters, retrain on 69, predict on 1 (closes the protocol gap left in exp_12).
- **GT sources**:
  - 2.1: `prostate-treatment-decision.json` (4-class).
  - 2.2: `prostate-treatment-decision-reasoning.json` -> `confidence` (clear=56, borderline=14, uncertain=0).
  - 2.3: same reasoning file -> `variable_weights` (11 vars; 18/70 omissions for bx_* vars = implicit `not_used`).
- **Preprocessing (DIFFERS from exp_12, matches exp_11)**:
  - Tabular: `StandardScaler` (replaces MinMaxScaler) — required for SVM margin.
  - MRI (1024D): `StandardScaler` fitted on train fold; no PCA (fallback PCA-64 pre-registered).
  - Text: TF-IDF (`500` for hard family; full vocabulary for soft family).
  - Leakage: `active_treatment_flag` excluded; all scalers/TF-IDF fit only on train folds.

---

## 3. Soft-Target Formulations (identical mapping as exp_12, applied to SVM)

| Subtask | Hard target | Soft/ordinal target |
| :--- | :--- | :--- |
| 2.1 Treatment | one-hot 4-class | y_k = 0.25 + 0.75c; y_j!=k = (1-y_k)/3 |
| 2.2 Confidence | one-hot 3-class | ordinal: clear=1.0, borderline=0.5, uncertain=0.25 |
| 2.3 Weights | one-hot 4-nivel per variable | ordinal: not_used=0, noted=1/3, important=2/3, decisive=1.0 |

Soft outputs are clipped to [0,1] and L1-normalized to a simplex before fusion (SVR does not guarantee sum-1).

---

## 4. Conditions per Subtask

### 4.1 Subtask 2.1 (Treatment — frozen exp_11 configs, no re-selection)

| Condition | Family | Config (from exp_11 best_hyperparams.json) | Fusion weights |
| :--- | :--- | :--- | :--- |
| C1 | Fuzzy-SVR | tab C=0.1 eps=0.05; mri C=0.1 eps=0.1; txt C=10 eps=0.05 full vocab | (0.55, 0.20, 0.25) |
| C2 | Hard-SVC | tab C=0.1; mri C=0.1; txt C=10 max_features=500 | (0.85, 0.15, 0.00) |

### 4.2 Subtask 2.2 (Confidence — SVM grids + fusion)

| Condition | Family | Grid |
| :--- | :--- | :--- |
| C3 | Hard SVC (3-class) | C in {0.1, 1.0, 10.0} x class_weight {None, 'balanced'} per modality |
| C4 | Soft SVR (ordinal) | C in {0.1, 1.0, 10.0} x epsilon {0.05, 0.1} per modality |
| C5 | **Fusion** | soft-voting w in Delta^2 (step 0.05, 231 weights) per family |
| C6 | **Binary clear-vs-borderline** (secondary) | SVC/SVR on the 2 present classes (uncertain excluded) |

### 4.3 Subtask 2.3 (Variable Weights — SVM meta-models)

| Condition | Algorithm | Per-variable |
| :--- | :--- | :--- |
| C7 | SVC (linear, probability=True) balanced | 11 independent models (4-class) |
| C8 | MultiOutputRegressor(LinearSVR) ordinal | 11 independent models (4-soft-targets) |
| C9 | **Best per-variable between C7/C8** | chosen by mean val Macro-F1 |

---

## 5. Evaluation Protocol

| Subtask | Primary metrics | Secondary metrics | Official gate |
| :--- | :--- | :--- | :--- |
| 2.1 | Macro-F1, Brier | Micro-F1, Weighted-F1, AUROC OVR, per-class F1 | `compute_treatment_decision_gate` |
| 2.2 | Brier (SVM calibration), ordinal S_conf | Macro-F1 3-class, Macro-F1 binary, per-class F1 | `compute_confidence_score` |
| 2.3 | S_weights, Decisive Factors Set-F1 | per-variable Spearman rho, per-variable Macro-F1 | `compute_variable_weights_score`, `compute_important_decisive_f1` |

Homologation: `sklearn.metrics` for Macro/Micro/Weighted-F1/AUROC; multiclass Brier implemented explicitly (same as exp_10/11).

---

## 6. Statistical Tests

1. Wilcoxon paired (two-sided, alpha=0.05) 2.1: C1 vs C2 (100 splits).
2. Wilcoxon paired exp_13-2.1 vs exp_11-H2 (cross, shared 100 splits).
3. Wilcoxon paired exp_13-2.2 (SVM) vs exp_12-2.2 (KNN) on both 3-class Brier and binary Macro-F1.
4. Per-variable Spearman rho (2.3) with p-values vs null rho=0; Bonferroni correction (alpha/11).

---

## 7. Storage Footprint & Model Persistence (Pickles)

### 7.1 JSON artifacts (minimal)
Per condition: `aggregate_metrics.json`, `split_metrics.json`, `wilcoxon_tests.json`, `best_hyperparams.json`, `version.txt`. NO prediction CSVs.

### 7.2 Persisted best models (REQUIRED, exp_12 pattern)
| Subtask | Pickle artifact | Content |
| :--- | :--- | :--- |
| 2.1 | `subtask1_treatment_svm.pkl` | Best fuzzy-SVR fusion + 3 preprocessors + fusion weights |
| 2.2 | `subtask2_confidence_svm.pkl` | Best SVC/SVR + 3 preprocessors + fusion weights |
| 2.3 | `subtask3_weights_svm.pkl` | 11 meta-models dict + preprocessors + metadata |

Each pickle loadable with the `sys.modules['__main__']` trick. Inference script `predict_case_task2_svm.py` uses them for the two official Task-2 JSONs per case.

---

## 8. Risks & Mitigations

| Risk | Prob./Sev. | Mitigation |
| :--- | :---: | :--- |
| **2.2 uncertain n=0 + SVM rigid margins** | Alta / Alta | Pre-registered: SVM never predicts `uncertain`; 3-class ordinal + binary clear-vs-borderline (C6) as interpretable secondary. |
| **2.1 watchful_waiting n=2 collapse** | Alta / Alta | Already documented in exp_11 (F1=0.0); reproduced and reported. |
| **SVC non-convergence on MRI 1024D (N~49)** | Media / Media | Fallback PCA-64 on train fold (pre-registered; exp_11 needed none). |
| **Platt probabilities poorly calibrated** | Media / Media | Fallback `decision_function` + softmax (validated in exp_11). |
| **2.3 raw-feature insufficiency** | Alta / Alta | SHAP-attribution augmentation (exp_8/9-style 23D) as fallback if C7/C8 underperform. |

---

## 9. Expected Results & Decision Rules

- **2.1 success**: Macro-F1 >= 0.6166 (non-inferiority, p >= 0.05 vs exp_11-H2).
- **2.2 success**: Brier significantly lower than exp_12 2.2 KNN baseline (p < 0.05); binary clear-vs-borderline Macro-F1 >= 0.70.
- **2.3 success**: S_weights >= 0.70 or per-variable Macro-F1 >= 0.30 for at least bx_isup, psa, psad, ct.
- **Failure/adjustment**: if 2.3 underperforms, trigger SHAP-attribution fallback and propose exp_14.

---

## 10. Reproducibility Checklist

- [ ] REUSE mccv_splits_100.csv (paired comparisons valid)
- [ ] StandardScaler for tabular and MRI (fit train-only)
- [ ] Zero leakage (active_treatment_flag excluded)
- [ ] Phase B LOOCV INCLUDED (closes protocol gap)
- [ ] Soft-target formulas documented
- [ ] GT omissions as not_used (audited)
- [ ] Metrics via chimera_metrics + sklearn homologation
- [ ] 3 pickle bundles persisted + loadable
- [ ] Storage minimal (JSON + pickles only)

---

## 11. Next Steps

1. PI approval of this DESIGN.md.
2. Generate IMPLEMENTATION.md (run_exp13_mccv.py handling 3 subtasks + LOOCV).
3. Build save_best_models.py (SVM bundles) + predict_case_task2_svm.py.
4. Execute in background; verify; commit to jtarquino_task2.
5. Update .logbook.md and .discussion.md after consensus.

---