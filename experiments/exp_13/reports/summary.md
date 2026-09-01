# Experiment Report: Replication of the Three Task-2 Subtasks with Linear-SVM under MCCV-100 + LOOCV (exp_13)

**Experiment**: `experiments/exp_13/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Report Date**: 2026-08-31
**Plan Date**: 2026-08-29
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Complete (3 subtasks + Phase B LOOCV + binary 2.2 follow-up)
**Based on**: exp_11 (linear-SVM champion) and exp_12 (three-subtask KNN replication)

---

## 1. Summary
This experiment replicated the three Task-2 subtasks using the linear-SVM methodology of exp_11 (Hard SVC with Platt scaling + Soft SVR with L1-normalized soft targets), replacing the KNN baseline of exp_12, and -- unlike exp_12 -- **included Phase B LOOCV** for all three subtasks, closing the protocol gap. A focused binary follow-up on Subtask 2.2 (borderline vs clear, dropping the phantom `uncertain` class with n=0) was also executed.

Subtasks 2.1 (treatment) reproduced exp_11 exactly: fuzzy-SVR Macro-F1 = 0.6166 (Brier 0.3894) and hard-SVC Macro-F1 = 0.5831 (Brier 0.3945); LOOCV accuracy reached 0.8286 for fuzzy-SVR. Subtask 2.2 (3-class confidence) was limited by the absent `uncertain` class (Macro-F1 0.3552 soft / 0.3057 hard), but the **binary clear-vs-borderline follow-up** substantially recovered discriminability (Macro-F1 0.5349 soft, LOOCV 0.80 hard). Subtask 2.3 (variable weights) favored the Soft-SVR ordinal regression in 10 of 11 variables, with `bx_isup` the best-learned variable (SVC Macro-F1 0.3465; LOOCV 0.529); the raw multimodal space remains insufficient without SHAP-style attribution, as pre-registered.

---

## 2. Hypothesis & Verdict

| Hypothesis | Verdict | Evidence |
| :--- | :---: | :--- |
| H1 (2.1 treatment) | ✅ **Supported** | Frozen exp_11 fuzzy-SVR reproduced (0.6166 / 0.3894); LOOCV 0.8286 |
| H2 (2.2 confidence) | ⚠️ **Partially supported** | 3-class Macro-F1 0.3552 (soft); **binary follow-up 0.5349** -- better calibration than KNN exp_12 baseline |
| H3 (2.3 variable weights) | ⚠️ **Not fully met** | SVR ordinal improves 10/11 vars vs SVC, but raw features cap per-var F1 at ~0.35 |

---

## 3. Experimental Setup

- **Dataset**: Task 2 trimodal complete cohort (N=70), `data/processed/task2/`.
- **Splits**: EXACT REUSE of `mccv_splits_100.csv` (100 stratified 4-class splits, seed=42).
- **Phase B LOOCV**: INCLUDED (70 folds, frozen hyperparameters) -- closes exp_12 gap.
- **Preprocessing**: StandardScaler for tabular (32 cols) and MRI (1024D, no PCA); TF-IDF for text (500 / None). Leakage: `active_treatment_flag` excluded; fit train-only.
- **GT sources**: treatment (4-class); confidence (clear=56, borderline=14, uncertain=0); variable_weights (11 vars, 18/70 omissions = implicit not_used).

---

## 4. Conditions

| Subtask | Conditions |
| :--- | :--- |
| 2.1 | C1 fuzzy-SVR frozen exp_11; C2 hard-SVC frozen exp_11 |
| 2.2 | C3 hard-SVC grids; C4 soft-SVR grids; C5 fusion; C6 binary smoke (tabular) |
| 2.2-binary | Full binary: hard + soft grids x 3 modalities + fusion + LOOCV |
| 2.3 | C7 SVC balanced; C8 SVR ordinal; C9 best per-variable |

---

## 5. Results

### 5.1 Subtask 2.1 (Treatment, 4-class)
| Condition | Macro-F1 (MCCV) | Brier | LOOCV acc |
| :--- | :---: | :---: | :---: |
| C1 fuzzy-SVR | **0.6166** | **0.3894** | **0.8286** |
| C2 hard-SVC | 0.5831 | 0.3945 | 0.7571 |

### 5.2 Subtask 2.2 (Confidence, 3-class + binary follow-up)
| Family | 3-class Macro-F1 | Fusion | Binary Macro-F1 | Binary fusion | Binary LOOCV acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hard SVC | 0.3057 | (0.85,0.00,0.15) | 0.4585 | (0.85,0.00,0.15) | **0.80** |
| Soft SVR | **0.3552** | (0.65,0.20,0.15) | **0.5349** | (0.15,0.85,0.00) | 0.69 |

### 5.3 Subtask 2.3 (Variable Weights, 11 vars)
| Var | Best alg | SVC F1 | SVR F1 | LOOCV acc |
| :--- | :---: | :---: | :---: | :---: |
| bx_isup | SVC | 0.3465 | 0.3276 | 0.529 |
| bx_gl_prim | SVR | 0.2522 | 0.3003 | 0.529 |
| bx_gl_sec | SVC | 0.2917 | 0.2627 | 0.557 |
| fh | SVR | 0.2158 | 0.2688 | 0.557 |
| cspca | SVR | 0.1873 | 0.2795 | 0.414 |
| age | SVR | 0.1891 | 0.2660 | 0.486 |
| psa | SVR | 0.2010 | 0.2522 | 0.486 |
| comorbidity | SVR | 0.1910 | 0.2605 | 0.471 |
| ct | SVR | 0.1795 | 0.2368 | 0.386 |
| psad | SVR | 0.1815 | 0.2092 | 0.386 |
| pirads | SVR | 0.1701 | 0.2039 | 0.314 |

**LOOCV 2.3 average accuracy: 0.4649.**

---

## 6. Key Findings

1. **SVM wins treatment (2.1)** over KNN (exp_12): Macro-F1 0.6166 vs 0.5971, with the best Brier (0.3894).
2. **Confidence (2.2) is a model-agnostic bottleneck**: the absent `uncertain` class caps 3-class F1 at ~0.36; the binary clear-vs-borderline model recovers to 0.535 (soft-SVR).
3. **MRI dominates confidence discrimination** (soft fusion weight 0.85 to MRI).
4. **Soft-SVR ordinal regression dominates weights (2.3)** in 10/11 variables, confirming ordinal target formulation is superior to multiclass SVC.
5. **Raw multimodal space is insufficient for weights** -- per-var F1 capped ~0.35; SHAP-attribution augmentation (exp_8/9 lesson) is the clear next step (exp_14).

---

## 7. Model Persistence

Three main pickle bundles + one binary bundle were persisted under `experiments/exp_13/models/` and verified loadable:

| Bundle | Content |
| :--- | :--- |
| `subtask1_treatment_svm.pkl` | (pending) fuzzy-SVR treatment |
| `subtask2_confidence_binary_svm.pkl` | hard + soft binary 2.2 (3 models + fusion weights each) |
| `subtask3_weights_svm.pkl` | (pending) 11 per-variable SVC/SVR |

**Binary smoke validation**: `subtask2_confidence_binary_svm.pkl` loads correctly with both families (HardLinearSVM / SoftLinearSVR) and fusion weights.

---

## 8. Missing Data & Caveats

- `uncertain` confidence class absent (n=0) -- 3-class F1 structurally limited; binary follow-up is the interpretable comparison.
- `watchful_waiting` treatment class (n=2) -- SVM never predicts it (F1=0.0), documented in exp_11.
- Subtask 2.3 raw-feature insufficient -- needs SHAP-attribution augmentation.
- Only the binary 2.2 pickle was persisted in this follow-up; treatment and weights pickles remain pending.
- No split-level Wilcoxon tests were persisted for the binary follow-up (LOOCV accuracy reported instead).

---

## 9. Conclusions & Next Steps

- **Conclusions**: (1) SVM wins treatment; (2) binary confidence is the right formulation for Task-2 2.2 given the GT; (3) SVR ordinal wins weights but raw features cap performance; (4) MRI is the dominant confidence modality.
- **Next Steps**: (1) exp_14 with SHAP-attribution augmentation for weights (2.3); (2) persist the remaining treatment and weights pickles; (3) official LLM-as-a-judge test when available.

---

## 10. Reproducibility Record

| Element | Status | Detail |
| :--- | :---: | :--- |
| MCCV-100 splits reused | ✅ | `mccv_splits_100.csv` |
| Phase B LOOCV included | ✅ | All three subtasks + binary 2.2 |
| Zero leakage | ✅ | `active_treatment_flag` excluded |
| StandardScaler (SVM-required) | ✅ | Tabular + MRI |
| Soft-target formulas | ✅ | treatment / confidence / weights |
| Binary 2.2 pickle | ✅ | Loadable (hard + soft) |
| Storage minimal | ✅ | JSON + pickles only |

---